import os
import math
import glob
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchmetrics import Accuracy
from tqdm import tqdm
from PIL import Image
from einops import repeat, rearrange
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

class ClassificationDataset(Dataset):
    def __init__(self, root_dir, image_set="train", transform=None):
        self.image_paths, self.labels = [], []
        self.transform = transform
        self.classes = sorted(os.listdir(os.path.join(root_dir, image_set)))
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, image_set, class_name)
            for image_path in glob.glob(os.path.join(class_dir, '*.png')):
                self.image_paths.append(image_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index], cv2.IMREAD_GRAYSCALE)
        image = Image.fromarray(image, mode='L')
        if self.transform:
            image = self.transform(image)
        label = self.labels[index]
        return image, label

train_transform = transforms.Compose([
    transforms.ToTensor(),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
n_workers = min(4, os.cpu_count())

train_dataset = ClassificationDataset(root_dir="/kaggle/input/c03-racom", image_set="train", transform=train_transform)
val_dataset = ClassificationDataset(root_dir="/kaggle/input/c03-racom", image_set="test", transform=train_transform)

trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers)
valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)

def nearest_perfect_square(val):
    if val <= 0: return 0
    sqrt_val = math.sqrt(val)
    floor_sq = math.floor(sqrt_val) ** 2
    ceil_sq = math.ceil(sqrt_val) ** 2
    return floor_sq if abs(floor_sq - val) <= abs(ceil_sq - val) else ceil_sq

class SSMCoreBase(nn.Module):
    def __init__(self, d_inner, d_state, dt_rank, dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0, dt_init_floor=1e-4, device=None, dtype=None, **kwargs):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank
        K = 4
        perK = max(1, d_inner // K)
        x_proj_weight = [nn.Linear(perK, (self.dt_rank + self.d_state*2), bias=False, **factory_kwargs).weight for _ in range(K)]
        self.x_proj_weight = nn.Parameter(torch.stack(x_proj_weight, dim=0))
        dt_projs = [self.dt_init_fn(self.dt_rank, perK, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs) for _ in range(K)]
        self.dt_projs_weight = nn.Parameter(torch.stack([dt_proj.weight for dt_proj in dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([dt_proj.bias for dt_proj in dt_projs], dim=0))
        self.A_logs = self.A_log_init_fn(self.d_state, perK, copies=K, merge=True)
        self.Ds = self.D_init_fn(perK, copies=K, merge=True)
    
    @staticmethod
    def dt_init_fn(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init_fn(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32, device=device), "n -> d n", d=d_inner).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge: A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log); A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init_fn(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge: D = D.flatten(0, 1)
        D = nn.Parameter(D); D._no_weight_decay = True
        return D

    def forward(self, x):
        raise NotImplementedError

class HQSM(SSMCoreBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x: Tensor) -> Tensor:
        self.selective_scan = selective_scan_fn if selective_scan_fn is not None else selective_scan_ref
        B, C, H, W = x.shape
        assert H == W
        L = H * W
        K = 4
        x_interleaved = rearrange(x, 'b (p k) h w -> b (k p) h w', k=K)
        x_split = torch.chunk(x_interleaved, K, dim=1)
        xs = []
        xs.append(x_split[0].view(B, -1, L))
        xs.append(torch.transpose(x_split[1], 2, 3).contiguous().view(B, -1, L))
        xs.append(torch.flip(x_split[2].view(B, -1, L), dims=[-1]))
        xs.append(torch.flip(torch.transpose(x_split[3], 2, 3).contiguous().view(B, -1, L), dims=[-1]))
        xs = torch.stack(xs, dim=1).view(B, K, -1, L)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        out_y = self.selective_scan(xs, dts, As, Bs, Cs, Ds, z=None, delta_bias=dt_projs_bias, delta_softplus=True, return_last_state=False).view(B, K, -1, L)
        inv_y = torch.flip(out_y[:, K // 2:K], dims=[-1]).view(B, K // 2, -1, L)
        y1 = out_y[:, 0]; y2 = inv_y[:, 0]
        y3 = torch.transpose(out_y[:, 1].view(B, -1, H, W), 2, 3).contiguous().view(B, -1, L)
        y4 = torch.transpose(inv_y[:, 1].view(B, -1, H, W), 2, 3).contiguous().view(B, -1, L)
        y = torch.cat([y1, y2, y3, y4], dim=1)
        return y

class FQSM(SSMCoreBase):
    def __init__(self, d_inner, win_size=7, k_ratio=0.25, **kwargs):
        super().__init__(d_inner=d_inner, **kwargs)
        self.win_size = win_size
        self.k_ratio = k_ratio
        self.router_proj = nn.Sequential(
            nn.Linear(d_inner, d_inner // 4),
            nn.GELU(),
            nn.Linear(d_inner // 4, 1)
        )

    def _partition_windows(self, x: torch.Tensor) -> torch.Tensor:
        windows = rearrange(x, 'b c (n_h p1) (n_w p2) -> b (n_h n_w) c p1 p2', p1=self.win_size, p2=self.win_size)
        return windows
    
    def _get_window_scores(self, windows: torch.Tensor) -> torch.Tensor:
        pooled = torch.mean(windows, dim=(-2, -1))
        scores = self.router_proj(pooled)
        return scores.squeeze(-1)

    def local_scan_bchw(self, x, B, C, flip=False, column_first=False):
        if column_first:
            x = x.permute(0, 3, 2, 1, 5, 4).reshape(B, C, -1)
        else:
            x = x.permute(0, 3, 1, 2, 4, 5).reshape(B, C, -1)
        if flip:
            x = x.flip([-1])
        return x

    def local_reverse(self, x, nH, nW, wH, wW, flip=False, column_first=False):
        B, C, L = x.shape
        if flip:
            x = x.flip([-1])
        if column_first:
            x = x.view(B, C, nW, nH, wW, wH).permute(0, 1, 3, 5, 2, 4).reshape(B, C, L)
        else:
            x = x.view(B, C, nH, nW, wH, wW).permute(0, 1, 2, 4, 3, 5).reshape(B, C, L)
        return x

    def forward(self, x: Tensor) -> Tensor:
        self.selective_scan = selective_scan_fn if selective_scan_fn is not None else selective_scan_ref
        B, C, H, W = x.shape
        assert H == W
        windows = self._partition_windows(x)
        B, N, C_inner, wh, ww = windows.shape
        raw_top_k = N * self.k_ratio
        top_k = nearest_perfect_square(raw_top_k)
        top_k = min(max(top_k, 1), N)
        n = int(math.sqrt(top_k))
        router_logits = self._get_window_scores(windows)
        original_routing_weights = F.softmax(router_logits, dim=1) 
        _, selected_token_id = torch.topk(original_routing_weights, top_k, dim=-1)
        routing_weights = original_routing_weights[torch.arange(B)[:, None], selected_token_id]
        windows = windows.view(B, N, -1)
        current_state = windows[torch.arange(B)[:, None], selected_token_id]
        current_state = current_state.view(B, n, n, C, wh, ww)
        L = n * n * wh * ww
        x_interleaved = rearrange(current_state, 'b n1 n2 (p k) w1 w2 -> b n1 n2 (k p) w1 w2', k=4)
        x_split = torch.chunk(x_interleaved, 4, dim=3)
        xs =[]
        xs.append(self.local_scan_bchw(x_split[0], B, C//4, column_first=False, flip = False))
        xs.append(self.local_scan_bchw(x_split[1], B, C//4, column_first=True, flip = False))
        xs.append(self.local_scan_bchw(x_split[2], B, C//4, column_first=False, flip = True))
        xs.append(self.local_scan_bchw(x_split[3], B, C//4, column_first=True, flip = True))
        xs = torch.stack(xs, dim=1).view(B, 4, -1, L)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, 4, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, 4, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, 4, -1, L)
        Cs = Cs.float().view(B, 4, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        out_y = self.selective_scan(xs, dts, As, Bs, Cs, Ds, z=None, delta_bias=dt_projs_bias, delta_softplus=True, return_last_state=False).view(B, 4, -1, L)
        y_split = torch.chunk(out_y, 4, dim=1)
        ys =[]
        ys.append(self.local_reverse(y_split[0].view(B,-1,L), n, n, wh, ww, column_first=False, flip = False))
        ys.append(self.local_reverse(y_split[1].view(B,-1,L), n, n, wh, ww, column_first=True, flip = False))
        ys.append(self.local_reverse(y_split[2].view(B,-1,L), n, n, wh, ww, column_first=False, flip = True))
        ys.append(self.local_reverse(y_split[3].view(B,-1,L), n, n, wh, ww, column_first=True, flip = True))
        y = torch.cat(ys, dim=1)
        y = y.view(B, C, n * n, wh * ww).permute(0,2,1,3).reshape(B, top_k, -1)
        current_state = y * routing_weights[:, :, None]
        residual_x = windows * original_routing_weights[:, :, None]
        residual_x[torch.arange(B)[:, None], selected_token_id] = current_state
        new_inputs = windows + residual_x
        return new_inputs.view(B, C, -1)

class AdaptivePooling(nn.Module):
    def __init__(self, dim, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1, dim, 1, 1) * p)
        self.eps = eps

    def forward(self, x):
        p = torch.clamp(self.p, min=1.0 + self.eps, max=50.0) 
        x_pow = x.clamp(min=self.eps).pow(p)
        avg = F.avg_pool2d(x_pow, (x.size(-2), x.size(-1)))
        return avg.pow(1.0 / p)
    
class AGCA(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        d = max(int(dim // reduction), 8)
        self.pool = AdaptivePooling(dim=dim, p=3.0)
        self.fc = nn.Sequential(
            nn.Linear(dim, d, bias=False),
            nn.BatchNorm1d(d),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Linear(d, dim)
        self.fc2 = nn.Linear(d, dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        U = x1 + x2 
        s = self.pool(U).flatten(1)
        z = self.fc(s)
        a1 = self.fc1(z)
        a2 = self.fc2(z)
        a = torch.stack([a1, a2], dim=1)
        a = self.softmax(a)
        w1 = a[:, 0, :].view(B, C, 1, 1)
        w2 = a[:, 1, :].view(B, C, 1, 1)
        out = w1 * x1 + w2 * x2
        return out

class HVSSB(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2, dt_rank="auto", dropout=0., conv_bias=True, bias=False, win_size=7, k_ratio=0.25, device=None, dtype=None, **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(self.d_inner, self.d_inner, kernel_size=d_conv, padding=(d_conv-1)//2, groups=self.d_inner, bias=conv_bias, **factory_kwargs)
        self.act = nn.SiLU()
        self.core_global = HQSM(d_inner=self.d_inner, d_state=d_state, dt_rank=self.dt_rank, device=device, dtype=dtype, **kwargs)
        self.core_sparse = FQSM(d_inner=self.d_inner, d_state=d_state, dt_rank=self.dt_rank, win_size=win_size, k_ratio=k_ratio, device=device, dtype=dtype, **kwargs)
        self.fusion = AGCAêM(dim=self.d_inner, reduction=8)
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    def forward(self, x: Tensor):
        x = x.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)
        x_conv = x_inner.permute(0, 3, 1, 2).contiguous()
        x_conv = self.act(self.conv2d(x_conv)) 
        y_global_flat = self.core_global(x_conv) 
        y_sparse_flat = self.core_sparse(x_conv)
        y_global = y_global_flat.view(B, self.d_inner, H, W)
        y_sparse = y_sparse_flat.view(B, self.d_inner, H, W)
        y_fused = self.fusion(y_global, y_sparse)
        y = y_fused.permute(0, 2, 3, 1).contiguous()
        y = self.out_norm(y)
        y = y * F.silu(z) 
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out.permute(0, 3, 1, 2).contiguous()

class CFE(nn.Module):
    def __init__(self, in_c=1, out_c=16):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32, bias=False),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, out_c, kernel_size=3, stride=2, padding=1, groups=out_c, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class HyVSSNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_in = CFE(in_c=1, out_c=16)
        self.stage1 = HVSSB(d_model=16, win_size=7 , k_ratio=0.25)
        self.down1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False, groups=16),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False, groups=32),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.stage2 = HVSSB(d_model=32, win_size=4, k_ratio=0.5)
        self.down2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False, groups=32),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False, groups=64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(64, 12),
        )

    def forward(self, x):
        x = self.block_in(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        out = self.head(x)
        return out

model = HyVSSNet().to(device)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of parameters: {total_params}")
print(f"Number of trainable parameters: {trainable_params}")

n_eps = 60
lr = 1e-3
scheduler_step_size = 20
scheduler_gamma = 0.2
num_classes = 12

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count

def evaluate(model, dataloader, criterion, device, current_num_classes):
    model.eval()
    val_loss_meter = AverageMeter()
    acc_metric = Accuracy(task="multiclass", num_classes=current_num_classes, average='micro').to(device)
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Validating"):
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            val_loss_meter.update(loss.item(), n=y.size(0))
            preds = torch.argmax(y_hat, dim=1)
            acc_metric.update(preds, y)
    return val_loss_meter.avg, acc_metric.compute().item()

start_epoch = 1
best_val_acc = 0.0
best_epoch_val = 0

for ep in range(start_epoch, n_eps + 1):
    model.train()
    train_loss_meter = AverageMeter()
    acc_metric = Accuracy(task="multiclass", num_classes=num_classes, average='micro').to(device)
    with tqdm(trainloader, desc=f"Training Epoch {ep}/{n_eps}", unit="batch") as tepoch:
        for x, y in tepoch:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss_meter.update(loss.item(), n=y.size(0))
            preds = torch.argmax(y_hat, dim=1)
            acc_metric.update(preds, y)
            tepoch.set_postfix(loss=train_loss_meter.avg, acc=acc_metric.compute().item())
    
    train_acc = acc_metric.compute().item()
    val_loss, val_acc = evaluate(model, valloader, criterion, device, num_classes)

    print(f"\nEpoch {ep}/{n_eps} | LR {scheduler.get_last_lr()[0]:.6f}")
    print(f"Train: loss={train_loss_meter.avg:.4f}, acc={train_acc:.4f}")
    print(f"Val  : loss={val_loss:.4f}, acc={val_acc:.4f}")
    scheduler.step()

    ckpt_path = f"epoch_{ep}_ValAcc_{val_acc:.4f}.pt"
    torch.save({
        'epoch': ep,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'current_val_accuracy': val_acc,
        'train_loss': train_loss_meter.avg,
        'val_loss': val_loss,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch_val,
    }, ckpt_path)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch_val = ep
        torch.save({'epoch': ep, 'model_state_dict': model.state_dict()}, 'best_model_state.pt')
        torch.save({
            'epoch': ep,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_acc': best_val_acc,
            'best_epoch': best_epoch_val,
        }, 'best_checkpoint.pt')

print(f"Done. Best Val Acc: {best_val_acc:.4f} @ epoch {best_epoch_val}")
