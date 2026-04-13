"""
Siamese 网络预训练脚本 — 逐张地图训练版 (省内存)

核心改动:
  每次只加载 1 张地图 → 生成配对 → 训练 → 释放内存 → 下一张
  内存中永远只有 1 张地图的数据, 不会 OOM

  同时提供轻量 CNN 选项 (--lightweight), 适合 64×64 小图 + 小显存

用法:
  python train_siamese.py
  python train_siamese.py --epochs 50 --lr 1e-4 --batch-size 128
  python train_siamese.py --lightweight   # 用轻量 CNN 替代 ResNet18
  python train_siamese.py --cpu           # 强制使用 CPU
"""

import argparse
import os
import gc
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import mysql.connector
import pickle
import time

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("WARN tqdm 未安装, 使用简易进度显示。安装: pip install tqdm")


# ============================================================================
#  Siamese Network — 两种 backbone 可选
# ============================================================================

class SiameseNetwork(nn.Module):
    """ResNet18 backbone (原版)"""
    def __init__(self, feature_dim=128):
        super(SiameseNetwork, self).__init__()
        import torchvision.models as models
        resnet = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, feature_dim)

    def forward_one(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, x1, x2):
        return self.forward_one(x1), self.forward_one(x2)


class SiameseNetworkLite(nn.Module):
    """轻量 CNN backbone — 参数量 ~0.24M, 适合 64×64 小图"""
    def __init__(self, feature_dim=128):
        super(SiameseNetworkLite, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),    # 64→32
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),   # 32→16
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2), # 16→8
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),                                                            # 8→1
        )
        self.fc = nn.Linear(256, feature_dim)

    def forward_one(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, x1, x2):
        return self.forward_one(x1), self.forward_one(x2)


# ============================================================================
#  Contrastive Loss (论文公式 9)
# ============================================================================

class ContrastiveLoss(nn.Module):
    def __init__(self, dm=10.0):
        super(ContrastiveLoss, self).__init__()
        self.dm = dm

    def forward(self, f1, f2, y):
        df = torch.norm(f1 - f2, p=2, dim=1)
        loss_pos = y * df.pow(2)
        loss_neg = (1 - y) * torch.clamp(self.dm - df, min=0).pow(2)
        return 0.5 * (loss_pos + loss_neg).mean()


# ============================================================================
#  单张地图 Dataset (用完即释放)
# ============================================================================

class SingleMapDataset(Dataset):
    """只保存一张地图的图块和配对, 训练完就释放"""

    def __init__(self, images, pairs):
        """
        images: list of numpy arrays (C, H, W), uint8
        pairs:  list of (idx_i, idx_j, label)
        """
        self.images = images
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j, label = self.pairs[idx]
        img1 = torch.tensor(self.images[i], dtype=torch.float32) / 255.0
        img2 = torch.tensor(self.images[j], dtype=torch.float32) / 255.0
        return img1, img2, torch.tensor(label, dtype=torch.float32)


# ============================================================================
#  地图加载 & 配对生成 (独立函数)
# ============================================================================

def load_single_map(conn, map_id, map_size, map_data_table='image_maps'):
    """从数据库加载一张地图, 返回 numpy array 或 None"""
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT image_data FROM {map_data_table} WHERE id = %s", (int(map_id),))
        row = cursor.fetchone()
        cursor.close()

        if row is None:
            return None

        blob = row[0]
        try:
            original_map = pickle.loads(blob)
        except:
            original_map = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)

        if original_map is None:
            return None

        if len(original_map.shape) == 2:
            original_map = cv2.cvtColor(original_map, cv2.COLOR_GRAY2RGB)
        elif original_map.shape[2] == 4:
            original_map = cv2.cvtColor(original_map, cv2.COLOR_BGRA2RGB)

        if original_map.shape[0] != map_size or original_map.shape[1] != map_size:
            original_map = cv2.resize(original_map, (map_size, map_size),
                                      interpolation=cv2.INTER_LINEAR)

        return original_map.astype(np.uint8)

    except Exception as e:
        print(f"  [WARN] 地图 {map_id} 加载失败: {e}")
        return None


def build_map_dataset(sat_map, map_size, grid_size, img_size, k, lam, pairs_per_map):
    """
    把一张大地图切成网格, 生成配对, 返回 SingleMapDataset
    用完后调用者负责 del
    """
    cell_size = map_size / grid_size
    images = []
    coords = []

    for gx in range(grid_size):
        for gy in range(grid_size):
            x_s = int(gx * cell_size)
            y_s = int(gy * cell_size)
            x_e = min(int((gx + 1) * cell_size), map_size)
            y_e = min(int((gy + 1) * cell_size), map_size)

            cell = sat_map[x_s:x_e, y_s:y_e]
            cell = cv2.resize(cell, (img_size, img_size))
            cell = cell.transpose(2, 0, 1)  # (C, H, W)

            idx = len(images)
            images.append(cell)
            coords.append((idx, gx, gy))

    # 生成配对
    lam_k = lam * k
    pos_pairs = []
    neg_pairs = []

    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            _, gx1, gy1 = coords[i]
            _, gx2, gy2 = coords[j]
            dist = np.sqrt((gx1 - gx2) ** 2 + (gy1 - gy2) ** 2)
            if dist < k:
                pos_pairs.append((i, j))
            elif dist > lam_k:
                neg_pairs.append((i, j))

    half = pairs_per_map // 2
    pairs = []

    if pos_pairs:
        chosen = np.random.choice(len(pos_pairs), size=min(half, len(pos_pairs)), replace=False)
        for idx in chosen:
            i, j = pos_pairs[idx]
            pairs.append((i, j, 1.0))

    if neg_pairs:
        chosen = np.random.choice(len(neg_pairs), size=min(half, len(neg_pairs)), replace=False)
        for idx in chosen:
            i, j = neg_pairs[idx]
            pairs.append((i, j, 0.0))

    np.random.shuffle(pairs)
    return SingleMapDataset(images, pairs)


# ============================================================================
#  辅助函数
# ============================================================================

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# ============================================================================
#  训练主函数
# ============================================================================

def train(args):
    # ---- 设备选择 ----
    if args.cpu:
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"{'=' * 60}")
    print(f"[Train] Device: {device}")
    if device.type == 'cuda':
        print(f"[Train] GPU: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        free_mem = (torch.cuda.get_device_properties(0).total_memory
                    - torch.cuda.memory_reserved(0)) / 1e9
        print(f"[Train] 显存: 总计 {total_mem:.1f} GB, 可用 ~{free_mem:.1f} GB")
    print(f"{'=' * 60}")

    # ---- 数据库连接, 获取地图列表 ----
    db_config = {
        'user': args.db_user,
        'password': args.db_password,
        'host': args.db_host,
        'database': args.db_name,
        'raise_on_warnings': True,
    }

    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    cursor.execute(f"SELECT id FROM {args.map_id_table}")
    map_ids = [x[0] for x in cursor.fetchall()]
    cursor.close()
    conn.close()

    print(f"[Train] 共 {len(map_ids)} 张地图")

    if len(map_ids) == 0:
        print("[Train] ERROR: 没有地图, 检查数据库。")
        return

    # ---- 模型 ----
    if args.lightweight:
        model = SiameseNetworkLite(feature_dim=128)
        print("[Train] 使用轻量 CNN backbone")
    else:
        model = SiameseNetwork(feature_dim=128)
        print("[Train] 使用 ResNet18 backbone")

    print(f"[Train] 模型参数量: {count_parameters(model):,}")

    try:
        model = model.to(device)
        if device.type == 'cuda':
            dummy1 = torch.randn(2, 3, 64, 64, device=device)
            dummy2 = torch.randn(2, 3, 64, 64, device=device)
            with torch.no_grad():
                model(dummy1, dummy2)
            del dummy1, dummy2
            torch.cuda.empty_cache()
            print("[Train] GPU 显存检测通过 ✓")
    except RuntimeError as e:
        if 'out of memory' in str(e) or 'CUDA' in str(e):
            print(f"[Train] GPU 显存不足, 自动回退到 CPU")
            torch.cuda.empty_cache()
            device = torch.device('cpu')
            if args.lightweight:
                model = SiameseNetworkLite(feature_dim=128).to(device)
            else:
                model = SiameseNetwork(feature_dim=128).to(device)
        else:
            raise e

    criterion = ContrastiveLoss(dm=args.dm)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # ---- 加载 checkpoint (如果存在) ----
    start_epoch = 1
    best_loss = float('inf')
    if os.path.exists(args.save_path):
        try:
            state = torch.load(args.save_path, map_location=device, weights_only=False)
            if isinstance(state, dict) and 'model_state_dict' in state:
                model.load_state_dict(state['model_state_dict'])
            else:
                model.load_state_dict(state)
            print(f"[Train] 已加载已有模型: {args.save_path}")
        except:
            print(f"[Train] 已有模型加载失败, 从头训练")

    print(f"\n[Train] 配置:")
    print(f"  Epochs:          {args.epochs}")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  Learning rate:   {args.lr}")
    print(f"  地图数量:         {len(map_ids)}")
    print(f"  每地图配对数:     {args.pairs_per_map}")
    print(f"  对比损失 dm:      {args.dm}")
    print(f"  模式: 逐张地图训练 (省内存)")
    print()

    use_pin = (device.type == 'cuda')
    train_start = time.time()

    # 外层: epoch 进度条
    epoch_iter = range(start_epoch, args.epochs + 1)
    if HAS_TQDM:
        epoch_pbar = tqdm(epoch_iter, desc="总进度", unit="epoch", position=0)
    else:
        epoch_pbar = epoch_iter

    for epoch in epoch_pbar:
        model.train()
        epoch_loss = 0.0
        epoch_batches = 0
        epoch_start = time.time()

        # 每个 epoch 随机打乱地图顺序
        map_order = np.random.permutation(len(map_ids))

        # 中层: 地图进度条
        if HAS_TQDM:
            map_pbar = tqdm(enumerate(map_order),
                            total=len(map_order),
                            desc=f"Epoch {epoch:3d}",
                            unit="map",
                            position=1,
                            leave=False)
        else:
            map_pbar = enumerate(map_order)

        for map_idx_i, map_order_idx in map_pbar:
            map_id = map_ids[map_order_idx]

            # --- 1. 加载地图 ---
            t_load = time.time()
            conn = mysql.connector.connect(**db_config)
            sat_map = load_single_map(conn, map_id, args.map_size, args.map_data_table)
            conn.close()
            load_time = time.time() - t_load

            if sat_map is None:
                continue

            # --- 2. 切图 + 生成配对 ---
            t_build = time.time()
            dataset = build_map_dataset(
                sat_map, args.map_size, args.grid_size, 64,
                args.k, args.lam, args.pairs_per_map
            )
            del sat_map  # 立即释放大图
            gc.collect()
            build_time = time.time() - t_build

            if len(dataset) == 0:
                del dataset
                continue

            dataloader = DataLoader(
                dataset, batch_size=args.batch_size, shuffle=True,
                num_workers=0, drop_last=False, pin_memory=use_pin,
            )

            # --- 3. 在这张地图上训练 ---
            map_loss = 0.0
            map_batches = 0
            t_train = time.time()

            # 内层: batch 进度条
            if HAS_TQDM:
                batch_pbar = tqdm(dataloader,
                                  desc=f"  地图{map_id}",
                                  unit="batch",
                                  position=2,
                                  leave=False,
                                  ncols=90)
            else:
                batch_pbar = dataloader

            for img1, img2, labels in batch_pbar:
                img1 = img1.to(device, non_blocking=True)
                img2 = img2.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                f1, f2 = model(img1, img2)
                loss = criterion(f1, f2, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                map_loss += loss.item()
                map_batches += 1

                if HAS_TQDM:
                    batch_pbar.set_postfix(loss=f"{loss.item():.4f}")

            if HAS_TQDM:
                batch_pbar.close()

            train_time = time.time() - t_train
            epoch_loss += map_loss
            epoch_batches += map_batches

            # 更新地图进度条
            if HAS_TQDM:
                avg_so_far = epoch_loss / max(epoch_batches, 1)
                map_pbar.set_postfix(
                    avg_loss=f"{avg_so_far:.4f}",
                    load=f"{load_time:.1f}s",
                    build=f"{build_time:.1f}s",
                    train=f"{train_time:.1f}s",
                )
            else:
                if (map_idx_i + 1) % max(1, len(map_ids) // 5) == 0 or (map_idx_i + 1) == len(map_ids):
                    avg_so_far = epoch_loss / max(epoch_batches, 1)
                    print(f"  Epoch {epoch}/{args.epochs} "
                          f"地图 {map_idx_i + 1}/{len(map_ids)} "
                          f"avg_loss={avg_so_far:.4f} "
                          f"load={load_time:.1f}s build={build_time:.1f}s train={train_time:.1f}s")

            # --- 4. 释放这张地图的数据 ---
            del dataset, dataloader
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

        if HAS_TQDM:
            map_pbar.close()

        # ---- Epoch 结束 ----
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / max(epoch_batches, 1)
        elapsed_total = time.time() - train_start
        eta = elapsed_total / epoch * (args.epochs - epoch)

        # 更新外层进度条
        if HAS_TQDM:
            epoch_pbar.set_postfix(
                loss=f"{avg_loss:.6f}",
                耗时=format_time(epoch_time),
                ETA=format_time(eta),
            )

        tqdm.write(f"  ✓ Epoch {epoch:3d}/{args.epochs}  "
                   f"Loss: {avg_loss:.6f}  "
                   f"耗时: {format_time(epoch_time)}  "
                   f"总计: {format_time(elapsed_total)}  "
                   f"ETA: {format_time(eta)}") if HAS_TQDM else \
        print(f"  ✓ Epoch {epoch:3d}/{args.epochs}  "
              f"Loss: {avg_loss:.6f}  "
              f"耗时: {format_time(epoch_time)}  "
              f"总计: {format_time(elapsed_total)}  "
              f"ETA: {format_time(eta)}")

        # 保存最优
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_type': 'lightweight' if args.lightweight else 'resnet18',
                'model_state_dict': model.state_dict(),
            }, args.save_path)
            print(f"  ★ 新最优模型已保存 (loss={best_loss:.6f})")

        # 每 10 个 epoch checkpoint
        if epoch % 10 == 0:
            ckpt_path = args.save_path.replace('.pth', f'_epoch{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_type': 'lightweight' if args.lightweight else 'resnet18',
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'best_loss': best_loss,
            }, ckpt_path)
            print(f"  📦 Checkpoint: {ckpt_path}")

    # ---- 训练结束 ----
    total_time = time.time() - train_start
    torch.save({
        'model_type': 'lightweight' if args.lightweight else 'resnet18',
        'model_state_dict': model.state_dict(),
    }, args.save_path)
    print(f"\n{'=' * 60}")
    print(f"[Train] 训练完成!")
    print(f"  总耗时:   {format_time(total_time)}")
    print(f"  最优Loss: {best_loss:.6f}")
    print(f"  模型路径: {args.save_path}")
    print(f"{'=' * 60}")


# ============================================================================
#  Main
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Siamese Pre-training (逐张地图版)')

    parser.add_argument('--db-user', type=str, default=os.getenv('UAV_DB_USER', 'root'))
    parser.add_argument('--db-password', type=str, default=os.getenv('UAV_DB_PASSWORD', ''))
    parser.add_argument('--db-host', type=str, default=os.getenv('UAV_DB_HOST', 'localhost'))
    parser.add_argument('--db-name', type=str, default=os.getenv('UAV_DB_NAME', 'senmap'))
    parser.add_argument('--map-id-table', type=str, default=os.getenv('UAV_MAP_ID_TABLE', 'image_maps'))
    parser.add_argument('--map-data-table', type=str, default=os.getenv('UAV_MAP_DATA_TABLE', 'image_maps'))

    parser.add_argument('--map-size', type=int, default=3000)
    parser.add_argument('--grid-size', type=int, default=30)
    parser.add_argument('--k', type=int, default=3, help='正样本距离阈值(网格)')
    parser.add_argument('--lam', type=int, default=5, help='λ: 负样本距离 > λk')
    parser.add_argument('--pairs-per-map', type=int, default=2000)

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--dm', type=float, default=10.0, help='对比损失阈值dm')
    parser.add_argument('--lightweight', action='store_true', help='使用轻量 CNN 替代 ResNet18')
    parser.add_argument('--cpu', action='store_true', help='强制使用 CPU')
    parser.add_argument('--print-every', type=int, default=5)
    parser.add_argument('--save-path', type=str, default='siamese_model.pth')

    args = parser.parse_args()
    train(args)
