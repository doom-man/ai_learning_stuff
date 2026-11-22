"""在相同的扫雷局面上比较一层、两层隐藏层，运行方式与 minesweeper.py 相同。"""

from statistics import mean

import torch
from torch import nn
from torch.nn import functional as F

from minesweeper import make_batch


ARCHITECTURES = {
    "单隐藏层": (256,),
    "双隐藏层": (256, 128),
    "一层 281": (281,),  # 参数量接近原来的两层模型。
    "加宽双隐藏层": (256, 256),  # 检查第二层收窄到 128 是否影响了结果。
}
CHECKPOINTS = (100, 500, 2000)
SEEDS = (11, 22, 33, 44, 55)
TEST_SIZE = 8192


def make_model(widths, seed):
    # 相同的种子让两种以 256 开头的网络拥有完全相同的第一层初始权重。
    torch.manual_seed(seed)
    layers = []
    inputs = 810
    for width in widths:
        layers.extend([nn.Linear(inputs, width), nn.ReLU()])
        inputs = width
    layers.append(nn.Linear(inputs, 81))
    return nn.Sequential(*layers)


def make_test_set(size):
    torch.manual_seed(71003)
    batches = []
    remaining = size
    while remaining:
        x, mines, unknown = make_batch(min(256, remaining))
        active = (unknown & ~mines).any(dim=1)
        if active.any():
            batches.append((x[active], mines[active], unknown[active]))
            remaining -= int(active.sum())
    return batches


@torch.no_grad()
def evaluate(model, test_batches):
    hits = boards = cells = 0
    loss_sum = 0.0
    for x, mines, unknown in test_batches:
        logits = model(x)
        loss_sum += F.binary_cross_entropy_with_logits(
            logits[unknown], mines[unknown].float(), reduction="sum"
        ).item()
        choices = logits.sigmoid().masked_fill(~unknown, float("inf")).argmin(dim=1)
        hits += mines.gather(1, choices[:, None]).sum().item()
        boards += len(x)
        cells += unknown.sum().item()
    return hits / boards, loss_sum / cells


def main():
    torch.set_num_threads(1)
    test_batches = make_test_set(TEST_SIZE)
    random_risk = torch.cat([
        (mines & unknown).sum(dim=1) / unknown.sum(dim=1)
        for _, mines, unknown in test_batches
    ]).mean().item()
    print(f"固定测试集 {TEST_SIZE} 张棋盘。均匀随机选一个未知格的期望踩雷率 {random_risk:.2%}。")
    scores = {(step, name): [] for step in CHECKPOINTS for name in ARCHITECTURES}
    for seed in SEEDS:
        models = {name: make_model(widths, seed) for name, widths in ARCHITECTURES.items()}
        optimizers = {name: torch.optim.Adam(model.parameters(), lr=1e-3)
                      for name, model in models.items()}
        if seed == SEEDS[0]:
            for name, model in models.items():
                print(f"{name} 参数量 {sum(p.numel() for p in model.parameters()):,}")
        torch.manual_seed(10000 + seed)
        for step in range(1, max(CHECKPOINTS) + 1):
            x, mines, unknown = make_batch(32)
            for name, model in models.items():
                optimizer = optimizers[name]
                optimizer.zero_grad()
                loss = F.binary_cross_entropy_with_logits(model(x)[unknown], mines[unknown].float())
                loss.backward()
                optimizer.step()
            if step in CHECKPOINTS:
                print(f"种子 {seed}，训练 {step} 步：", end=" ")
                for name, model in models.items():
                    risk, bce = evaluate(model, test_batches)
                    scores[step, name].append((risk, bce))
                    print(f"{name} 踩雷 {risk:.2%} / 损失 {bce:.4f}", end="  ")
                print(flush=True)
    print(f"\n{len(SEEDS)} 组随机种子的平均结果（每张测试棋盘选一次）")
    for (step, name), results in scores.items():
        risks, losses = zip(*results)
        print(f"{step:4d} 步  {name}  踩雷 {mean(risks):.2%}  损失 {mean(losses):.4f}")


if __name__ == "__main__":
    main()
