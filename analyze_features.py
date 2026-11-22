"""复现棋盘特征实验。直接运行会训练各模型并打印结果。"""

from collections import Counter

import torch
from torch.nn import functional as F

from compare_depth import SEEDS, TEST_SIZE, make_model, make_test_set
from minesweeper import make_batch


MODELS = {"单隐藏层": (256,), "双隐藏层": (256, 128), "加宽双隐藏层": (256, 256)}
VIEWS = ("完整数字", "区分零与正数", "仅开格形状", "区分0、1、2及以上")
NEIGHBORS = ("不挨数字", "只挨1", "只挨2及以上", "同时挨1和较大数字")
SHUFFLES = ("打乱正数", "仅打乱2及以上")
REPEATS = 8


def encode_view(x, view):
    if view == "完整数字":
        return x
    cells = x.view(-1, 81, 10)
    encoded = torch.zeros_like(cells)
    encoded[:, :, 9] = cells[:, :, 9]
    if view == "区分零与正数":
        encoded[:, :, 0] = cells[:, :, 0]
        encoded[:, :, 1] = cells[:, :, 1:9].sum(dim=-1)
    elif view == "区分0、1、2及以上":
        encoded[:, :, :2] = cells[:, :, :2]
        encoded[:, :, 2] = cells[:, :, 2:9].sum(dim=-1)
    elif view == "仅开格形状":
        encoded[:, :, 1] = 1 - cells[:, :, 9]
    else:
        raise ValueError(view)
    return encoded.flatten(1)


def train(seed, view):
    models = {name: make_model(widths, seed) for name, widths in MODELS.items()}
    optimizers = {name: torch.optim.Adam(model.parameters(), lr=1e-3)
                  for name, model in models.items()}
    torch.manual_seed(10000 + seed)
    for _ in range(2000):
        x, mines, unknown = make_batch(32)
        x = encode_view(x, view)
        labels = mines[unknown].float()
        for name, model in models.items():
            optimizer = optimizers[name]
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(x)[unknown], labels)
            loss.backward()
            optimizer.step()
    return models


def around(mask):
    grid = mask.reshape(-1, 1, 9, 9).float()
    return F.conv2d(grid, torch.ones(1, 1, 3, 3), padding=1).flatten(1)


def local_features(cells, unknown):
    ones = around(cells == 1) > 0
    larger = around((cells >= 2) & (cells <= 8)) > 0
    group = ones.long() + 2 * larger.long()
    # 数字等于相邻未知格数，则这些未知格全是雷。
    clue = (cells >= 1) & (cells <= 8)
    full = clue & (around(unknown) == cells)
    certain_mines = unknown & (around(full) > 0)
    # 某数字周围的雷已全部找齐，其余相邻未知格必安全。
    satisfied = clue & (around(certain_mines) == cells)
    certain_safe = unknown & ~certain_mines & (around(satisfied) > 0)
    return group, certain_safe, certain_mines


def shuffle_clues(cells, threshold, rng):
    selected = (cells >= threshold) & (cells <= 8)
    target = (~selected).int().argsort(dim=1, stable=True)
    order = (torch.rand(cells.shape, generator=rng) + 2 * (~selected)).argsort(dim=1)
    shuffled = cells.scatter(1, target, cells.gather(1, order))
    shuffled = torch.where(selected, shuffled, cells)
    return F.one_hot(shuffled, num_classes=10).flatten(1).float()


def choose(model, x, unknown):
    return model(x).sigmoid().masked_fill(~unknown, float("inf")).argmin(dim=1)


@torch.no_grad()
def show_example(models, batches):
    # 固定测试集的第 4 张棋盘，展示正文中可手算核对的例子。
    x, mines, unknown = (t[3:4] for t in batches[0])
    cells = x.reshape(9, 9, 10).argmax(dim=2).tolist()
    print("种子 11 的第 4 张测试棋盘（· 表示未知）")
    for row in cells:
        print(" ".join("·" if n == 9 else str(n) for n in row))
    for name, model in models.items():
        pos = choose(model, x, unknown).item()
        print(name, divmod(pos, 9), "踩雷" if mines[0, pos] else "安全")


@torch.no_grad()
def measure(models, batches, view):
    rng = torch.Generator().manual_seed(31877)
    scores = {name: Counter() for name in models}
    for x, mines, unknown in batches:
        cells = x.view(-1, 81, 10).argmax(dim=2)
        group, safe, certain_mines = local_features(cells, unknown)
        assert not (safe & mines).any() and not (certain_mines & ~mines).any()
        inputs = encode_view(x, view)
        rows = torch.arange(len(x))
        for name, model in models.items():
            s = scores[name]
            picks = choose(model, inputs, unknown)
            hits = mines[rows, picks]
            s["棋盘数"] += len(x)
            s["踩雷"] += hits.sum().item()
            if view == "完整数字":
                can_prove = safe.any(dim=1)
                s["有可证安全格"] += can_prove.sum().item()
                s["选中可证安全格"] += safe[rows, picks].sum().item()
                s["有可证安全格时踩雷"] += (hits & can_prove).sum().item()
                for i, label in enumerate(NEIGHBORS):
                    picked = group[rows, picks] == i
                    s[label, "选中"] += picked.sum().item()
                    s[label, "踩雷"] += (picked & hits).sum().item()
        if view == "完整数字":
            for threshold, label in enumerate(SHUFFLES, start=1):
                for _ in range(REPEATS):
                    inputs = shuffle_clues(cells, threshold, rng)
                    for name, model in models.items():
                        picks = choose(model, inputs, unknown)
                        scores[name][label] += mines[rows, picks].sum().item()
    return scores


def report(results):
    for view in VIEWS:
        for name in MODELS:
            per_seed = [results[view, seed][name] for seed in SEEDS]
            totals = sum(per_seed, Counter())
            n = totals["棋盘数"]
            rates = [round(100*s["踩雷"]/s["棋盘数"], 3) for s in per_seed]
            print(view, name, "各次踩雷率(%)", rates, f"平均 {totals['踩雷']/n:.3%}")
            if view != "完整数字":
                continue
            for label in SHUFFLES:
                print(f"  {label}：{totals[label]/(REPEATS*n):.3%}")
            for label in NEIGHBORS:
                picks, hits = totals[label, "选中"], totals[label, "踩雷"]
                print(f"  {label}：占全部选格 {picks/n:.2%}，每千张棋盘在此踩雷 {1000*hits/n:.2f}")
            p = totals["有可证安全格"]
            print(f"  有可证安全格的棋盘 {p/n:.2%}，其中选中可证安全格 {totals['选中可证安全格']/p:.2%}")
            print(f"  有可证安全格时踩雷 {totals['有可证安全格时踩雷']/p:.2%}，"
                  f"其余棋盘踩雷 {(totals['踩雷']-totals['有可证安全格时踩雷'])/(n-p):.2%}")


def main():
    torch.set_num_threads(1)
    batches = make_test_set(TEST_SIZE)
    results = {}
    for view in VIEWS:
        for seed in SEEDS:
            models = train(seed, view)
            if view == "完整数字" and seed == 11:
                show_example(models, batches)
            results[view, seed] = measure(models, batches, view)
            print(f"已完成 {view}，随机种子 {seed}", flush=True)
    report(results)


if __name__ == "__main__":
    main()
