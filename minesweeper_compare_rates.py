"""从同一份 207 万步存档比较两个学习率，按预定条件复测新棋盘。"""

import json

import numpy as np
import torch

from minesweeper import RNG, model
from minesweeper_experiment import (
    BATCH_SIZE, EVALUATION_GAMES, PLATEAU_INTERVAL, REPLAY_PER_BATCH, SEED,
    SELF_PLAY_INTERVAL, FailedGameDataset, TorchLearner, evaluate, load_checkpoint,
    make_games, play_games, save_checkpoint, save_report,
)
from minesweeper_priority_training import (
    BEST, CHECKPOINT, PRIORITY_PER_BATCH, ROOT, SELF_PLAY_GAMES, PriorityPool,
    train_priority_steps,
)


START = 2_070_000
END = START + 300_000
RATES = (0.0001, 0.00005)
HOLDOUT_SEED = 207237
REPORT = ROOT / "minesweeper_lr_comparison_results.json"


def path_for(rate, kind):
    return ROOT / f"minesweeper_comparison_lr{rate:.5f}_{kind}.npz"


def restore(path, rate, games):
    generators = (RNG, *(np.random.default_rng(0) for _ in range(3)))
    dataset = FailedGameDataset()
    step, failures, history = load_checkpoint(path, dataset, generators)
    with np.load(path, allow_pickle=False) as archive:
        pool = PriorityPool(archive["priority_cases"])
    learner = TorchLearner(model, rate)
    row = next(r for r in history["checkpoints"] if r["updates"] == step)
    actual = evaluate(games, learner.predict, diagnose=True)
    if (any(row[key] != value for key, value in actual.items())
            or (dataset.size, pool.size, failures) != (
                row["replay_states"], row["priority_cases"], row["failed_training_games"])):
        raise ValueError(f"模型复测或回放池计数与记录不一致：{path}")
    return step, failures, history, dataset, pool, learner, generators


def train_until(rate, target, games):
    checkpoint = path_for(rate, "checkpoint")
    source = checkpoint if checkpoint.exists() else CHECKPOINT if rate == RATES[0] else BEST
    step, failures, history, dataset, pool, learner, generators = restore(source, rate, games)
    rows = [r for r in history["checkpoints"] if r["updates"] >= START]
    if rows[0]["updates"] != START or rows[0]["wins"] != 723:
        raise ValueError("缺少 207 万步的共同起点")
    if source != BEST and history["learning_rate"] != rate:
        raise ValueError("存档学习率与分组不符")
    best = max(rows, key=lambda r: r["wins"])
    report = dict(source=BEST.name, learning_rate=rate, cpu_threads=torch.get_num_threads(),
                  checkpoints=rows, best=best)
    print(f"学习率 {rate:g}｜{step:,} 步存档复测 {rows[-1]['wins']} 胜｜"
          f"{torch.get_num_threads()} 线程", flush=True)
    for current, failures in train_priority_steps(
        range(step + 1, target + 1), learner, dataset, pool, failures, *generators[1:],
    ):
        if (current - START) % PLATEAU_INTERVAL:
            if current % 10_000 == 0:
                print(f"{rate:g}｜{current:,} 步", flush=True)
            continue
        row = evaluate(games, learner.predict, diagnose=True)
        row.update(updates=current, failed_training_games=failures, replay_states=dataset.size,
                   priority_cases=pool.size)
        rows.append(row)
        learner.copy_state_to(model)
        extra = {"priority_cases": pool.rows[:pool.size]}
        if row["wins"] > best["wins"]:
            best = report["best"] = row
            save_checkpoint(path_for(rate, "best"), current, dataset, failures, generators, report,
                            extra_arrays=extra)
        save_checkpoint(checkpoint, current, dataset, failures, generators, report, extra_arrays=extra)
        print(f"{rate:g}｜{current:,} 步｜{row['wins']}/1000｜"
              f"有安全格却踩雷 {row['failure_diagnostics']['safe_move_available']} 局", flush=True)
    return report


def holdout_results():
    games = make_games(np.random.default_rng(HOLDOUT_SEED), EVALUATION_GAMES)
    outcomes = {}
    for label, path in [("baseline", BEST)] + [(str(r), path_for(r, "checkpoint")) for r in RATES]:
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(archive["metadata"].item())
            if meta["step"] != (START if label == "baseline" else END):
                raise ValueError("新棋盘复测的存档步数不正确")
            for name, array in model.parameters.items():
                array[:] = archive[f"param_{name}"]
        learner = TorchLearner(model)
        outcomes[label] = np.concatenate([
            play_games(*(a[i:i + 128] for a in games), predict=learner.predict)[0]
            for i in range(0, EVALUATION_GAMES, 128)
        ])
    high, low = (outcomes[str(r)] for r in RATES)
    return dict(seed=HOLDOUT_SEED, games=EVALUATION_GAMES,
                wins={k: int(x.sum()) for k, x in outcomes.items()},
                low_only=int((low & ~high).sum()), high_only=int((high & ~low).sum()))


def run():
    games = make_games(np.random.default_rng(np.random.SeedSequence(SEED).spawn(3)[2]), EVALUATION_GAMES)
    results = dict(start=START, end=END, evaluation_interval=PLATEAU_INTERVAL,
                   original_board_seed=SEED, original_board_spawn_index=2,
                   fresh_per_batch=BATCH_SIZE - REPLAY_PER_BATCH,
                   replay_per_batch=REPLAY_PER_BATCH - PRIORITY_PER_BATCH,
                   priority_per_batch=PRIORITY_PER_BATCH, games_per_round=SELF_PLAY_GAMES,
                   self_play_interval=SELF_PLAY_INTERVAL, control_continuation=CHECKPOINT.name,
                   required_consecutive_leads=2, holdout_seed=HOLDOUT_SEED, groups={})
    # 两组轮流独占 CPU。对照组可沿用原先从同一起点走过的前三个检查点。
    for target in range(START + PLATEAU_INTERVAL, END + 1, PLATEAU_INTERVAL):
        for rate in RATES:
            group = results["groups"].get(str(rate))
            if group is None or group["checkpoints"][-1]["updates"] < target:
                results["groups"][str(rate)] = train_until(rate, target, games)
                save_report(REPORT, results)
    high, low = (results["groups"][str(rate)]["checkpoints"] for rate in RATES)
    expected = list(range(START, END + 1, PLATEAU_INTERVAL))
    if any([r["updates"] for r in rows] != expected for rows in (high, low)):
        raise ValueError("固定评估的检查点不完整")
    leads = [l["wins"] > h["wins"] for h, l in zip(high[1:], low[1:])]
    results["holdout_triggered"] = any(a and b for a, b in zip(leads, leads[1:]))
    if results["holdout_triggered"]:
        results["holdout"] = holdout_results()
    save_report(REPORT, results)
    print(f"对照训练完成。新棋盘复测：{results.get('holdout', '未触发')}", flush=True)


if __name__ == "__main__":
    run()
