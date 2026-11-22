"""从 237 万步低学习率存档续训，在之前的两组固定棋盘上比较胜率。"""

import json

import numpy as np
import torch

from minesweeper import model
from minesweeper_compare_rates import HOLDOUT_SEED, REPORT as COMPARISON_REPORT, path_for, restore
from minesweeper_experiment import (
    BATCH_SIZE, EVALUATION_GAMES, PLATEAU_INTERVAL, REPLAY_PER_BATCH, SEED,
    SELF_PLAY_INTERVAL, evaluate, make_games, save_checkpoint, save_report,
)
from minesweeper_priority_training import PRIORITY_PER_BATCH, ROOT, SELF_PLAY_GAMES, train_priority_steps


START = 2_370_000
END = START + 300_000
RATE = 0.00005
SOURCE = path_for(RATE, "checkpoint")
CHECKPOINT = ROOT / "minesweeper_low_rate_2670k_checkpoint.npz"
BEST = ROOT / "minesweeper_low_rate_2670k_best.npz"
REPORT = ROOT / "minesweeper_low_rate_2670k_results.json"


def fixed_games():
    original_seed = np.random.SeedSequence(SEED).spawn(3)[2]
    return tuple(make_games(np.random.default_rng(seed), EVALUATION_GAMES)
                 for seed in (original_seed, HOLDOUT_SEED))


def run():
    original, second = fixed_games()
    resume = CHECKPOINT.exists()
    path = CHECKPOINT if resume else SOURCE
    step, failures, history, dataset, pool, learner, generators = restore(path, RATE, original)
    if history["learning_rate"] != RATE:
        raise ValueError(f"存档的学习率不符：{path}")
    second_result = evaluate(second, learner.predict, diagnose=True)
    last = history["checkpoints"][-1]
    if resume:
        report = history
        if last["second"] != second_result:
            raise ValueError("第二组固定棋盘的胜场或诊断与续跑记录不符")
    else:
        previous = json.loads(COMPARISON_REPORT.read_text())
        if step != START or second_result["wins"] != previous["holdout"]["wins"][str(RATE)]:
            raise ValueError("起点步数或第二组起点胜场与此前对照实验不符")
        row = dict(last, second=second_result, combined_wins=last["wins"] + second_result["wins"])
        report = dict(
            source=SOURCE.name, start=START, end=END, learning_rate=RATE,
            cpu_threads=torch.get_num_threads(), evaluation_interval=PLATEAU_INTERVAL,
            original_board_seed=SEED, original_board_spawn_index=2, second_board_seed=HOLDOUT_SEED,
            fresh_per_batch=BATCH_SIZE - REPLAY_PER_BATCH,
            replay_per_batch=REPLAY_PER_BATCH - PRIORITY_PER_BATCH, priority_per_batch=PRIORITY_PER_BATCH,
            self_play_interval=SELF_PLAY_INTERVAL, games_per_round=SELF_PLAY_GAMES,
            checkpoints=[row], best=row,
        )
    save_report(REPORT, report)
    print(f"{step:,} 步复测：{last['wins']}、{second_result['wins']} 胜；"
          f"CPU {torch.get_num_threads()} 线程，学习率 {RATE:g}", flush=True)

    for current, failures in train_priority_steps(
        range(step + 1, END + 1), learner, dataset, pool, failures, *generators[1:],
    ):
        if (current - START) % PLATEAU_INTERVAL:
            if current % 10_000 == 0:
                print(f"{current:,} 步", flush=True)
            continue
        row = evaluate(original, learner.predict, diagnose=True)
        second_result = evaluate(second, learner.predict, diagnose=True)
        row.update(
            second=second_result, combined_wins=row["wins"] + second_result["wins"],
            updates=current, failed_training_games=failures,
            replay_states=dataset.size, priority_cases=pool.size,
        )
        report["checkpoints"].append(row)
        learner.copy_state_to(model)
        extra = {"priority_cases": pool.rows[:pool.size]}
        if row["combined_wins"] > report["best"]["combined_wins"]:
            report["best"] = row
            save_checkpoint(BEST, current, dataset, failures, generators, report, extra_arrays=extra)
        save_checkpoint(CHECKPOINT, current, dataset, failures, generators, report, extra_arrays=extra)
        save_report(REPORT, report)
        print(f"{current:,} 步：{row['wins']}、{second_result['wins']} 胜；"
              f"合计 {row['combined_wins']}/2000，专项池 {pool.size:,}", flush=True)


if __name__ == "__main__":
    run()
