"""持续收录有确定安全格却踩雷的局面，并增加这类样本的训练频率。"""

from itertools import count
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from minesweeper import RNG, encode_board, make_batch_block, model
from minesweeper_experiment import (
    BATCH_SIZE,
    EVALUATION_GAMES,
    PLATEAU_INTERVAL,
    PLATEAU_MIN_GAIN,
    PLATEAU_PATIENCE,
    REPLAY_PER_BATCH,
    SAVE_INTERVAL,
    SEED,
    SELF_PLAY_INTERVAL,
    FailedGameDataset,
    TorchLearner,
    evaluate,
    load_checkpoint,
    make_games,
    play_games,
    save_checkpoint,
    save_report,
    track_plateau,
)
from minesweeper_logic_training import RuleCases, SYMMETRIES, annotate, rule_loss


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "minesweeper_logic_balanced_best.npz"
REPORT = ROOT / "minesweeper_priority_results.json"
CHECKPOINT = ROOT / "minesweeper_priority_checkpoint.npz"
BEST = ROOT / "minesweeper_priority_best.npz"
PRIORITY_PER_BATCH = 6
WARMUP_GAMES = 2_048
SELF_PLAY_GAMES = 64


class PriorityPool:
    """每行保存一次实际踩雷前的棋盘、真实雷区和两组可推导标签。"""

    def __init__(self, rows=None):
        self.size = 0 if rows is None else len(rows)
        self.rows = np.empty((1024, 4, 81), dtype=np.uint8) if rows is None else rows

    def add(self, failed_positions):
        if not failed_positions:
            return
        visible, mines = (np.stack(values) for values in zip(*failed_positions))
        annotated = np.stack(annotate(visible, mines), axis=1)
        end = self.size + len(annotated)
        if end > len(self.rows):
            enlarged = np.empty((max(end, 2 * len(self.rows)), 4, 81), dtype=np.uint8)
            enlarged[:self.size] = self.rows[:self.size]
            self.rows = enlarged
        self.rows[self.size:end] = annotated
        self.size = end

    def sample(self, size, rng):
        chosen = rng.integers(self.size, size=size)
        columns = SYMMETRIES[rng.integers(len(SYMMETRIES), size=size)]
        rows = self.rows[chosen[:, None, None], np.arange(4)[None, :, None], columns[:, None, :]]
        return RuleCases(rows[:, 0], rows[:, 1].view(bool), rows[:, 2].view(bool), rows[:, 3].view(bool))


def collect_games(size, rng, learner, dataset, priority_pool):
    """所有失败都进入原回放池，符合条件的那一步同时进入专项池。"""
    failures = 0
    for start in range(0, size, 128):
        failed_positions = []
        won, _ = play_games(
            *make_games(rng, min(size - start, 128)),
            predict=learner.predict,
            failed_dataset=dataset,
            failure_cases=failed_positions,
        )
        failures += len(won) - int(won.sum())
        priority_pool.add(failed_positions)
    return failures


def train_priority_steps(steps, learner, dataset, priority_pool, failures,
                         self_play_rng, replay_rng, priority_rng):
    """按同一套自玩、失败回放和专项损失规则执行连续的训练更新。"""
    fresh_batches = iter(())
    for current_step in steps:
        if (current_step - 1) % SELF_PLAY_INTERVAL == 0:
            failures += collect_games(SELF_PLAY_GAMES, self_play_rng, learner, dataset, priority_pool)

        fresh = next(fresh_batches, None)
        if fresh is None:
            fresh_batches = zip(*make_batch_block(BATCH_SIZE - REPLAY_PER_BATCH, 25))
            fresh = next(fresh_batches)
        priority_size = PRIORITY_PER_BATCH if priority_pool.size else 0
        replay = dataset.sample(REPLAY_PER_BATCH - priority_size, replay_rng)
        if priority_size:
            cases = priority_pool.sample(priority_size, priority_rng)
            priority = encode_board(cases.visible), cases.mines, cases.visible == 9
            batch = tuple(np.concatenate(parts) for parts in zip(fresh, replay, priority))
            safe = torch.from_numpy(cases.known_safe)
            known_mines = torch.from_numpy(cases.known_mines)
            actual_mines = torch.from_numpy(cases.mines)

            def extra_loss(logits, unknown):
                return rule_loss(
                    logits[-priority_size:], unknown[-priority_size:], safe, known_mines,
                    mine_targets=actual_mines, weight=0.1,
                )

            learner.train(*batch, extra_loss=extra_loss)
        else:
            learner.train(*(np.concatenate(parts) for parts in zip(fresh, replay)))

        yield current_step, failures


def run():
    self_play_rng, replay_rng, evaluation_rng = [
        np.random.default_rng(seed) for seed in np.random.SeedSequence(SEED).spawn(3)
    ]
    priority_rng = np.random.default_rng(26831)
    evaluation_games = make_games(evaluation_rng, EVALUATION_GAMES)
    generators = (RNG, self_play_rng, replay_rng, priority_rng)
    dataset = FailedGameDataset()
    resume = CHECKPOINT.exists()
    step, failures, previous = load_checkpoint(CHECKPOINT if resume else SOURCE, dataset, generators)
    learning_rate = previous["learning_rate"]
    learner = TorchLearner(model, learning_rate)
    print(
        f"从 {step:,} 步续跑｜学习率 {learning_rate:g}｜CPU 线程数 {torch.get_num_threads()}。",
        flush=True,
    )

    if resume:
        report = previous
        with np.load(CHECKPOINT, allow_pickle=False) as archive:
            priority_pool = PriorityPool(archive["priority_cases"])
        save_report(REPORT, report)
        if report["plateau"]["status"] == "reached":
            print(f"已在第 {report['plateau']['stop_update']:,} 步达到平台期。")
            return
    else:
        priority_pool = PriorityPool()
        baseline = evaluate(evaluation_games, learner.predict, diagnose=True)
        baseline.update(
            updates=step, failed_training_games=failures, replay_states=dataset.size,
            priority_cases=0,
        )
        print(
            f"起点复测 {baseline['wins']}/{baseline['games']}（{baseline['win_rate']:.2%}）。"
            f"先自玩 {WARMUP_GAMES:,} 局初始化专项池。",
            flush=True,
        )
        failures += collect_games(WARMUP_GAMES, self_play_rng, learner, dataset, priority_pool)
        report = {
            "source": str(SOURCE),
            "seed": SEED,
            "learning_rate": learning_rate,
            "cpu_threads": torch.get_num_threads(),
            "batch_size": BATCH_SIZE,
            "priority_cases_per_batch": PRIORITY_PER_BATCH,
            "self_play_games_per_round": SELF_PLAY_GAMES,
            "self_play_interval": SELF_PLAY_INTERVAL,
            "warmup_games": WARMUP_GAMES,
            "initial_priority_cases": priority_pool.size,
            "checkpoints": [baseline],
            "best": baseline,
            "plateau": {
                "status": "training",
                "start_update": step,
                "evaluation_interval": PLATEAU_INTERVAL,
                "minimum_gain_wins": PLATEAU_MIN_GAIN,
                "patience_evaluations": PLATEAU_PATIENCE,
                "anchor_wins": baseline["wins"],
                "evaluations_without_gain": 0,
                "best": baseline,
            },
        }
        print(f"专项池已收录 {priority_pool.size:,} 个局面。", flush=True)

    def snapshot(path, current_step):
        save_checkpoint(
            path, current_step, dataset, failures, generators, report,
            extra_arrays={"priority_cases": priority_pool.rows[:priority_pool.size]},
        )

    if not resume:
        snapshot(BEST, step)
        snapshot(CHECKPOINT, step)
        save_report(REPORT, report)

    plateau = report["plateau"]
    started = perf_counter()
    for current_step, failures in train_priority_steps(
        count(step + 1), learner, dataset, priority_pool, failures,
        self_play_rng, replay_rng, priority_rng,
    ):
        offset = current_step - plateau["start_update"]
        assessed = offset % plateau["evaluation_interval"] == 0
        is_best = False
        if assessed:
            result = evaluate(evaluation_games, learner.predict, diagnose=True)
            result.update(
                updates=current_step, failed_training_games=failures, replay_states=dataset.size,
                priority_cases=priority_pool.size,
            )
            report["checkpoints"].append(result)
            is_best = track_plateau(plateau, result)
            report["best"] = plateau["best"]
            diagnostics = result["failure_diagnostics"]
            print(
                f"{current_step:,} 步｜通关 {result['wins']}/{result['games']}（{result['win_rate']:.2%}）"
                f"｜失败 {diagnostics['failed_games']} 局中有 {diagnostics['safe_move_available']} 局可先点安全格"
                f"｜专项池 {priority_pool.size:,} 个"
                f"｜平台期计数 {plateau['evaluations_without_gain']}/{plateau['patience_evaluations']}",
                flush=True,
            )
        elif offset % 5_000 == 0:
            print(
                f"{current_step:,} 步｜专项池 {priority_pool.size:,} 个"
                f"｜已耗时 {perf_counter() - started:.0f} 秒。",
                flush=True,
            )

        if assessed or offset % SAVE_INTERVAL == 0:
            learner.copy_state_to(model)
            if is_best:
                snapshot(BEST, current_step)
            snapshot(CHECKPOINT, current_step)
            if assessed:
                save_report(REPORT, report)
        if plateau["status"] == "reached":
            print(f"第 {current_step:,} 步触发平台期停止。结果已保存到 {REPORT}。", flush=True)
            break


if __name__ == "__main__":
    run()
