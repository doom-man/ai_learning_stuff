"""训练时回收失败对局，在固定评估集上比较不同训练步数的通关率。"""

import argparse
import json
import os
from itertools import count
from pathlib import Path
from time import perf_counter

import numpy as np

from minesweeper import (
    RNG,
    count_adjacent_mines,
    encode_board,
    expand_zero_regions,
    make_batch,
    make_batch_block,
    model,
    sigmoid,
    train_step,
)


CHECKPOINTS = (1_000, 10_000, 100_000, 1_000_000)
SAVE_INTERVAL = 25_000
TORCH_AFTER = 100_000
FRESH_BLOCK = 25
BATCH_SIZE = 32
REPLAY_PER_BATCH = 16
SELF_PLAY_INTERVAL = 100
SELF_PLAY_GAMES = 32
EVALUATION_GAMES = 1_000
PLATEAU_INTERVAL = 50_000
PLATEAU_MIN_GAIN = 10
PLATEAU_PATIENCE = 3
LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 5e-4
SAFE_CELLS = 71
SEED = 2026
NEIGHBORS = tuple(
    tuple(
        r * 9 + c
        for r in range(max(0, row - 1), min(9, row + 2))
        for c in range(max(0, col - 1), min(9, col + 2))
        if (r, c) != (row, col)
    )
    for row in range(9)
    for col in range(9)
)


class FailedGameDataset:
    """保留全部失败轨迹，只存每格的状态编号，抽到后再做 one-hot。"""

    def __init__(self):
        self.visible = np.empty((1024, 81), dtype=np.uint8)
        self.mine_bits = np.empty((1024, 11), dtype=np.uint8)
        self.size = 0

    def add(self, visible, mines):
        end = self.size + len(visible)
        if end > len(self.visible):
            capacity = max(end, len(self.visible) * 2)
            new_visible = np.empty((capacity, 81), dtype=self.visible.dtype)
            new_mine_bits = np.empty((capacity, 11), dtype=np.uint8)
            new_visible[:self.size] = self.visible[:self.size]
            new_mine_bits[:self.size] = self.mine_bits[:self.size]
            self.visible, self.mine_bits = new_visible, new_mine_bits
        self.visible[self.size:end] = visible
        self.mine_bits[self.size:end] = np.packbits(mines, axis=1)
        self.size = end

    def sample(self, size, rng):
        indices = rng.integers(self.size, size=size)
        visible = self.visible[indices]
        mines = np.unpackbits(self.mine_bits[indices], axis=1, count=81).view(bool)
        return encode_board(visible), mines, visible == 9


class TorchLearner:
    """用 PyTorch 接着训练同一个单隐藏层模型，并保留 Adam 的历史状态。"""

    def __init__(self, source, learning_rate=LEARNING_RATE):
        import torch
        from torch.nn import functional as F

        self.torch = torch
        self.F = F
        self.step = source.adam_step
        torch.set_num_threads(os.cpu_count() or 1)
        self.parameters = {
            name: torch.nn.Parameter(self._tensor(name, value))
            for name, value in source.parameters.items()
        }
        self.optimizer = torch.optim.Adam(self.parameters.values(), lr=learning_rate, foreach=True)
        for name, parameter in self.parameters.items():
            self.optimizer.state[parameter].update(
                step=torch.tensor(float(self.step)),
                exp_avg=self._tensor(name, source.first_moment[name]),
                exp_avg_sq=self._tensor(name, source.second_moment[name]),
            )

    def _tensor(self, name, value):
        # F.linear 的权重排列是 [输出宽度, 输入宽度]。
        return self.torch.from_numpy((value.T if name.startswith("W") else value).copy())

    def _forward(self, x):
        params = self.parameters
        hidden = self.F.linear(x, params["W1"], params["b1"]).relu()
        return self.F.linear(hidden, params["W2"], params["b2"])

    def predict(self, x):
        with self.torch.inference_mode():
            return self._forward(self.torch.from_numpy(x)).numpy()

    def train(self, x, mines, unknown, *, extra_loss=None):
        torch = self.torch
        self.optimizer.zero_grad(set_to_none=True)
        logits = self._forward(torch.from_numpy(x))
        target = torch.from_numpy(mines).float()
        mask = torch.from_numpy(unknown)
        loss_per_cell = self.F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        loss = (loss_per_cell * mask).sum() / mask.sum()
        if extra_loss is not None:
            loss = loss + extra_loss(logits, mask)
        loss.backward()
        self.optimizer.step()
        self.step += 1

    def copy_state_to(self, destination):
        """存档继续沿用 NumPy 版的权重布局，方便恢复和复用。"""
        for name, parameter in self.parameters.items():
            state = self.optimizer.state[parameter]
            for target, tensor in (
                (destination.parameters[name], parameter),
                (destination.first_moment[name], state["exp_avg"]),
                (destination.second_moment[name], state["exp_avg_sq"]),
            ):
                value = tensor.detach().numpy()
                target[:] = value.T if name.startswith("W") else value
        destination.adam_step = self.step


def numpy_predict(x):
    return model.forward(x)[0]


def make_games(rng, count):
    """随机生成雷区和一个确保安全的开局格。"""
    mines = np.zeros((count, 81), dtype=bool)
    board_ids = np.arange(count)
    mine_positions = np.argsort(rng.random((count, 81)), axis=1)[:, :10]
    mines[board_ids[:, None], mine_positions] = True
    opening_scores = np.where(mines, np.inf, rng.random((count, 81)))
    return mines, opening_scores.argmin(axis=1)


def deduce_cells(visible):
    """只从可见数字和总雷数推导确定安全格和确定雷位。"""
    hidden = set(np.flatnonzero(visible == 9))
    clues = [
        (int(value), hidden.intersection(NEIGHBORS[index]))
        for index, value in enumerate(visible)
        if value != 9
    ]
    known_safe, known_mines = set(), set()
    while True:
        before = len(known_safe) + len(known_mines)
        for value, neighbors in clues:
            unaccounted = value - len(neighbors.intersection(known_mines))
            undecided = neighbors - known_mines - known_safe
            if unaccounted == 0:
                known_safe.update(undecided)
            elif unaccounted == len(undecided):
                known_mines.update(undecided)
        if len(known_mines) == 10:
            known_safe.update(hidden - known_mines)
        if len(hidden - known_safe) == 10:
            known_mines.update(hidden - known_safe)
        if len(known_safe) + len(known_mines) == before:
            return known_safe, known_mines


def play_games(
    mines, openings, *, predict=numpy_predict, failed_dataset=None, diagnostics=None, failure_cases=None,
):
    """批量玩到终局，并可收录每场失败对局的所有模型选格前状态。"""
    count = len(mines)
    clues = count_adjacent_mines(mines)
    revealed = np.zeros_like(mines)
    revealed[np.arange(count), openings] = True
    revealed = expand_zero_regions(revealed.reshape(-1, 9, 9), clues).reshape(count, 81)
    revealed_count = revealed.sum(axis=1)
    won = revealed_count == SAFE_CELLS
    active = np.flatnonzero(~won)
    trajectory = []

    while active.size:
        visible = np.where(revealed[active], clues[active].reshape(-1, 81), 9)
        if failed_dataset is not None:
            trajectory.append((active, visible.astype(np.uint8)))
        logits = predict(encode_board(visible))
        # 与单局演示一致，只比较未知格的踩雷概率。
        risk = np.where(visible == 9, sigmoid(logits), np.inf)
        selected = risk.argmin(axis=1)
        safe = ~mines[active, selected]
        if diagnostics is not None or failure_cases is not None:
            for index in np.flatnonzero(~safe):
                board = visible[index]
                known_safe, known_mines = deduce_cells(board)
                if diagnostics is not None:
                    diagnostics["safe_move_available"] += bool(known_safe)
                    diagnostics["selected_provable_mine"] += selected[index] in known_mines
                if failure_cases is not None and known_safe:
                    failure_cases.append((board.copy(), mines[active[index]].copy()))
        survivors = active[safe]
        if survivors.size == 0:
            break
        revealed[survivors, selected[safe]] = True
        revealed[survivors] = expand_zero_regions(
            revealed[survivors].reshape(-1, 9, 9), clues[survivors]
        ).reshape(-1, 81)
        revealed_count[survivors] = revealed[survivors].sum(axis=1)
        won[survivors] = revealed_count[survivors] == SAFE_CELLS
        active = survivors[~won[survivors]]

    if failed_dataset is not None:
        for board_ids, visible in trajectory:
            failed = ~won[board_ids]
            failed_dataset.add(visible[failed], mines[board_ids[failed]])
    return won, revealed_count


def evaluate(games, predict, *, diagnose=False, failure_cases=None):
    wins = 0
    revealed = []
    mines, openings = games
    diagnostics = {"safe_move_available": 0, "selected_provable_mine": 0} if diagnose else None
    # 每批并行评估少量棋盘，避免把所有对局状态同时展开。
    for start in range(0, len(mines), 128):
        end = start + 128
        won, revealed_count = play_games(
            mines[start:end], openings[start:end], predict=predict,
            diagnostics=diagnostics, failure_cases=failure_cases,
        )
        wins += int(won.sum())
        revealed.extend(revealed_count.tolist())
    result = {
        "wins": wins,
        "games": len(mines),
        "win_rate": wins / len(mines),
        "mean_safe_revealed": float(np.mean(revealed)),
    }
    if diagnostics is not None:
        diagnostics["failed_games"] = len(mines) - wins
        result["failure_diagnostics"] = diagnostics
    return result


def save_checkpoint(path, step, dataset, failures, generators, report, *, extra_arrays=None):
    """原子替换上一次存档，保存优化器、随机数状态和完整失败数据。"""
    metadata = {
        "step": step,
        "adam_step": model.adam_step,
        "failures": failures,
        "random_states": [rng.bit_generator.state for rng in generators],
        "report": report,
    }
    arrays = {}
    for prefix, values in (
        ("param", model.parameters),
        ("first", model.first_moment),
        ("second", model.second_moment),
    ):
        arrays.update({f"{prefix}_{name}": value for name, value in values.items()})
    if extra_arrays is not None:
        arrays.update(extra_arrays)
    pending = path.with_suffix(".tmp.npz")
    np.savez(
        pending,
        metadata=json.dumps(metadata),
        visible=dataset.visible[:dataset.size],
        mine_bits=dataset.mine_bits[:dataset.size],
        **arrays,
    )
    pending.replace(path)


def load_checkpoint(path, dataset, generators):
    """恢复训练和采样所需的全部状态。"""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata"].item())
        dataset.visible = archive["visible"]
        dataset.mine_bits = archive["mine_bits"]
        dataset.size = len(dataset.visible)
        for prefix, values in (
            ("param", model.parameters),
            ("first", model.first_moment),
            ("second", model.second_moment),
        ):
            for name, value in values.items():
                value[:] = archive[f"{prefix}_{name}"]
    model.adam_step = metadata.get("adam_step", metadata["step"])
    for rng, state in zip(generators, metadata["random_states"]):
        rng.bit_generator.state = state
    return metadata["step"], metadata["failures"], metadata["report"]


def save_report(path, report):
    pending = path.with_suffix(".tmp.json")
    pending.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def load_best_weights(path):
    """从旧的最佳权重文件开一个分支，从零初始化 Adam。"""
    with np.load(path, allow_pickle=False) as archive:
        result = json.loads(archive["metadata"].item())
        for name, values in model.parameters.items():
            values[:] = archive[f"param_{name}"]
            model.first_moment[name].fill(0)
            model.second_moment[name].fill(0)
    model.adam_step = 0
    return result


def track_plateau(plateau, result):
    """独立记录最高胜率，以及最近一次至少提高 10 局的位置。"""
    is_best = result["wins"] > plateau["best"]["wins"]
    if is_best:
        plateau["best"] = result
    if result["wins"] >= plateau["anchor_wins"] + plateau["minimum_gain_wins"]:
        plateau["anchor_wins"] = result["wins"]
        plateau["evaluations_without_gain"] = 0
    else:
        plateau["evaluations_without_gain"] += 1
    if plateau["evaluations_without_gain"] >= plateau["patience_evaluations"]:
        plateau["status"] = "reached"
        plateau["stop_update"] = result["updates"]
    return is_best


def run_experiment(
    *, until_plateau=False, fine_tune_best=False, resume_best=None, new_learning_rate=None,
    through_step=None,
):
    until_plateau = through_step is None and (until_plateau or fine_tune_best or resume_best is not None)
    final_step = through_step if through_step is not None else CHECKPOINTS[-1]
    self_play_rng, sample_rng, evaluation_rng = [
        np.random.default_rng(seed) for seed in np.random.SeedSequence(SEED).spawn(3)
    ]
    evaluation_games = make_games(evaluation_rng, EVALUATION_GAMES)
    dataset = FailedGameDataset()
    base = "minesweeper_experiment"
    stem = "minesweeper_finetune" if fine_tune_best else base
    if resume_best is not None:
        stem = f"{resume_best.stem.removesuffix('_best')}_lr{new_learning_rate:g}"
    output_path = Path(__file__).with_name(f"{stem}_results.json")
    checkpoint_path = output_path.with_name(f"{stem}_checkpoint.npz")
    best_path = output_path.with_name(f"{stem}_best.npz")
    generators = (RNG, self_play_rng, sample_rng)
    report = {
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "replay_per_batch": REPLAY_PER_BATCH,
        "self_play_interval": SELF_PLAY_INTERVAL,
        "self_play_games_per_round": SELF_PLAY_GAMES,
        "evaluation_games": EVALUATION_GAMES,
        "checkpoints": [],
    }
    failures = 0
    start_step = 0
    learner = None
    predict = numpy_predict
    train_batch = train_step
    fresh_batches = iter(())
    learning_rate = LEARNING_RATE
    if checkpoint_path.exists():
        start_step, failures, report = load_checkpoint(checkpoint_path, dataset, generators)
        learning_rate = report.get("learning_rate", LEARNING_RATE)
        save_report(output_path, report)
        print(f"已恢复到第 {start_step:,} 步，失败数据池有 {dataset.size:,} 个状态。", flush=True)
    elif fine_tune_best or resume_best is not None:
        if resume_best is not None:
            start_step, failures, previous_report = load_checkpoint(resume_best, dataset, generators)
            previous_best = previous_report.get("best", previous_report.get("plateau", {}).get("best"))
            if previous_best is None or previous_best["updates"] != start_step:
                raise SystemExit("指定的存档不在上一次评估的最佳步数。")
            replay_step = start_step
            previous_learning_rate = previous_report.get("learning_rate", LEARNING_RATE)
            learning_rate = new_learning_rate
            adam_status = "沿用 Adam 状态"
            report["starting_checkpoint"] = str(resume_best.resolve())
        else:
            previous_path = output_path.with_name(f"{base}_checkpoint.npz")
            previous_best_path = output_path.with_name(f"{base}_best.npz")
            replay_step, failures, _ = load_checkpoint(previous_path, dataset, generators)
            original_result = load_best_weights(previous_best_path)
            start_step = original_result["updates"]
            previous_learning_rate = LEARNING_RATE
            learning_rate = FINE_TUNE_LEARNING_RATE
            adam_status = "Adam 重新初始化"

        learner = TorchLearner(model, learning_rate)
        predict, train_batch = learner.predict, learner.train
        baseline = evaluate(evaluation_games, predict)
        baseline.update(updates=start_step, failed_training_games=failures, replay_states=dataset.size)
        report.update(
            learning_rate=learning_rate,
            previous_learning_rate=previous_learning_rate,
            starting_model_update=start_step,
            starting_replay_update=replay_step,
            adam_reset_at_start=resume_best is None,
            pytorch_cpu_threads=learner.torch.get_num_threads(),
            pytorch_from_update=start_step + 1,
            checkpoints=[baseline],
        )
        print(
            f"已载入 {start_step:,} 步最佳权重；同组棋盘复测 {baseline['wins']}/{baseline['games']} "
            f"（{baseline['win_rate']:.2%}）。沿用截至 {replay_step:,} 步的失败数据。",
            flush=True,
        )
        print(
            f"{adam_status}，学习率 {learning_rate:g}，"
            f"CPU 线程数：{report['pytorch_cpu_threads']}。",
            flush=True,
        )

    fixed_run = None
    if through_step is not None:
        if not report["checkpoints"] or start_step > final_step:
            raise SystemExit("固定步数续训需要已有评估的存档，目标步数须不早于存档。")
        fixed_run = report.get("fixed_run")
        if fixed_run is None or fixed_run["end_update"] != final_step:
            fixed_run = {"start_update": start_step, "end_update": final_step}
            report["fixed_run"] = fixed_run
            report["best"] = max(report["checkpoints"], key=lambda result: result["wins"])
            if learner is None:
                learner = TorchLearner(model, learning_rate)
                predict, train_batch = learner.predict, learner.train
            baseline = evaluate(evaluation_games, predict, diagnose=True)
            baseline.update(updates=start_step, failed_training_games=failures, replay_states=dataset.size)
            if report["checkpoints"][-1]["updates"] == start_step:
                report["checkpoints"][-1] = baseline
            else:
                report["checkpoints"].append(baseline)
            fixed_run["baseline"] = baseline
            save_checkpoint(checkpoint_path, start_step, dataset, failures, generators, report)
            save_report(output_path, report)
            diagnostic = baseline["failure_diagnostics"]
            print(
                f"起点复测 {baseline['wins']}/{baseline['games']}（{baseline['win_rate']:.2%}）；"
                f"失败 {diagnostic['failed_games']} 局中，{diagnostic['safe_move_available']} 局踩雷时已有确定安全格。",
                flush=True,
            )
        print(
            f"固定训练到 {final_step:,} 步，每 {PLATEAU_INTERVAL:,} 步评估，平台期不提前结束。"
            f"学习率 {learning_rate:g}，CPU 线程数：{os.cpu_count() or 1}。",
            flush=True,
        )

    plateau = None
    if until_plateau:
        if start_step < CHECKPOINTS[-1]:
            raise SystemExit("请先用原实验跑到 1,000,000 步，以该评估结果作为平台期比较的起点。")
        plateau = report.get("plateau")
        if plateau is None:
            baseline = report["checkpoints"][-1]
            plateau = {
                "status": "training",
                "start_update": start_step,
                "evaluation_interval": PLATEAU_INTERVAL,
                "minimum_gain_wins": PLATEAU_MIN_GAIN,
                "patience_evaluations": PLATEAU_PATIENCE,
                "anchor_wins": baseline["wins"],
                "evaluations_without_gain": 0,
                "best": baseline,
            }
            report["plateau"] = plateau
            save_checkpoint(best_path, start_step, dataset, failures, generators, report)
            save_checkpoint(checkpoint_path, start_step, dataset, failures, generators, report)
            save_report(output_path, report)
        if plateau["status"] == "reached":
            print(f"已在第 {plateau['stop_update']:,} 步达到平台期，结果在 {output_path}。")
            return
        print(
            f"每 {plateau['evaluation_interval']:,} 步评估。连续 "
            f"{plateau['patience_evaluations']} 次未多赢至少 "
            f"{plateau['minimum_gain_wins']} 局时停止。",
            flush=True,
        )

    started = perf_counter()
    print(
        f"固定评估 {EVALUATION_GAMES:,} 局；每 {SELF_PLAY_INTERVAL} 次更新自玩 "
        f"{SELF_PLAY_GAMES} 局；每批最多一半来自失败对局。",
        flush=True,
    )

    steps = count(start_step + 1) if until_plateau else range(start_step + 1, final_step + 1)
    for step in steps:
        if learner is None and step == max(start_step, TORCH_AFTER) + 1:
            learner = TorchLearner(model, learning_rate)
            predict, train_batch = learner.predict, learner.train
            report.setdefault("pytorch_from_update", TORCH_AFTER + 1)
            report["pytorch_cpu_threads"] = learner.torch.get_num_threads()
            print(
                f"第 {step:,} 步开始用 PyTorch 训练，已接续 Adam 状态，"
                f"CPU 线程数：{report['pytorch_cpu_threads']}。",
                flush=True,
            )

        # 用最近一轮更新后的模型自玩，收集到的失败状态立即进入采样池。
        if step > 1 and (step - 1) % SELF_PLAY_INTERVAL == 0:
            won, _ = play_games(
                *make_games(self_play_rng, SELF_PLAY_GAMES), predict=predict, failed_dataset=dataset
            )
            failures += len(won) - int(won.sum())

        replay_count = REPLAY_PER_BATCH if dataset.size else 0
        fresh_count = BATCH_SIZE - replay_count
        if learner is not None and dataset.size:
            batch = next(fresh_batches, None)
            if batch is None:
                # 预生成的批次不会越过存档点，恢复时只需要存一个训练随机数状态。
                until_save = SAVE_INTERVAL - (step - 1) % SAVE_INTERVAL
                block = min(FRESH_BLOCK, until_save)
                if not until_plateau:
                    block = min(block, final_step - step + 1)
                fresh_batches = zip(*make_batch_block(fresh_count, block))
                batch = next(fresh_batches)
        else:
            batch = make_batch(fresh_count)
        if replay_count:
            replay = dataset.sample(replay_count, sample_rng)
            batch = tuple(np.concatenate((fresh, old)) for fresh, old in zip(batch, replay))
        train_batch(*batch)

        if fixed_run is not None:
            should_evaluate = (
                step == final_step or (step - fixed_run["start_update"]) % PLATEAU_INTERVAL == 0
            )
        elif plateau is None:
            should_evaluate = step in CHECKPOINTS
        else:
            should_evaluate = (step - plateau["start_update"]) % plateau["evaluation_interval"] == 0
        is_best = False
        if should_evaluate:
            result = evaluate(evaluation_games, predict, diagnose=fixed_run is not None)
            result.update(updates=step, failed_training_games=failures, replay_states=dataset.size)
            report["checkpoints"].append(result)
            if plateau is not None:
                is_best = track_plateau(plateau, result)
            if fixed_run is not None and result["wins"] > report["best"]["wins"]:
                is_best = True
                report["best"] = result
            print(
                f"{step:>7,} 步｜通关 {result['wins']}/{result['games']} ({result['win_rate']:.2%})"
                f"｜平均翻开 {result['mean_safe_revealed']:.2f}/{SAFE_CELLS} 格"
                f"｜已收录 {failures:,} 场失败对局、{dataset.size:,} 个状态",
                flush=True,
            )
            if plateau is not None:
                print(
                    f"最高胜率 {plateau['best']['win_rate']:.2%}"
                    f"｜距上次有效进展 {plateau['evaluations_without_gain']}/"
                    f"{plateau['patience_evaluations']} 次评估",
                    flush=True,
                )
            if fixed_run is not None:
                diagnostic = result["failure_diagnostics"]
                print(
                    f"失败 {diagnostic['failed_games']} 局中，"
                    f"{diagnostic['safe_move_available']} 局踩雷时已有确定安全格；"
                    f"{diagnostic['selected_provable_mine']} 局选择了可从数字推导出的雷。",
                    flush=True,
                )
        elif step % 5_000 == 0:
            target = "（训练至平台期）" if until_plateau else f"/{final_step:,}"
            print(
                f"进度 {step:,}{target} 步，已耗时 {perf_counter() - started:.0f} 秒。",
                flush=True,
            )

        if step % SAVE_INTERVAL == 0 or should_evaluate:
            if learner is not None:
                learner.copy_state_to(model)
            if is_best:
                save_checkpoint(best_path, step, dataset, failures, generators, report)
            save_checkpoint(checkpoint_path, step, dataset, failures, generators, report)
            if should_evaluate:
                save_report(output_path, report)

        if plateau is not None and plateau["status"] == "reached":
            print(f"已在第 {step:,} 步达到平台期。最高胜率权重已保存到：{best_path}", flush=True)
            break

    print(f"结果已保存：{output_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--until-plateau", action="store_true", help="从百万步存档继续，直至评估胜率进入平台期")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--fine-tune-best", action="store_true", help="从上一轮最佳权重开始，减半学习率并训练至平台期")
    source.add_argument("--resume-best", type=Path, help="从指定的完整最佳存档开始，保留 Adam 和失败数据池")
    parser.add_argument("--learning-rate", type=float, help="与 --resume-best 一起使用，设置续训学习率")
    parser.add_argument("--through-step", type=int, help="续训到这个固定步数，平台期不提前停止")
    args = parser.parse_args()
    if (args.resume_best is None) != (args.learning_rate is None):
        parser.error("--resume-best 和 --learning-rate 必须同时提供")
    if args.learning_rate is not None and (not np.isfinite(args.learning_rate) or args.learning_rate <= 0):
        parser.error("--learning-rate 必须是正的有限数")
    run_experiment(
        until_plateau=args.until_plateau,
        fine_tune_best=args.fine_tune_best,
        resume_best=args.resume_best,
        new_learning_rate=args.learning_rate,
        through_step=args.through_step,
    )
