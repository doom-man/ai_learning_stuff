"""用可证明的安全格和雷位补充监督，并在独立棋盘上评估选格。"""

import argparse
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch.nn import functional as F

from minesweeper import RNG, count_adjacent_mines, encode_board, expand_zero_regions, make_batch_block, model, sigmoid
from minesweeper_experiment import (
    BATCH_SIZE,
    EVALUATION_GAMES,
    REPLAY_PER_BATCH,
    SEED,
    SELF_PLAY_GAMES,
    SELF_PLAY_INTERVAL,
    FailedGameDataset,
    TorchLearner,
    deduce_cells,
    evaluate,
    load_checkpoint,
    make_games,
    play_games,
    save_checkpoint,
    save_report,
)


UPDATES = (10_000, 25_000, 50_000)
RULE_EXAMPLES_PER_BATCH = 8
CASE_POOL_SIZE = 48_000
HOLDOUT_SIZE = 1_000
ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "minesweeper_finetune_lr0.00025_best.npz"
HISTORICAL_FAILURE_SOURCE = ROOT / "minesweeper_finetune_lr0.00025_checkpoint.npz"
BOARD_INDICES = np.arange(81).reshape(9, 9)
SYMMETRIES = np.stack([
    np.rot90(board, turn).ravel()
    for board in (BOARD_INDICES, np.fliplr(BOARD_INDICES))
    for turn in range(4)
])


class RuleCases(NamedTuple):
    visible: np.ndarray
    mines: np.ndarray
    known_safe: np.ndarray
    known_mines: np.ndarray


def annotate(visible, mines):
    """推理标签只取自模型也能看到的数字。真实雷区用于另行评分。"""
    known_safe = np.zeros(visible.shape, dtype=bool)
    known_mines = np.zeros_like(known_safe)
    for row, board in enumerate(visible):
        safe_cells, mine_cells = deduce_cells(board)
        known_safe[row, list(safe_cells)] = True
        known_mines[row, list(mine_cells)] = True
    return RuleCases(visible, mines, known_safe, known_mines)


def corner_examples():
    """角落的 1 确定斜对角是雷，旁边两个 1 随后确定数格安全。"""
    mines = np.zeros(81, dtype=bool)
    mines[[10, 30, 31, 34, 40, 43, 50, 53, 66, 78]] = True
    visible = np.full(81, 9, dtype=np.uint8)
    visible[[0, 1, 9]] = 1
    return annotate(visible[SYMMETRIES], mines[SYMMETRIES])


def rule_loss(logits, unknown, known_safe, known_mines, *, mine_targets=None, weight=0.3):
    """对确定格加权监督，并把至少一个确定安全格推到其他未知格前面。"""
    certified = known_safe | known_mines
    cells = F.binary_cross_entropy_with_logits(logits, known_mines.to(logits.dtype), reduction="none")
    certified_loss = (cells * certified).sum() / certified.sum().clamp_min(1)

    other = unknown & ~known_safe
    if mine_targets is not None:
        other = other & mine_targets
    can_compare = known_safe.any(dim=1) & other.any(dim=1)
    scores = logits[can_compare]
    safe_score = scores.masked_fill(~known_safe[can_compare], torch.inf).amin(dim=1)
    other_score = scores.masked_fill(~other[can_compare], torch.inf).amin(dim=1)
    ranking_loss = F.relu(0.5 + safe_score - other_score).sum() / can_compare.sum().clamp_min(1)
    return weight * certified_loss + weight * ranking_loss


def choose_from_cases(cases, predict):
    parts = []
    for start in range(0, len(cases.visible), 256):
        visible = cases.visible[start:start + 256]
        logits = predict(encode_board(visible))
        parts.append(np.where(visible == 9, sigmoid(logits), np.inf).argmin(axis=1))
    return np.concatenate(parts)


def measure_cases(cases, predict):
    picked = choose_from_cases(cases, predict)
    rows = np.arange(len(picked))
    return {
        "cases": len(picked),
        "selected_provably_safe": int(cases.known_safe[rows, picked].sum()),
        "selected_actual_mine": int(cases.mines[rows, picked].sum()),
        "selected_provable_mine": int(cases.known_mines[rows, picked].sum()),
    }


def make_holdout(rng, size):
    """生成独立雷区，每张保留一个有确定安全格的可达状态。"""
    parts = []
    collected = 0
    while collected < size:
        mines, _ = make_games(rng, 128)
        clues = count_adjacent_mines(mines)
        order = np.where(mines, np.inf, rng.random(mines.shape)).argsort(axis=1)
        open_count = rng.integers(1, 15, size=len(mines))
        revealed = np.zeros_like(mines)
        revealed[np.arange(len(mines))[:, None], order] = np.arange(81) < open_count[:, None]
        revealed = expand_zero_regions(revealed.reshape(-1, 9, 9), clues)
        visible = np.where(revealed, clues, 9).reshape(-1, 81)
        cases = annotate(visible, mines)
        eligible = cases.known_safe.any(axis=1)
        parts.append(tuple(values[eligible] for values in cases))
        collected += int(eligible.sum())
    return RuleCases(*(np.concatenate(values)[:size] for values in zip(*parts)))


def make_train_pool(dataset, rng, predict):
    indices = rng.choice(dataset.size, size=min(CASE_POOL_SIZE, dataset.size), replace=False)
    visible = dataset.visible[indices]
    mines = np.unpackbits(dataset.mine_bits[indices], axis=1, count=81).view(bool)
    annotated = annotate(visible, mines)
    eligible = (annotated.known_safe | annotated.known_mines).any(axis=1)
    cases = RuleCases(*(values[eligible] for values in annotated))
    picks = choose_from_cases(cases, predict)
    rows = np.arange(len(picks))
    hard = np.flatnonzero(cases.known_safe.any(axis=1) & cases.mines[rows, picks])
    return cases, hard


def make_corner_training_cases(rng, size):
    """生成带额外线索的角落训练棋盘，保留仅有 3 个 1 的棋盘作为测试。"""
    mines = np.zeros((size, 81), dtype=bool)
    protected = [0, 1, 2, 9, 10, 11, 18, 19]
    scores = rng.random(mines.shape)
    scores[:, protected] = np.inf
    mines[np.arange(size)[:, None], scores.argsort(axis=1)[:, :9]] = True
    mines[:, 10] = True
    clues = count_adjacent_mines(mines).reshape(-1, 81)
    reveal_scores = np.where(~mines & (clues > 0), rng.random(mines.shape), np.inf)
    reveal_scores[:, protected] = np.inf
    extra = reveal_scores.argmin(axis=1)
    visible = np.full_like(clues, 9)
    visible[:, [0, 1, 9]] = 1
    visible[np.arange(size), extra] = clues[np.arange(size), extra]
    return annotate(visible, mines)


def sample_rule_batch(cases, hard, rng, size=RULE_EXAMPLES_PER_BATCH):
    indices = rng.integers(len(cases.visible), size=size)
    if hard.size:
        count = size // 2
        indices[:count] = rng.choice(hard, size=count)
    columns = SYMMETRIES[rng.integers(len(SYMMETRIES), size=len(indices))]
    return RuleCases(*(values[indices[:, None], columns] for values in cases))


def train_learner(*, balanced=False, guided=False):
    balanced = balanced or guided
    self_play_rng, replay_rng, evaluation_rng = [
        np.random.default_rng(seed) for seed in np.random.SeedSequence(SEED).spawn(3)
    ]
    evaluation_games = make_games(evaluation_rng, EVALUATION_GAMES)
    training_cases_rng = np.random.default_rng(26831)
    holdout = make_holdout(np.random.default_rng(90402), HOLDOUT_SIZE)
    corners = corner_examples()

    # 把 200 万步模型曾经踩雷的局面固定下来，只做回归测试。
    with np.load(HISTORICAL_FAILURE_SOURCE, allow_pickle=False) as archive:
        for name, values in model.parameters.items():
            values[:] = archive[f"param_{name}"]
    historical = []
    historical_learner = TorchLearner(model)
    evaluate(evaluation_games, historical_learner.predict, failure_cases=historical)
    historical_cases = annotate(*(np.stack(values) for values in zip(*historical)))

    dataset = FailedGameDataset()
    generators = (RNG, self_play_rng, replay_rng, training_cases_rng)
    start_step, failures, previous_report = load_checkpoint(SOURCE, dataset, generators)
    learning_rate = 0.0001 if balanced else previous_report["learning_rate"]
    updates = (5_000, 10_000, 15_000) if guided else (5_000, 10_000, 20_000) if balanced else UPDATES
    learner = TorchLearner(model, learning_rate)
    report = {
        "source": SOURCE.name,
        "seed": SEED,
        "learning_rate": learning_rate,
        "cpu_threads": torch.get_num_threads(),
        "rule_examples_per_batch": RULE_EXAMPLES_PER_BATCH,
        "rule_loss_weight": 0.1 if balanced else 0.3,
        "rank_against": "actual_mines" if balanced else "all_other_unknown",
        "checkpoints": [],
    }
    stem = "minesweeper_logic"
    if guided:
        stem += "_guided"
    elif balanced:
        stem += "_balanced"
    report_path = ROOT / f"{stem}_results.json"
    best_path = ROOT / f"{stem}_best.npz"
    final_path = ROOT / f"{stem}_checkpoint.npz"

    def assess(step):
        result = evaluate(evaluation_games, learner.predict, diagnose=True)
        result.update(
            updates=step,
            failed_training_games=failures,
            replay_states=dataset.size,
            forced_choice=measure_cases(holdout, learner.predict),
            historical_failures=measure_cases(historical_cases, learner.predict),
            corner_examples=measure_cases(corners, learner.predict),
        )
        report["checkpoints"].append(result)
        forced = result["forced_choice"]
        old_failures = result["historical_failures"]
        print(
            f"{step:,} 步｜整局 {result['wins']}/{result['games']}"
            f"｜专项测试选确定安全格 {forced['selected_provably_safe']}/{forced['cases']}"
            f"、踩雷 {forced['selected_actual_mine']} 次"
            f"｜历史失败局面踩雷 {old_failures['selected_actual_mine']}/{old_failures['cases']}",
            flush=True,
        )
        return result

    def selection_score(result):
        forced = result["forced_choice"]
        if guided:
            return (
                result["corner_examples"]["selected_provably_safe"],
                result["wins"],
                forced["selected_provably_safe"],
            )
        if balanced:
            return result["wins"], forced["selected_provably_safe"], -forced["selected_actual_mine"]
        return forced["selected_provably_safe"], -forced["selected_actual_mine"], result["wins"]

    baseline = assess(start_step)
    report["best"] = baseline
    save_checkpoint(best_path, start_step, dataset, failures, generators, report)
    save_report(report_path, report)
    train_cases, hard = make_train_pool(dataset, training_cases_rng, learner.predict)
    corner_pool = make_corner_training_cases(training_cases_rng, 2_000) if guided else None
    report.update(training_rule_states=len(train_cases.visible), training_hard_states=len(hard))
    if corner_pool is not None:
        report["training_corner_states"] = len(corner_pool.visible)
        report["corner_cases_per_batch"] = RULE_EXAMPLES_PER_BATCH // 2
    print(
        f"专项训练池 {len(train_cases.visible):,} 个状态，其中 {len(hard):,} 个是起始模型"
        f"有确定安全格却选雷的局面。CPU 线程数：{torch.get_num_threads()}。",
        flush=True,
    )

    fresh_batches = iter(())
    guided_count = RULE_EXAMPLES_PER_BATCH // 2 if guided else 0
    ordinary_rule_count = RULE_EXAMPLES_PER_BATCH - guided_count
    no_hard_cases = np.empty(0, dtype=np.intp)
    for offset in range(1, updates[-1] + 1):
        if (offset - 1) % SELF_PLAY_INTERVAL == 0:
            won, _ = play_games(
                *make_games(self_play_rng, SELF_PLAY_GAMES), predict=learner.predict, failed_dataset=dataset
            )
            failures += len(won) - int(won.sum())

        fresh = next(fresh_batches, None)
        if fresh is None:
            fresh_batches = zip(*make_batch_block(BATCH_SIZE - REPLAY_PER_BATCH, 25))
            fresh = next(fresh_batches)
        replay = dataset.sample(REPLAY_PER_BATCH - RULE_EXAMPLES_PER_BATCH, replay_rng)
        rule = sample_rule_batch(train_cases, hard, training_cases_rng, ordinary_rule_count)
        if guided:
            examples = sample_rule_batch(corner_pool, no_hard_cases, training_cases_rng, guided_count)
            rule = RuleCases(*(np.concatenate(parts) for parts in zip(rule, examples)))
        targeted = encode_board(rule.visible), rule.mines, rule.visible == 9
        batch = tuple(np.concatenate(parts) for parts in zip(fresh, replay, targeted))
        safe, mines = torch.from_numpy(rule.known_safe), torch.from_numpy(rule.known_mines)
        mine_targets = torch.from_numpy(rule.mines) if balanced else None

        def extra_loss(logits, unknown):
            size = len(safe)
            if guided:
                ordinary = slice(-size, -guided_count)
                targeted_corner = slice(-guided_count, None)
                return rule_loss(
                    logits[ordinary], unknown[ordinary], safe[:ordinary_rule_count],
                    mines[:ordinary_rule_count],
                    mine_targets=mine_targets[:ordinary_rule_count], weight=report["rule_loss_weight"],
                ) + rule_loss(
                    logits[targeted_corner], unknown[targeted_corner],
                    safe[-guided_count:], mines[-guided_count:], weight=0.1,
                )
            return rule_loss(
                logits[-size:], unknown[-size:], safe, mines,
                mine_targets=mine_targets, weight=report["rule_loss_weight"],
            )

        learner.train(*batch, extra_loss=extra_loss)
        if offset in updates:
            step = start_step + offset
            result = assess(step)
            learner.copy_state_to(model)
            if selection_score(result) > selection_score(report["best"]):
                report["best"] = result
                save_checkpoint(best_path, step, dataset, failures, generators, report)
            save_checkpoint(final_path, step, dataset, failures, generators, report)
            save_report(report_path, report)
        elif offset % 5_000 == 0:
            print(f"专项训练进度 {offset:,}/{updates[-1]:,} 步。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument("--balanced", action="store_true", help="降低专项损失权重，只与真实雷排序")
    variants.add_argument("--guided", action="store_true", help="加入额外线索的角落棋盘，保留原角落棋盘只作测试")
    args = parser.parse_args()
    train_learner(balanced=args.balanced, guided=args.guided)
