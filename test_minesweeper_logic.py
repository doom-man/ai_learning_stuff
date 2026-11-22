"""回归测试覆盖确定雷位、确定安全格和专项训练梯度。"""

import unittest

import numpy as np
import torch

from minesweeper import count_adjacent_mines
from minesweeper_experiment import deduce_cells
from minesweeper_logic_training import corner_examples, make_corner_training_cases, rule_loss


class DeductionTests(unittest.TestCase):
    def test_one_with_only_one_hidden_neighbor_proves_mine(self):
        visible = corner_examples().visible[0]
        _, known_mines = deduce_cells(visible)
        self.assertEqual(known_mines, {10})

    def test_confirmed_mine_makes_other_neighbors_safe(self):
        visible = corner_examples().visible[0]
        known_safe, _ = deduce_cells(visible)
        self.assertEqual(known_safe, {2, 11, 18, 19})

    def test_one_with_three_hidden_neighbors_is_ambiguous(self):
        visible = np.full(81, 9)
        visible[0] = 1
        self.assertEqual(deduce_cells(visible), (set(), set()))

    def test_rotated_fixtures_match_real_mines_and_clues(self):
        visible, mines, known_safe, known_mines = corner_examples()
        clues = count_adjacent_mines(mines).reshape(-1, 81)
        revealed = visible != 9
        np.testing.assert_array_equal(visible[revealed], clues[revealed])
        self.assertFalse((mines & known_safe).any())
        self.assertFalse((~mines & known_mines).any())
        np.testing.assert_array_equal(known_safe.sum(axis=1), 4)
        np.testing.assert_array_equal(known_mines.sum(axis=1), 1)

    def test_corner_training_has_extra_clue_and_separate_inputs(self):
        cases = make_corner_training_cases(np.random.default_rng(17), 32)
        revealed = cases.visible != 9
        clues = count_adjacent_mines(cases.mines).reshape(-1, 81)
        np.testing.assert_array_equal(revealed.sum(axis=1), 4)
        np.testing.assert_array_equal(cases.visible[revealed], clues[revealed])
        self.assertTrue(cases.known_mines[:, 10].all())
        self.assertTrue(cases.known_safe[:, [2, 11, 18, 19]].all())


class TrainingLossTests(unittest.TestCase):
    def test_wrong_ranking_gradients_push_safe_down_and_mine_up(self):
        scores = torch.tensor([[2.0, -2.0, 4.0]], requires_grad=True)
        unknown = torch.tensor([[True, True, False]])
        safe = torch.tensor([[True, False, False]])
        mines = torch.tensor([[False, True, False]])

        loss = rule_loss(scores, unknown, safe, mines)
        loss.backward()
        self.assertGreater(scores.grad[0, 0].item(), 0)
        self.assertLess(scores.grad[0, 1].item(), 0)
        self.assertEqual(scores.grad[0, 2].item(), 0)

    def test_no_proven_cells_adds_finite_zero_loss(self):
        scores = torch.tensor([[1.0, -1.0]], requires_grad=True)
        unknown = torch.tensor([[True, True]])
        empty = torch.tensor([[False, False]])
        loss = rule_loss(scores, unknown, empty, empty)
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertTrue(torch.all(scores.grad == 0).item())

    def test_mine_only_ranking_spares_another_actual_safe_cell(self):
        scores = torch.tensor([[2.0, -2.0, -4.0]], requires_grad=True)
        unknown = torch.tensor([[True, True, True]])
        safe = torch.tensor([[True, False, False]])
        mines = torch.tensor([[False, True, False]])
        rule_loss(scores, unknown, safe, mines, mine_targets=mines).backward()
        self.assertGreater(scores.grad[0, 0].item(), 0)
        self.assertLess(scores.grad[0, 1].item(), 0)
        self.assertEqual(scores.grad[0, 2].item(), 0)


if __name__ == "__main__":
    unittest.main()
