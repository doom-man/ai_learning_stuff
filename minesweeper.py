"""用 NumPy 训练一个多层神经网络，在 9×9 扫雷棋盘上选择下一格。"""

import json
from pathlib import Path

import numpy as np


RNG = np.random.default_rng(42)


def sigmoid(values):
    """数值稳定的 sigmoid。"""
    values = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-values))


class MinesweeperModel:
    """810 → 256 → 81 的全连接网络，使用 NumPy 手动反向传播和 Adam。"""

    def __init__(self):
        dimensions = (810, 256, 81)
        self.parameters = {}
        for layer, (fan_in, fan_out) in enumerate(zip(dimensions[:-1], dimensions[1:]), start=1):
            # 与常见线性层的均匀初始化相同，权重和偏置范围均为 ±1/sqrt(fan_in)。
            limit = 1.0 / np.sqrt(fan_in)
            self.parameters[f"W{layer}"] = RNG.uniform(
                -limit, limit, (fan_in, fan_out)
            ).astype(np.float32)
            self.parameters[f"b{layer}"] = RNG.uniform(
                -limit, limit, fan_out
            ).astype(np.float32)

        self.first_moment = {
            name: np.zeros_like(value) for name, value in self.parameters.items()
        }
        self.second_moment = {
            name: np.zeros_like(value) for name, value in self.parameters.items()
        }
        largest_parameter = max(value.size for value in self.parameters.values())
        self._adam_work = np.empty(largest_parameter, dtype=np.float32)
        self._adam_denom = np.empty_like(self._adam_work)
        self.adam_step = 0

    def forward(self, x):
        """正向计算，返回 81 个格子的 logits 和反向传播所需缓存。"""
        # ① 输入层：接收已编码的棋盘，每张 81 格 × 10 个状态 = 810 个数。
        # x 的形状为 [批大小, 810]，转成 float32 后直接传给隐藏层。
        x = np.asarray(x, dtype=np.float32)

        # ② 隐藏层：256 个神经元各自对 810 个输入做加权求和，再加偏置。
        # [批大小, 810] @ W1[810, 256] + b1[256] → z1[批大小, 256]。
        z1 = x @ self.parameters["W1"] + self.parameters["b1"]
        # ReLU：正数保留，负数归零，得到 256 个隐藏层激活值。
        a1 = np.maximum(z1, 0.0)

        # ③ 输出层：对 256 个激活值做加权求和，给棋盘每格生成一个分数。
        # [批大小, 256] @ W2[256, 81] + b2[81] → logits[批大小, 81]。
        # logits 是原始分数，经过 sigmoid 才是每格的踩雷概率。
        logits = a1 @ self.parameters["W2"] + self.parameters["b2"]
        return logits, (x, z1, a1)

    def update(self, gradients, learning_rate=1e-3, *, update_norms=None):
        """用 Adam 更新权重和偏置。"""
        self.adam_step += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        correction1 = 1.0 - beta1**self.adam_step
        correction2 = 1.0 - beta2**self.adam_step
        for name, gradient in gradients.items():
            first = self.first_moment[name]
            second = self.second_moment[name]
            work = self._adam_work[:gradient.size].reshape(gradient.shape)
            denominator = self._adam_denom[:gradient.size].reshape(gradient.shape)
            first *= beta1
            np.multiply(gradient, 1.0 - beta1, out=work)
            first += work
            second *= beta2
            np.square(gradient, out=work)
            work *= 1.0 - beta2
            second += work

            # 复用数组，减少长时间训练中反复创建的大型中间结果。
            np.divide(second, correction2, out=denominator)
            np.sqrt(denominator, out=denominator)
            denominator += epsilon
            np.divide(first, correction1, out=work)
            work *= learning_rate
            work /= denominator
            parameter = self.parameters[name]
            if update_norms is not None:
                before = parameter.copy()
            parameter -= work
            if update_norms is not None:
                update_norms[name] = float(np.linalg.norm(parameter - before))


model = MinesweeperModel()


def activation_stats(values):
    """记录正向张量的形状和数值范围，供训练流程图使用。"""
    return {
        "shape": list(values.shape),
        "min": float(values.min()),
        "mean": float(values.mean()),
        "max": float(values.max()),
    }


def sample_trace(inputs, z1, a1, logits, dlogits, da1, dz1, mines, unknown):
    """保存本批第一张棋盘的各层信号，以及两条示例连接的更新前权重。"""
    # 输入不是模型参数。这个梯度只供回放展示，不参与 Adam 更新。
    dx = dz1[0] @ model.parameters["W1"].T
    input_index = int(inputs[0, :10].argmax())
    hidden_index = int(a1[0].argmax())
    output_index = int(np.argmax(unknown[0]))
    return {
        "activations": {
            "X": inputs[0].astype(np.int8).tolist(),
            "Z1": z1[0].tolist(),
            "A1": a1[0].tolist(),
            "logits": logits[0].tolist(),
        },
        "gradients": {
            "X": dx.tolist(),
            "Z1": dz1[0].tolist(),
            "A1": da1[0].tolist(),
            "logits": dlogits[0].tolist(),
        },
        "mines": mines[0].tolist(),
        "unknown": unknown[0].tolist(),
        "example": {
            "input": input_index,
            "hidden": hidden_index,
            "output": output_index,
            "w1": float(model.parameters["W1"][input_index, hidden_index]),
            "w2": float(model.parameters["W2"][hidden_index, output_index]),
        },
    }


def train_step(x, mines, unknown, *, trace=None):
    """训练一步：用雷区答案计算误差，并据此更新模型参数。

    x 的形状为 [批大小, 810]，雷区答案和未知格掩码为 [批大小, 81]。
    """
    # ===== 正向传播：从棋盘到损失 =====
    logits, (inputs, z1, a1) = model.forward(x)
    # 只取未翻开的格子，真实标签为雷 1、安全 0。
    targets = mines[unknown].astype(np.float32)
    selected_logits = logits[unknown]
    # 直接用 logits 计算二元交叉熵，对整批未知格求平均。
    loss = np.mean(np.logaddexp(0.0, selected_logits) - targets * selected_logits)

    # ===== 反向传播：从损失算出各层梯度 =====
    # 只对未知格计算损失，因此已翻开格子的梯度为 0。
    dlogits = np.zeros_like(logits)
    # 未知格的梯度 = (预测踩雷概率 - 真实标签) / 整批未知格数。
    dlogits[unknown] = (sigmoid(selected_logits) - targets) / targets.size

    # 输出层：用隐藏层激活和输出梯度算 W2、b2 的梯度。
    # 再用 W2 的转置把梯度传回隐藏层，得到 da1。
    gradients = {}
    gradients["W2"] = a1.T @ dlogits
    gradients["b2"] = dlogits.sum(axis=0)
    da1 = dlogits @ model.parameters["W2"].T

    # 隐藏层：ReLU 只让 z1 > 0 的位置传递梯度。
    # 用原始输入和通过 ReLU 的梯度算 W1、b1 的梯度。
    dz1 = da1 * (z1 > 0.0)
    gradients["W1"] = inputs.T @ dz1
    gradients["b1"] = dz1.sum(axis=0)

    update_norms = None
    if trace is not None:
        update_norms = {}
        trace.update(
            loss=float(loss),
            unknown_count=int(targets.size),
            activations={
                name: activation_stats(values)
                for name, values in (
                    ("X", inputs), ("Z1", z1), ("A1", a1), ("logits", logits),
                )
            },
            gradient_norms={
                name: float(np.linalg.norm(values))
                for name, values in (
                    ("logits", dlogits), ("A1", da1), ("Z1", dz1),
                )
            },
            parameter_gradient_norms={
                name: float(np.linalg.norm(values))
                for name, values in gradients.items()
            },
            update_norms=update_norms,
            sample=sample_trace(inputs, z1, a1, logits, dlogits, da1, dz1, mines, unknown),
        )

    # ===== 参数更新：Adam 根据梯度调整权重和偏置 =====
    model.update(gradients, update_norms=update_norms)
    return float(loss)


def choose_cell(x, unknown):
    """预测：估计未知格是雷的风险，返回风险最低格子的行、列（从 0 开始）。"""
    logits, _ = model.forward(np.asarray(x)[None, :])
    risk = sigmoid(logits[0])
    # 排除已翻开的格子，选踩雷概率最低的未知格，再换算为行和列。
    risk = np.where(unknown, risk, np.inf)
    return divmod(int(np.argmin(risk)), 9)


def count_adjacent_mines(mines):
    """用 3×3 窗口统计每格周围八格的雷数。"""
    mine_grid = mines.reshape(-1, 9, 9).astype(np.uint8)
    # 先纵向求和，再横向求和，正好覆盖 3×3 窗口。
    vertical = mine_grid.copy()
    vertical[:, 1:] += mine_grid[:, :-1]
    vertical[:, :-1] += mine_grid[:, 1:]
    clues = vertical.copy()
    clues[:, :, 1:] += vertical[:, :, :-1]
    clues[:, :, :-1] += vertical[:, :, 1:]
    return clues - mine_grid


def expand_zero_regions(revealed, clues):
    """已翻开的 0 会展开周围格子，直到没有新的格子可以展开。"""
    is_zero = clues == 0
    zeros = revealed & is_zero
    if not zeros.any():
        return revealed

    while True:
        vertical = zeros.copy()
        vertical[:, 1:] |= zeros[:, :-1]
        vertical[:, :-1] |= zeros[:, 1:]
        neighbors = vertical.copy()
        neighbors[:, :, 1:] |= vertical[:, :, :-1]
        neighbors[:, :, :-1] |= vertical[:, :, 1:]
        next_zeros = neighbors & is_zero
        if np.array_equal(next_zeros, zeros):
            return revealed | neighbors
        zeros = next_zeros


def encode_board(visible):
    """把可见棋盘编码为模型需要的输入。未翻开的格用状态 9。"""
    # 用单位矩阵取出每个状态的 one-hot 向量，例如数字 2 只把第 3 位置为 1。
    # [批大小, 9, 9, 10] 按行展开为 [批大小, 810]，作为网络输入。
    return np.eye(10, dtype=np.float32)[visible].reshape(visible.shape[0], 810)


def _batch_from_scores(mine_scores, safe_scores):
    batch_size = len(mine_scores)
    mines = np.zeros((batch_size, 81), dtype=bool)
    mine_positions = np.argsort(mine_scores, axis=1)[:, :10]
    batch_indices = np.arange(batch_size)[:, None]
    mines[batch_indices, mine_positions] = True
    clues = count_adjacent_mines(mines)

    # 随机翻开四个安全格，再按扫雷规则展开 0 周围的格子。
    safe_scores[mines] = 2.0
    safe_clicks = np.argsort(safe_scores, axis=1)[:, :4]
    revealed = np.zeros((batch_size, 81), dtype=bool)
    revealed[batch_indices, safe_clicks] = True
    revealed = expand_zero_regions(revealed.reshape(batch_size, 9, 9), clues)

    # 输入编码：已翻开的格保留数字 0～8，未翻开的格统一用状态 9。
    x = encode_board(np.where(revealed, clues, 9))
    return x, mines, ~revealed.reshape(batch_size, 81)


def make_batch(batch_size):
    """生成训练样本。每格用 10 个数编码，状态 0～8 为周围雷数，9 为未翻开。"""
    shape = (batch_size, 81)
    return _batch_from_scores(RNG.random(shape), RNG.random(shape))


def make_batch_block(batch_size, steps):
    """按原来的随机数顺序一次生成多个训练批次。"""
    scores = RNG.random((steps, 2, batch_size, 81))
    batches = _batch_from_scores(
        scores[:, 0].reshape(-1, 81),
        scores[:, 1].reshape(-1, 81),
    )
    return tuple(values.reshape(steps, batch_size, -1) for values in batches)


def play_game(seed=2026):
    """让已训练的模型从空棋盘开始玩一局，直到胜利或踩雷。"""
    game_rng = np.random.default_rng(seed)
    revealed = np.zeros((1, 9, 9), dtype=bool)
    mines = np.zeros_like(revealed)
    clues = np.zeros_like(revealed, dtype=np.int32)
    safe_total = 81 - 10
    revealed_count = 0
    hit = None
    turn = 0

    print("\n完整对局：9×9，10 颗雷。首格保证安全，行列从 1 开始。")
    while revealed_count < safe_total:
        # 模型只接收可见棋盘；选格时用掩码排除已翻开的格子。
        visible = np.where(revealed, clues, 9)
        x = encode_board(visible)[0]
        row, col = choose_cell(x, ~revealed.ravel())
        cell = (0, row, col)
        turn += 1

        if turn == 1:
            # 首次选格后再布雷，从其余 80 格随机抽取 10 格。
            candidates = np.delete(np.arange(81), row * 9 + col)
            mines.flat[game_rng.choice(candidates, size=10, replace=False)] = True
            clues = count_adjacent_mines(mines)

        move = f"第 {turn:2d} 步｜第 {row + 1} 行、第 {col + 1} 列"
        if mines[cell]:
            hit = (row, col)
            print(f"{move}｜踩雷")
            break

        revealed[cell] = True
        revealed = expand_zero_regions(revealed, clues)
        revealed_count = int(revealed.sum())
        print(f"{move}｜翻开 {clues[cell]}，已翻开 {revealed_count}/{safe_total} 个安全格")

    final_board = np.where(revealed, clues, 9)[0].astype(str)
    final_board[~revealed[0]] = "·"
    if hit is not None:
        final_board[hit] = "×"
    print("\n终局棋盘（· 未翻开，× 踩雷）")
    for line in final_board:
        print(" ".join(line))
    result = "胜利" if revealed_count == safe_total else "失败"
    print(f"结果：{result}，共点击 {turn} 次，翻开 {revealed_count}/{safe_total} 个安全格。")


def save_training_plot(losses):
    """用标准库生成训练损失 SVG：浅色线为每批损失，蓝线为 10 步滑动平均。"""
    output_path = Path(__file__).with_name("training_loss.svg")
    width, height = 960, 560
    left, right, top, bottom = 86, 30, 58, 72
    plot_width = width - left - right
    plot_height = height - top - bottom

    values = np.asarray(losses, dtype=np.float64)
    low, high = float(values.min()), float(values.max())
    spread = high - low
    padding = spread * 0.12 if spread else max(abs(high) * 0.08, 0.01)
    low -= padding
    high += padding

    def point(step_index, value):
        px = left + step_index / max(len(values) - 1, 1) * plot_width
        py = top + (high - value) / (high - low) * plot_height
        return px, py

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="86" y="34" font-family="sans-serif" font-size="22" font-weight="600" fill="#172033">Training loss</text>',
    ]

    for tick in np.linspace(low, high, 5):
        py = top + (high - tick) / (high - low) * plot_height
        elements.append(
            f'<line x1="{left}" y1="{py:.2f}" x2="{width - right}" y2="{py:.2f}" stroke="#e7eaf0"/> '
            f'<text x="{left - 12}" y="{py + 5:.2f}" text-anchor="end" font-family="sans-serif" font-size="13" fill="#596579">{tick:.3f}</text>'
        )

    elements.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#758096"/>',
            f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#758096"/>',
        ]
    )

    for step in np.linspace(1, len(values), 6, dtype=int):
        px, _ = point(step - 1, low)
        elements.append(
            f'<line x1="{px:.2f}" y1="{height - bottom}" x2="{px:.2f}" y2="{height - bottom + 5}" stroke="#758096"/> '
            f'<text x="{px:.2f}" y="{height - bottom + 24}" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#596579">{step}</text>'
        )

    batch_points = " ".join(
        f"{point(index, value)[0]:.2f},{point(index, value)[1]:.2f}"
        for index, value in enumerate(values)
    )
    elements.append(
        f'<polyline points="{batch_points}" fill="none" stroke="#a9b8cc" stroke-width="1.5" opacity="0.8"/>'
    )

    window = min(10, len(values))
    moving_average = np.convolve(values, np.ones(window) / window, mode="valid")
    average_points = " ".join(
        f"{point(index + window - 1, value)[0]:.2f},{point(index + window - 1, value)[1]:.2f}"
        for index, value in enumerate(moving_average)
    )
    elements.append(
        f'<polyline points="{average_points}" fill="none" stroke="#2563eb" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
    )
    elements.extend(
        [
            f'<text x="{left + plot_width / 2:.2f}" y="{height - 16}" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#344054">Training step</text>',
            f'<text x="23" y="{top + plot_height / 2:.2f}" text-anchor="middle" transform="rotate(-90 23 {top + plot_height / 2:.2f})" font-family="sans-serif" font-size="14" fill="#344054">Binary cross-entropy loss</text>',
            f'<line x1="{width - 300}" y1="35" x2="{width - 275}" y2="35" stroke="#a9b8cc" stroke-width="2"/>',
            f'<text x="{width - 268}" y="40" font-family="sans-serif" font-size="12" fill="#596579">Batch loss</text>',
            f'<line x1="{width - 180}" y1="35" x2="{width - 155}" y2="35" stroke="#2563eb" stroke-width="3"/>',
            f'<text x="{width - 148}" y="40" font-family="sans-serif" font-size="12" fill="#596579">10-step average</text>',
            '</svg>',
        ]
    )
    output_path.write_text("\n".join(elements), encoding="utf-8")
    return output_path


def save_training_trace(traces):
    """把本次训练的每一步数据嵌进 HTML，打开后可离线查看。"""
    output_path = Path(__file__).with_name("training_trace.html")
    template_path = output_path.with_name("training_trace_template.html")
    template = template_path.read_text(encoding="utf-8")
    data = json.dumps(traces, separators=(",", ":"), allow_nan=False)
    data_tag = '<script type="application/json" id="trace-data">'
    output_path.write_text(
        template.replace(f"{data_tag}[]</script>", f"{data_tag}{data}</script>", 1),
        encoding="utf-8",
    )
    return output_path


def main():
    # ===== 训练阶段 =====
    # 生成带有真实雷区答案的棋盘样本，记录每步正向传播、反向传播和参数更新。
    traces = []
    for step in range(1, 101):
        x, mines, unknown = make_batch(32)
        trace = {}
        loss = train_step(x, mines, unknown, trace=trace)
        traces.append(trace)
        if step in (1, 25, 50, 75, 100):
            print(f"训练 {step:3d}/100 步，当前批次损失 {loss:.4f}")

    plot_path = save_training_plot([trace["loss"] for trace in traces])
    print(f"训练损失图已保存：{plot_path}")
    trace_path = save_training_trace(traces)
    print(f"训练流程图已保存：{trace_path}")

    # ===== 预测阶段：用刚训练好的模型玩完整一局 =====
    play_game()


if __name__ == "__main__":
    main()
