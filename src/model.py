"""纯 NumPy 三层 MLP 模型，实现前向传播、反向传播和 SGD 更新。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ForwardCache:
    """保存反向传播需要复用的中间结果。"""

    x: np.ndarray
    z1: np.ndarray
    a1: np.ndarray
    z2: np.ndarray
    a2: np.ndarray
    logits: np.ndarray
    probs: np.ndarray


class MLPClassifier:
    """三层全连接神经网络分类器。

    网络结构为 ``输入层 -> 隐藏层1 -> 隐藏层2 -> 输出层``。
    所有梯度都在本类中显式推导和计算，不依赖自动微分框架。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim1: int,
        hidden_dim2: int,
        num_classes: int,
        activation: str = "relu",
        seed: int = 42,
    ) -> None:
        self.activation_name = activation
        self.rng = np.random.default_rng(seed)
        self.params = {
            "W1": self._init_weight(input_dim, hidden_dim1, activation),
            "b1": np.zeros(hidden_dim1, dtype=np.float32),
            "W2": self._init_weight(hidden_dim1, hidden_dim2, activation),
            "b2": np.zeros(hidden_dim2, dtype=np.float32),
            "W3": self._init_weight(hidden_dim2, num_classes, "linear"),
            "b3": np.zeros(num_classes, dtype=np.float32),
        }

    def _init_weight(self, fan_in: int, fan_out: int, activation: str) -> np.ndarray:
        """按激活函数选择初始化尺度。

        ReLU 使用 He 初始化，Sigmoid/Tanh 使用 Xavier 初始化，以降低前几轮训练中
        梯度过大或过小的概率。
        """

        if activation == "relu":
            scale = np.sqrt(2.0 / fan_in)
        else:
            scale = np.sqrt(1.0 / fan_in)
        return self.rng.normal(0.0, scale, size=(fan_in, fan_out)).astype(np.float32)

    def _activate(self, z: np.ndarray) -> np.ndarray:
        """计算隐藏层激活值。"""

        if self.activation_name == "relu":
            return np.maximum(z, 0.0)
        if self.activation_name == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))
        if self.activation_name == "tanh":
            return np.tanh(z)
        raise ValueError(f"不支持的激活函数：{self.activation_name}")

    def _activation_grad(self, z: np.ndarray, activated: np.ndarray) -> np.ndarray:
        """计算隐藏层激活函数对输入 z 的导数。"""

        if self.activation_name == "relu":
            return (z > 0).astype(np.float32)
        if self.activation_name == "sigmoid":
            return activated * (1.0 - activated)
        if self.activation_name == "tanh":
            return 1.0 - activated**2
        raise ValueError(f"不支持的激活函数：{self.activation_name}")

    def forward(self, x: np.ndarray) -> ForwardCache:
        """执行前向传播并返回缓存。"""

        z1 = x @ self.params["W1"] + self.params["b1"]
        a1 = self._activate(z1)
        z2 = a1 @ self.params["W2"] + self.params["b2"]
        a2 = self._activate(z2)
        logits = a2 @ self.params["W3"] + self.params["b3"]
        probs = softmax(logits)
        return ForwardCache(x=x, z1=z1, a1=a1, z2=z2, a2=a2, logits=logits, probs=probs)

    def loss_and_grads(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weight_decay: float = 0.0,
    ) -> tuple[float, dict[str, np.ndarray]]:
        """计算交叉熵损失和所有参数梯度。

        softmax 与交叉熵组合后，输出层 logits 的梯度可简化为
        ``(probs - one_hot) / batch_size``，这是手写反向传播中最关键的一步。
        L2 正则只作用在权重矩阵上，不作用在偏置上。
        """

        cache = self.forward(x)
        batch_size = x.shape[0]
        y_onehot = np.zeros_like(cache.probs)
        y_onehot[np.arange(batch_size), y] = 1.0

        data_loss = cross_entropy(cache.probs, y)
        reg_loss = 0.5 * weight_decay * sum(
            np.sum(self.params[name] ** 2) for name in ("W1", "W2", "W3")
        )
        loss = data_loss + reg_loss

        dlogits = (cache.probs - y_onehot) / batch_size
        grads: dict[str, np.ndarray] = {}
        grads["W3"] = cache.a2.T @ dlogits + weight_decay * self.params["W3"]
        grads["b3"] = np.sum(dlogits, axis=0)

        da2 = dlogits @ self.params["W3"].T
        dz2 = da2 * self._activation_grad(cache.z2, cache.a2)
        grads["W2"] = cache.a1.T @ dz2 + weight_decay * self.params["W2"]
        grads["b2"] = np.sum(dz2, axis=0)

        da1 = dz2 @ self.params["W2"].T
        dz1 = da1 * self._activation_grad(cache.z1, cache.a1)
        grads["W1"] = cache.x.T @ dz1 + weight_decay * self.params["W1"]
        grads["b1"] = np.sum(dz1, axis=0)
        return float(loss), grads

    def predict_proba(self, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """分批预测类别概率，避免一次性推理占用过多内存。"""

        probs = []
        for start in range(0, len(x), batch_size):
            probs.append(self.forward(x[start : start + batch_size]).probs)
        return np.vstack(probs)

    def predict(self, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """返回预测类别编号。"""

        return np.argmax(self.predict_proba(x, batch_size=batch_size), axis=1)

    def step(self, grads: dict[str, np.ndarray], learning_rate: float) -> None:
        """执行一次 SGD 参数更新。"""

        for name, grad in grads.items():
            self.params[name] -= learning_rate * grad.astype(np.float32)

    def state_dict(self) -> dict[str, np.ndarray | str]:
        """导出模型权重和激活函数名称。"""

        state: dict[str, np.ndarray | str] = {name: value.copy() for name, value in self.params.items()}
        state["activation"] = self.activation_name
        return state

    def load_state_dict(self, state: dict[str, np.ndarray | str]) -> None:
        """加载已保存的模型权重。"""

        self.activation_name = str(state["activation"])
        for name in ("W1", "b1", "W2", "b2", "W3", "b3"):
            self.params[name] = np.asarray(state[name], dtype=np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    """数值稳定的 softmax。"""

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def cross_entropy(probs: np.ndarray, y: np.ndarray) -> float:
    """计算平均交叉熵损失。"""

    clipped = np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)
    return float(-np.mean(np.log(clipped)))
