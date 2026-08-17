# 02_2.1.3_回归任务

"""
Lecture: 2_第二部分_机器学习基础/2.1_机器学习任务
Content: 02_2.1.3_回归任务
"""

import numpy as np
from typing import List


def mse(y: np.ndarray, pred: np.ndarray) -> float:
    """Mean squared error."""
    return float(np.mean((y - pred) ** 2))


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mse(y, pred)))


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(y - pred)))


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    """Coefficient of determination (~1 is best)."""
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


class LinearRegression:
    """Linear regression f(x) = x^T w + b, trained by gradient descent."""

    def __init__(self, n_features: int, learning_rate: float = 0.01) -> None:
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.lr = learning_rate

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 2000) -> List[float]:
        losses = []
        n = len(X)
        for _ in range(epochs):
            pred = self.predict(X)
            losses.append(mse(y, pred))
            err = pred - y
            self.w -= self.lr * (2 / n) * (X.T @ err)
            self.b -= self.lr * (2 / n) * err.sum()
        return losses


def main() -> None:
    print("回归任务 (Regression Task) Demo")
    # 简单线性关系 y = 3x + 2 + 噪声
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 10, size=(200, 1))
    y = 3 * X[:, 0] + 2 + rng.normal(0, 1, size=200)

    model = LinearRegression(n_features=1)
    losses = model.fit(X, y, epochs=2000)
    pred = model.predict(X)
    print(f"初始MSE={losses[0]:.4f}, 最终MSE={losses[-1]:.4f}")
    print(f"MSE={mse(y, pred):.4f}  RMSE={rmse(y, pred):.4f}  MAE={mae(y, pred):.4f}")
    print(f"R^2 = {r2(y, pred):.4f}")
    print(f"学到的参数: w={model.w[0]:.3f} (真值3), b={model.b:.3f} (真值2)")


if __name__ == "__main__":
    main()