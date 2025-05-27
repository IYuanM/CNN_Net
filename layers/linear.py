import numpy as np


class Linear:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        # 使用He初始化
        scale = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(in_features, out_features) * scale
        self.b = np.zeros(out_features)

        self.dW = None
        self.db = None
        self.cache = None

    def forward(self, x):
        """
        前向传播
        x: (N, in_features) 形状的输入
        """
        # 保存输入用于反向传播
        self.cache = x

        # 线性变换
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, grad):
        """
        反向传播
        grad: 上游梯度
        """
        x = self.cache

        # 计算权重和偏置的梯度
        self.dW = np.dot(x.T, grad)
        self.db = np.sum(grad, axis=0)

        # 计算输入的梯度
        dx = np.dot(grad, self.W.T)

        return dx