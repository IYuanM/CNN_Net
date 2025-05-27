import numpy as np


class BatchNorm2D:
    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.training = True  # 添加训练模式标志

        # 可学习参数
        self.gamma = np.ones(num_features)  # 缩放参数
        self.beta = np.zeros(num_features)  # 平移参数

        # 运行时统计量
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        # 反向传播需要的中间变量
        self.cache = None
        self.x_norm = None

    def forward(self, x):
        """
        前向传播
        x: (N, C, H, W) 形状的输入
        """
        N, C, H, W = x.shape

        if self.training:
            # 训练模式：使用当前批次的统计量
            mean = np.mean(x, axis=(0, 2, 3))
            var = np.var(x, axis=(0, 2, 3))

            # 更新运行时统计量
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            # 推理模式：使用运行时统计量
            mean = self.running_mean
            var = self.running_var

        # 归一化
        x_norm = (x - mean.reshape(1, C, 1, 1)) / np.sqrt(var.reshape(1, C, 1, 1) + self.eps)

        # 缩放和平移
        out = self.gamma.reshape(1, C, 1, 1) * x_norm + self.beta.reshape(1, C, 1, 1)

        # 保存中间变量用于反向传播
        if self.training:
            self.cache = (x, x_norm, mean, var)

        return out

    def backward(self, grad):
        """
        反向传播
        grad: 上游梯度
        """
        if not self.training:
            return grad

        x, x_norm, mean, var = self.cache
        N, C, H, W = x.shape

        # 计算gamma和beta的梯度
        dgamma = np.sum(grad * x_norm, axis=(0, 2, 3))
        dbeta = np.sum(grad, axis=(0, 2, 3))

        # 计算x_norm的梯度
        dx_norm = grad * self.gamma.reshape(1, C, 1, 1)

        # 计算x的梯度
        dx = dx_norm / np.sqrt(var.reshape(1, C, 1, 1) + self.eps)

        # 更新参数
        self.gamma -= 0.01 * dgamma  # 使用学习率0.01
        self.beta -= 0.01 * dbeta

        return dx

    def train(self):
        """设置为训练模式"""
        self.training = True

    def eval(self):
        """设置为推理模式"""
        self.training = False


class BatchNorm1D:
    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.training = True  # 添加训练模式标志

        # 可学习参数
        self.gamma = np.ones(num_features)  # 缩放参数
        self.beta = np.zeros(num_features)  # 平移参数

        # 运行时统计量
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        # 反向传播需要的中间变量
        self.cache = None
        self.x_norm = None

    def forward(self, x):
        """
        前向传播
        x: (N, C) 形状的输入
        """
        N, C = x.shape

        if self.training:
            # 训练模式：使用当前批次的统计量
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)

            # 更新运行时统计量
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            # 推理模式：使用运行时统计量
            mean = self.running_mean
            var = self.running_var

        # 归一化
        x_norm = (x - mean) / np.sqrt(var + self.eps)

        # 缩放和平移
        out = self.gamma * x_norm + self.beta

        # 保存中间变量用于反向传播
        if self.training:
            self.cache = (x, x_norm, mean, var)

        return out

    def backward(self, grad):
        """
        反向传播
        grad: 上游梯度
        """
        if not self.training:
            return grad

        x, x_norm, mean, var = self.cache
        N, C = x.shape

        # 计算gamma和beta的梯度
        dgamma = np.sum(grad * x_norm, axis=0)
        dbeta = np.sum(grad, axis=0)

        # 计算x_norm的梯度
        dx_norm = grad * self.gamma

        # 计算x的梯度
        dx = dx_norm / np.sqrt(var + self.eps)

        # 更新参数
        self.gamma -= 0.01 * dgamma  # 使用学习率0.01
        self.beta -= 0.01 * dbeta

        return dx

    def train(self):
        """设置为训练模式"""
        self.training = True

    def eval(self):
        """设置为推理模式"""
        self.training = False