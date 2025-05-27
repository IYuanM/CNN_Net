import numpy as np
from layers import Conv2D, MaxPool2D, Linear, ReLU, Softmax, Flatten, BatchNorm2D, BatchNorm1D


class ResidualBlock:
    def __init__(self, in_channels, out_channels, stride=1):
        self.conv1 = Conv2D(in_channels, out_channels, kernel_size=3, stride=stride, pad=1)
        self.bn1 = BatchNorm2D(out_channels)
        self.relu1 = ReLU()
        self.conv2 = Conv2D(out_channels, out_channels, kernel_size=3, stride=1, pad=1)
        self.bn2 = BatchNorm2D(out_channels)

        # 如果输入输出通道数不同或步长不为1，需要调整shortcut
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Conv2D(in_channels, out_channels, kernel_size=1, stride=stride, pad=0)
            self.bn_shortcut = BatchNorm2D(out_channels)

        self.relu2 = ReLU()

    def forward(self, x):
        identity = x

        out = self.conv1.forward(x)
        out = self.bn1.forward(out)
        out = self.relu1.forward(out)

        out = self.conv2.forward(out)
        out = self.bn2.forward(out)

        if self.shortcut is not None:
            identity = self.shortcut.forward(x)
            identity = self.bn_shortcut.forward(identity)

        out += identity
        out = self.relu2.forward(out)

        return out

    def backward(self, grad):
        grad = self.relu2.backward(grad)

        if self.shortcut is not None:
            grad_shortcut = self.bn_shortcut.backward(grad)
            grad_shortcut = self.shortcut.backward(grad_shortcut)
        else:
            grad_shortcut = grad

        grad = self.bn2.backward(grad)
        grad = self.conv2.backward(grad)
        grad = self.relu1.backward(grad)
        grad = self.bn1.backward(grad)
        grad = self.conv1.backward(grad)

        grad += grad_shortcut
        return grad


class CNN:
    def __init__(self, num_classes):
        self.layers = [
            # 第一个卷积块
            Conv2D(3, 32, kernel_size=3, stride=1, pad=1),
            BatchNorm2D(32),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),

            # 第二个卷积块
            Conv2D(32, 64, kernel_size=3, stride=1, pad=1),
            BatchNorm2D(64),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),

            # 第三个卷积块
            Conv2D(64, 128, kernel_size=3, stride=1, pad=1),
            BatchNorm2D(128),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),

            # 一个残差块
            ResidualBlock(128, 256, stride=1),

            Flatten(),

            # 全连接层
            Linear(256 * 8 * 8, 512),
            BatchNorm1D(512),
            ReLU(),

            Linear(512, num_classes),
            Softmax()
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def save(self, path):
        params = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'W'):
                params[f"layer_{i}"] = {
                    'W': layer.W,
                    'b': layer.b
                }
            elif hasattr(layer, 'gamma'):
                params[f"layer_{i}"] = {
                    'gamma': layer.gamma,
                    'beta': layer.beta,
                    'running_mean': layer.running_mean,
                    'running_var': layer.running_var
                }
        np.save(path, params)

    def load(self, path):
        params = np.load(path, allow_pickle=True).item()
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'W'):
                layer.W = params[f"layer_{i}"]['W']
                layer.b = params[f"layer_{i}"]['b']
            elif hasattr(layer, 'gamma'):
                layer.gamma = params[f"layer_{i}"]['gamma']
                layer.beta = params[f"layer_{i}"]['beta']
                layer.running_mean = params[f"layer_{i}"]['running_mean']
                layer.running_var = params[f"layer_{i}"]['running_var']