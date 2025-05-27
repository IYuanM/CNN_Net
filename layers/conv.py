import numpy as np


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """将输入数据转换为列矩阵"""
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """将列矩阵转换回输入数据形状"""
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

        # 使用He初始化
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.b = np.zeros(out_channels)

        self.dW = None
        self.db = None
        self.cache = None

    def forward(self, x):
        """
        前向传播
        x: (N, C, H, W) 形状的输入
        """
        N, C, H, W = x.shape
        out_h = (H + 2 * self.pad - self.kernel_size) // self.stride + 1
        out_w = (W + 2 * self.pad - self.kernel_size) // self.stride + 1

        # 保存输入用于反向传播
        self.cache = x

        # 使用im2col转换输入
        col = im2col(x, self.kernel_size, self.kernel_size, self.stride, self.pad)

        # 重塑权重
        W_col = self.W.reshape(self.out_channels, -1)

        # 计算卷积
        out = np.dot(col, W_col.T) + self.b

        # 重塑输出
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        return out

    def backward(self, grad):
        """
        反向传播
        grad: 上游梯度
        """
        x = self.cache
        N, C, H, W = x.shape
        out_h = (H + 2 * self.pad - self.kernel_size) // self.stride + 1
        out_w = (W + 2 * self.pad - self.kernel_size) // self.stride + 1

        # 重塑梯度
        grad = grad.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)

        # 使用im2col转换输入
        col = im2col(x, self.kernel_size, self.kernel_size, self.stride, self.pad)

        # 计算权重和偏置的梯度
        W_col = self.W.reshape(self.out_channels, -1)
        self.dW = np.dot(grad.T, col).reshape(self.W.shape)
        self.db = np.sum(grad, axis=0)

        # 计算输入的梯度
        dx_col = np.dot(grad, W_col)
        dx = col2im(dx_col, x.shape, self.kernel_size, self.kernel_size, self.stride, self.pad)

        return dx