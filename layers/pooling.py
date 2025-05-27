import numpy as np
from numpy.lib.stride_tricks import as_strided


class Flatten:
    """展平层，用于将多维输入转换为一维特征向量"""

    def __init__(self):
        self.original_shape = None
        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播

        参数：
        x: 输入数据，形状 (N, C, H, W)

        返回：
        out: 展平后的数据，形状 (N, C*H*W)
        """
        self.cache['original_shape'] = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        反向传播

        参数：
        grad: 梯度，形状 (N, C*H*W)

        返回：
        dx: 梯度，形状 (N, C, H, W)
        """
        return grad.reshape(self.cache['original_shape'])


class MaxPool2D:
    """优化的最大池化层实现"""

    def __init__(self, kernel_size=2, stride=None):
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride if stride is not None else self.kernel_size
        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播

        参数：
        x: 输入数据，形状 (N, C, H, W)

        返回：
        out: 池化结果，形状 (N, C, out_h, out_w)
        """
        N, C, H, W = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride,self.stride

        # 计算输出尺寸
        out_h = (H - kh) // sh + 1
        out_w = (W - kw) // sw + 1

        # 使用as_strided创建高效窗口视图
        strides = (
            x.strides[0],  # 样本维度 (N)
            x.strides[1],  # 通道维度 (C)
            sh * x.strides[2],  # 垂直步长 (H)
            sw * x.strides[3],  # 水平步长 (W)
            x.strides[2],  # 窗口内垂直步长
            x.strides[3]  # 窗口内水平步长
        )

        # 生成窗口视图 (N, C, out_h, out_w, kh, kw)
        windows = as_strided(
            x,
            shape=(N, C, out_h, out_w, kh, kw),
            strides=strides
        )

        # 沿最后两个维度取最大值
        out = np.max(windows, axis=(4, 5))

        # 缓存反向传播所需数据
        self.cache = {
            'input_shape': x.shape,
            'windows': windows,
            'max_mask': (windows == out[..., None, None]),  # 保持维度
            'output_shape': out.shape
        }

        return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        反向传播

        参数：
        grad: 梯度，形状 (N, C, out_h, out_w)

        返回：
        dx: 梯度，形状 (N, C, H, W)
        """
        # 确保梯度形状正确
        if grad.shape != self.cache['output_shape']:
            grad = grad.reshape(self.cache['output_shape'])

        N, C, H, W = self.cache['input_shape']
        kh, kw = self.kernel_size
        sh, sw = self.stride,self.stride

        # 初始化梯度张量
        dx = np.zeros((N, C, H, W), dtype=grad.dtype)

        # 将梯度分配到最大值位置
        for i in range(kh):
            for j in range(kw):
                # 计算每个位置的覆盖区域
                h_start = i
                w_start = j
                h_end = H - kh + i + 1
                w_end = W - kw + j + 1

                # 提取对应位置的mask
                mask_slice = self.cache['max_mask'][..., i, j]

                # 累积梯度
                dx[:, :, h_start:h_end:sh, w_start:w_end:sw] += grad * mask_slice

        return dx


class AvgPool2D:
    """优化的平均池化层实现"""

    def __init__(self, kernel_size=2, stride=None):
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride if stride is not None else self.kernel_size
        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播

        参数：
        x: 输入数据，形状 (N, C, H, W)

        返回：
        out: 池化结果，形状 (N, C, out_h, out_w)
        """
        N, C, H, W = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride

        # 计算输出尺寸
        out_h = (H - kh) // sh + 1
        out_w = (W - kw) // sw + 1

        # 使用as_strided创建高效窗口视图
        strides = (
            x.strides[0],  # 样本维度 (N)
            x.strides[1],  # 通道维度 (C)
            sh * x.strides[2],  # 垂直步长 (H)
            sw * x.strides[3],  # 水平步长 (W)
            x.strides[2],  # 窗口内垂直步长
            x.strides[3]  # 窗口内水平步长
        )

        # 生成窗口视图 (N, C, out_h, out_w, kh, kw)
        windows = as_strided(
            x,
            shape=(N, C, out_h, out_w, kh, kw),
            strides=strides
        )

        # 沿最后两个维度取平均值
        out = np.mean(windows, axis=(4, 5))

        # 缓存反向传播所需数据
        self.cache = {
            'input_shape': x.shape,
            'windows': windows,
            'pool_size': kh * kw,
            'output_shape': out.shape
        }

        return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        反向传播

        参数：
        grad: 梯度，形状 (N, C, out_h, out_w)

        返回：
        dx: 梯度，形状 (N, C, H, W)
        """
        # 确保梯度形状正确
        if grad.shape != self.cache['output_shape']:
            grad = grad.reshape(self.cache['output_shape'])

        N, C, H, W = self.cache['input_shape']
        kh, kw = self.kernel_size
        sh, sw = self.stride
        pool_size = self.cache['pool_size']

        # 初始化梯度张量
        dx = np.zeros((N, C, H, W), dtype=grad.dtype)

        # 将梯度平均分配到每个位置
        for i in range(kh):
            for j in range(kw):
                # 计算每个位置的覆盖区域
                h_start = i
                w_start = j
                h_end = H - kh + i + 1
                w_end = W - kw + j + 1

                # 累积梯度
                dx[:, :, h_start:h_end:sh, w_start:w_end:sw] += grad / pool_size

        return dx