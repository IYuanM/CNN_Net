import numpy as np

class ReLU:
    def __init__(self):
        self.cache = None
    
    def forward(self, x):
        """
        前向传播
        x: 任意形状的输入
        """
        # 保存输入用于反向传播
        self.cache = x
        
        # ReLU激活
        return np.maximum(0, x)
    
    def backward(self, grad):
        """
        反向传播
        grad: 上游梯度
        """
        x = self.cache
        
        # 计算梯度
        dx = grad * (x > 0)
        
        return dx

class Softmax:
    def __init__(self):
        self.cache = None
    
    def forward(self, x):
        """
        前向传播
        x: (N, C) 形状的输入
        """
        # 数值稳定性处理
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        
        # 计算softmax
        out = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        
        # 保存输出用于反向传播
        self.cache = out
        
        return out
    
    def backward(self, grad):
        """
        反向传播
        grad: 上游梯度
        """
        y = self.cache
        
        # 计算softmax的梯度（向量化实现）
        # 创建雅可比矩阵的批量版本
        batch_size = y.shape[0]
        y_reshaped = y.reshape(batch_size, -1, 1)
        jacobian = np.eye(y.shape[1])[None, :, :] - np.matmul(y_reshaped, y_reshaped.transpose(0, 2, 1))
        
        # 计算梯度
        dx = np.matmul(jacobian, grad[:, :, None]).squeeze(-1)
        
        return dx

class Dropout:
    def __init__(self, p=0.5):
        """
        初始化Dropout层
        p: dropout概率
        """
        self.p = p
        self.mask = None
        self.training = True
    
    def forward(self, x):
        """
        前向传播
        x: 任意形状的输入
        """
        if self.training:
            # 生成dropout掩码
            self.mask = np.random.binomial(1, 1-self.p, size=x.shape) / (1-self.p)
            return x * self.mask
        return x
    
    def backward(self, grad):
        """
        反向传播
        grad: 上游梯度
        """
        if self.training:
            return grad * self.mask
        return grad
    
    def train(self):
        """设置为训练模式"""
        self.training = True
    
    def eval(self):
        """设置为评估模式"""
        self.training = False 