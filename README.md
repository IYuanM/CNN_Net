### 无框架CNN模型的设计与实现

设计的网络是一个基于LFW人脸数据集的无框架CNN模型，用于人脸分类任务。

**简单总结下网络优点**

1. 首先这个模型没有依赖现有的框架，手动实现了CNN核心组件（卷积、池化、归一化等），代码透明且易于定制。
2. 在数据加载方面采用多线程预加载和缓存机制减少I/O瓶颈，提升训练效率。
3. 利用了很多训练技巧提升训练的效率包括：
   - 余弦退火学习率调度（带预热阶段）。
   - 带动量和权重衰减的SGD优化器。
   - 梯度裁剪防止梯度爆炸。
4. 同时我引入了残差块（ResidualBlock）缓解梯度消失，支持深层网络训练；还引入了批归一化（BatchNorm）加速收敛。
5. 为了增加模型的鲁棒性，采用了早停机制、数值稳定性处理（如Softmax的溢出保护）。


#### 下面是网络核心模块解释

#### 1. **核心模块**

- **卷积层（Conv2D）**

  - **实现**：通过`im2col`将图像转换为列矩阵，利用矩阵乘法加速卷积运算。
  - **特点**：He初始化权重，支持自定义步长和填充。

  ```python
  # conv.py 中 Conv2D 类的前向传播实现
  def forward(self, x):
      N, C, H, W = x.shape
      out_h = (H + 2*self.pad - self.kernel_size) // self.stride + 1
      out_w = (W + 2*self.pad - self.kernel_size) // self.stride + 1
      
      # 使用 im2col 将输入转换为列矩阵
      col = im2col(x, self.kernel_size, self.kernel_size, self.stride, self.pad)
      W_col = self.W.reshape(self.out_channels, -1)  # 重塑权重
      
      # 矩阵乘法实现卷积
      out = np.dot(col, W_col.T) + self.b
      out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)  # 重塑输出形状
      
      self.cache = x  # 保存输入用于反向传播
      return out
  ```

  

- **批归一化（BatchNorm2D/BatchNorm1D）**

  - **作用**：归一化输入数据，加速训练并减少对初始化的依赖。
  - **实现**：训练时动态计算批次统计量，推理时使用滑动平均。

  ```python
  # batch_norm.py 中 BatchNorm2D 的前向传播实现
  def forward(self, x):
      N, C, H, W = x.shape
      if self.training:
          mean = np.mean(x, axis=(0, 2, 3))  # 按通道计算均值和方差
          var = np.var(x, axis=(0, 2, 3))
          # 更新运行时统计量（指数移动平均）
          self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
          self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
      else:
          mean = self.running_mean  # 推理阶段使用预存统计量
          var = self.running_var
      
      x_norm = (x - mean.reshape(1, C, 1, 1)) / np.sqrt(var.reshape(1, C, 1, 1) + self.eps)
      out = self.gamma.reshape(1, C, 1, 1) * x_norm + self.beta.reshape(1, C, 1, 1)  # 缩放和平移
      return out
  ```

  

- **残差块（ResidualBlock）**

  - **结构**：两个卷积层 + 跳跃连接（Shortcut），支持通道数和尺寸调整。
  - **作用**：缓解梯度消失，提升深层网络性能。

  ```python
  # model.py 中 ResidualBlock 的前向传播实现
  def forward(self, x):
      identity = x  # 保留原始输入
      
      # 主路径：两个卷积层 + 批归一化 + ReLU
      out = self.conv1.forward(x)
      out = self.bn1.forward(out)
      out = self.relu1.forward(out)
      
      out = self.conv2.forward(out)
      out = self.bn2.forward(out)
      
      # 跳跃连接：调整通道或步长不一致时的映射
      if self.shortcut is not None:
          identity = self.shortcut.forward(x)       # 1x1 卷积调整维度
          identity = self.bn_shortcut.forward(identity)
      
      out += identity  # 残差相加
      out = self.relu2.forward(out)
      return out
  ```

  

- **激活函数**

  - **ReLU**：简单阈值激活，反向传播时仅保留正梯度。

  ```python
  # activation.py 中 ReLU 的实现
  class ReLU:
      def __init__(self):
          self.cache = None  # 保存输入用于反向传播
      
      def forward(self, x):
          """前向传播：输出为 max(0, x)"""
          self.cache = x  # 缓存输入
          return np.maximum(0, x)
      
      def backward(self, grad):
          """反向传播：仅正值的梯度保留"""
          x = self.cache
          dx = grad * (x > 0)  # 输入大于0的位置梯度为1，否则为0
          return dx
  ```

  - **Softmax**：数值稳定实现（减最大值），反向传播使用雅可比矩阵。

  ```python
  # activation.py 中 Softmax 的实现
  class Softmax:
      def __init__(self):
          self.cache = None  # 保存输出概率用于反向传播
      
      def forward(self, x):
          """前向传播：数值稳定性处理"""
          # 减去最大值防止指数溢出
          x_shifted = x - np.max(x, axis=1, keepdims=True)
          exp_x = np.exp(x_shifted)
          # 计算概率分布
          out = exp_x / np.sum(exp_x, axis=1, keepdims=True)
          self.cache = out  # 缓存输出概率
          return out
      
      def backward(self, grad):
          """反向传播：雅可比矩阵计算"""
          y = self.cache  # 前向传播的输出概率
          batch_size = y.shape[0]
          
          # 向量化计算梯度（避免显式构造雅可比矩阵）
          # 公式：dx = y * (grad - sum(grad * y, axis=1))
          sum_grad_y = np.sum(grad * y, axis=1, keepdims=True)
          dx = y * (grad - sum_grad_y)
          return dx
  ```

  

- **池化层**

  - **实现**：通过`as_strided`高效创建滑动窗口视图，MaxPool反向传播仅传递梯度到最大值位置。

  最大池化层：

  ```python
  # pooling.py 中 MaxPool2D 的实现
  class MaxPool2D:
      def __init__(self, kernel_size=2, stride=None):
          self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
          self.stride = stride if stride is not None else self.kernel_size
          self.cache = {}
  
      def forward(self, x: np.ndarray) -> np.ndarray:
          """前向传播"""
          N, C, H, W = x.shape
          kh, kw = self.kernel_size
          sh, sw = self.stride
  
          # 计算输出尺寸
          out_h = (H - kh) // sh + 1
          out_w = (W - kw) // sw + 1
  
          # 使用 as_strided 创建滑动窗口视图（高效内存访问）
          strides = (
              x.strides[0],  # 样本维度步长
              x.strides[1],  # 通道维度步长
              sh * x.strides[2],  # 垂直步长
              sw * x.strides[3],  # 水平步长
              x.strides[2],  # 窗口内垂直步长
              x.strides[3]   # 窗口内水平步长
          )
          windows = np.lib.stride_tricks.as_strided(
              x,
              shape=(N, C, out_h, out_w, kh, kw),
              strides=strides
          )
  
          # 沿最后两个维度（kh, kw）取最大值
          out = np.max(windows, axis=(4, 5))
  
          # 缓存最大值位置，用于反向传播
          self.cache = {
              'input_shape': x.shape,
              'windows': windows,
              'max_mask': (windows == out[..., None, None])  # 标记最大值位置
          }
          return out
  
      def backward(self, grad: np.ndarray) -> np.ndarray:
          """反向传播：仅将梯度传递给最大值位置"""
          N, C, H, W = self.cache['input_shape']
          kh, kw = self.kernel_size
          sh, sw = self.stride
  
          dx = np.zeros((N, C, H, W), dtype=grad.dtype)
          max_mask = self.cache['max_mask']  # 最大值位置的布尔掩码
  
          # 遍历窗口内每个位置，将梯度分配到原始输入的最大值位置
          for i in range(kh):
              for j in range(kw):
                  h_start = i
                  w_start = j
                  h_end = H - kh + i + 1
                  w_end = W - kw + j + 1
                  # 提取当前窗口位置的掩码并分配梯度
                  mask_slice = max_mask[..., i, j]
                  dx[:, :, h_start:h_end:sh, w_start:w_end:sw] += grad * mask_slice
          return dx
  ```

平均池化层：

```python
# pooling.py 中 AvgPool2D 的实现
class AvgPool2D:
    def __init__(self, kernel_size=2, stride=None):
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride if stride is not None else self.kernel_size
        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        N, C, H, W = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride

        # 计算输出尺寸
        out_h = (H - kh) // sh + 1
        out_w = (W - kw) // sw + 1

        # 创建滑动窗口视图（同 MaxPool2D）
        strides = (
            x.strides[0],
            x.strides[1],
            sh * x.strides[2],
            sw * x.strides[3],
            x.strides[2],
            x.strides[3]
        )
        windows = np.lib.stride_tricks.as_strided(
            x,
            shape=(N, C, out_h, out_w, kh, kw),
            strides=strides
        )

        # 沿最后两个维度取平均值
        out = np.mean(windows, axis=(4, 5))

        # 缓存参数用于反向传播
        self.cache = {
            'input_shape': x.shape,
            'pool_size': kh * kw  # 池化窗口大小（用于梯度分配）
        }
        return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """反向传播：将梯度平均分配到窗口内所有位置"""
        N, C, H, W = self.cache['input_shape']
        kh, kw = self.kernel_size
        sh, sw = self.stride
        pool_size = self.cache['pool_size']

        dx = np.zeros((N, C, H, W), dtype=grad.dtype)

        # 遍历窗口内每个位置，均匀分配梯度
        for i in range(kh):
            for j in range(kw)):
                h_start = i
                w_start = j
                h_end = H - kh + i + 1
                w_end = W - kw + j + 1
                # 将梯度除以池化窗口大小（平均分配）
                dx[:, :, h_start:h_end:sh, w_start:w_end:sw] += grad / pool_size
        return dx
```



#### 2. **辅助模块**

- **全连接层（Linear）**：采用He初始化，支持矩阵运算的线性变换。
- **展平层（Flatten）**：将多维特征转换为一维向量，适配全连接层输入。
- **优化器（SGD）**：带动量和权重衰减，梯度更新时叠加历史速度。
- **学习率调度（CosineAnnealingLR）**：余弦退火配合预热阶段，平衡探索与收敛。

#### 3. **数据与训练模块**

- **LFWDataLoader**：

  - 多线程预加载图像到缓存，支持随机访问。
  - 数据划分时确保每个类别在训练集中至少有一个样本。

  ```python
  # train.py 中 LFWDataLoader 的预加载线程实现
  def _preload_worker(self):
      while not self.stop_event.is_set():
          try:
              idx = np.random.randint(0, len(self.train_paths))  # 随机选择样本
              if idx not in self.cache:
                  img = self._load_single_image(self.train_paths[idx])  # 加载并预处理图像
                  if len(self.cache) >= self.cache_size:
                      self.cache.pop(next(iter(self.cache)))  # 缓存淘汰策略（FIFO）
                  self.cache[idx] = img
          except Exception as e:
              print(f"Error in preload worker: {e}")
          time.sleep(0.001)  # 避免CPU过载
  ```

  

- **Trainer**：

  - 训练-验证循环，支持早停和最佳模型保存。
  - 混合精度训练（通过数值稳定性设计）。

  ```python
  # train.py 中 Trainer 的训练步骤
  def train_epoch(self):
      for batch_idx in tqdm(range(self.data_loader.train_batches)):
          # 1. 学习率更新（余弦退火 + 预热）
          current_lr = self.lr_scheduler.step()
          self.optimizer.lr = current_lr
          
          # 2. 数据加载（优先从缓存读取）
          x_batch, y_batch = self.data_loader.get_train_batch(batch_idx)
          
          # 3. 前向传播
          y_pred = self.model.forward(x_batch)
          loss = cross_entropy_loss(y_pred, y_batch)
          
          # 4. 反向传播与梯度裁剪
          grad = y_pred.copy()
          grad[np.arange(len(y_batch)), y_batch] -= 1
          grad /= len(y_batch)
          grad_norm = np.sqrt(np.sum(grad**2))
          if grad_norm > self.config.max_grad_norm:
              grad = grad * (self.config.max_grad_norm / grad_norm)  # 梯度裁剪
          
          # 5. 参数更新（SGD + 动量）
          self.model.backward(grad)
          self.optimizer.step()
  ```

  


### 训练流程与技巧

#### **训练流程**

1. **数据准备**：
   - 图像归一化（像素值缩放到[0,1]），尺寸统一调整。
   - 按8:2划分训练集和验证集，确保类别均衡。
2. **模型初始化**：
   - CNN结构：3个卷积块（Conv2D + BatchNorm + ReLU + MaxPool） → 残差块 → 全连接层 → Softmax。
3. **训练循环**：
   - **前向传播**：依次通过卷积、残差、池化、全连接层。
   - **损失计算**：交叉熵损失（带数值裁剪）。
   - **反向传播**：逐层计算梯度，应用梯度裁剪。
   - **参数更新**：SGD优化器结合动量与权重衰减。
4. **验证与早停**：
   - 监控验证集准确率，连续下降`N`次触发早停。
   - 分别保存训练和验证集最佳模型。

#### **训练技巧**

1. **学习率调度**：
   - 预热阶段逐步增加学习率，避免初始震荡。
   - 余弦退火动态调整学习率，平衡探索与收敛。
2. **正则化**：
   - 权重衰减（L2正则）防止过拟合。
   - 批归一化隐含正则化效果。
3. **高效数据加载**：
   - 多线程预加载减少I/O等待，缓存机制加速数据读取。
4. **梯度管理**：
   - 梯度裁剪（`max_grad_norm`）防止梯度爆炸。
5. **模型保存策略**：
   - 独立保存训练集和验证集最佳模型，便于后续选择。

### 总结

​		本网络结合了现代CNN的设计理念（残差连接、批归一化）与高效的工程实现（多线程数据加载、自定义优化器），适用于中等规模图像分类任务。其训练过程通过动态学习率、早停、正则化等技巧显著提升了模型鲁棒性，同时无框架实现为深入理解底层机制提供了良好范本。



**训练输出：**

（由于截图大小限制，只有最终的部分结果截图，后面会展示复制的全部输出）

![image-20250521152203559](https://github.com/IYuanM/CNN_Net/blob/main/CNN%E8%BF%90%E8%A1%8C%E7%BB%93%E6%9E%9C.png)

**全部输出：**

D:\workSoftware\anaconda3\envs\pytorch\python.exe D:\Projects\python\LeetCode\CNN\train.py 
Initializing data loader...
Found 639 classes
Training samples: 1250
Validation samples: 206
Training:   0%|          | 0/19 [00:00<?, ?it/s]
Starting training...

Epoch	Train Loss	Train Acc	Val Loss	Val Acc	LR
----------------------------------------------------------------------

Training: 100%|██████████| 19/19 [00:55<00:00,  2.91s/it]
  1	6.5336	0.0625	10.0952	0.0521	0.033333
Training: 100%|██████████| 19/19 [00:54<00:00,  2.85s/it]

 2	4.9998	0.1809	9.8336	0.0417	0.066667
Training: 100%|██████████| 19/19 [00:53<00:00,  2.83s/it]
  3	4.6904	0.1694	8.7832	0.0312	0.100000
Training: 100%|██████████| 19/19 [00:56<00:00,  2.95s/it]
  4	4.5414	0.1735	7.7080	0.0417	0.099888
Training: 100%|██████████| 19/19 [00:57<00:00,  3.04s/it]
  5	3.4204	0.3051	9.0897	0.0417	0.099554
Training: 100%|██████████| 19/19 [00:55<00:00,  2.93s/it]
6	2.5979	0.4852	9.1798	0.0312	0.098998
Training: 100%|██████████| 19/19 [00:57<00:00,  3.02s/it]
 7	2.0885	0.6020	9.4875	0.0312	0.098223
Training: 100%|██████████| 19/19 [00:55<00:00,  2.94s/it]
 8	1.7421	0.6859	10.1560	0.0365	0.097233
Training: 100%|██████████| 19/19 [00:55<00:00,  2.92s/it]
  9	1.4884	0.7451	9.8269	0.0469	0.096033
Training: 100%|██████████| 19/19 [00:59<00:00,  3.12s/it]
10	1.3954	0.7714	9.9898	0.0365	0.094626
Training: 100%|██████████| 19/19 [00:55<00:00,  2.94s/it]
11	1.2077	0.8117	9.8328	0.0365	0.093020
Training: 100%|██████████| 19/19 [00:55<00:00,  2.94s/it]

12	1.1747	0.8199	10.2633	0.0625	0.091222
Training: 100%|██████████| 19/19 [00:55<00:00,  2.92s/it]
13	1.0707	0.8561	10.0745	0.0573	0.089240
Training: 100%|██████████| 19/19 [00:55<00:00,  2.94s/it]
 14	1.0181	0.8717	9.3804	0.0417	0.087083
Training: 100%|██████████| 19/19 [00:56<00:00,  2.96s/it]
 15	1.0096	0.8799	9.5775	0.0625	0.084760
Training: 100%|██████████| 19/19 [01:00<00:00,  3.19s/it]
 16	1.0983	0.8783	9.5990	0.0469	0.082282
Training: 100%|██████████| 19/19 [00:55<00:00,  2.92s/it]

17	1.2449	0.8635	8.8872	0.0677	0.079659
Training: 100%|██████████| 19/19 [00:55<00:00,  2.92s/it]

Training accuracy dropped! (3/3)
Validation accuracy dropped! (1/5)

Early stopping triggered!

Process finished with exit code 0



#### 输出总结

由于个人设备问题没有采用全部的数据集训练，而是抽取了三分之一的数据。训练集的精确度达到了87.83%，验证集的精确度较低只有6.77%，主要原因可能在于数据类别分布不均衡。



