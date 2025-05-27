import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from model import CNN
import threading
from queue import Queue
import time


class LFWDataLoader:
    def __init__(self, data_dir, img_size=(64, 64), batch_size=64, num_workers=4):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache = {}
        self.cache_size = 1000  # 缓存大小
        self.cache_queue = Queue(maxsize=self.cache_size)
        self.stop_event = threading.Event()

        # 获取所有类别和图像路径
        self.classes = []
        self.image_paths = []
        self.labels = []

        # 遍历数据集目录
        for person_name in os.listdir(data_dir):
            person_path = os.path.join(data_dir, person_name)
            if os.path.isdir(person_path):
                self.classes.append(person_name)
                class_idx = len(self.classes) - 1

                for img_file in os.listdir(person_path):
                    if img_file.endswith(('.jpg', '.png')):
                        img_path = os.path.join(person_path, img_file)
                        self.image_paths.append(img_path)
                        self.labels.append(class_idx)

        self.image_paths = np.array(self.image_paths)
        self.labels = np.array(self.labels)
        self.num_classes = len(self.classes)

        # 划分训练集和验证集
        indices = np.arange(len(self.image_paths))
        train_idx, val_idx = train_test_split(
            indices, test_size=0.2, random_state=42
        )

        # 确保每个类别至少有一张图片在训练集中
        unique_labels = np.unique(self.labels)
        for label in unique_labels:
            label_indices = np.where(self.labels == label)[0]
            if len(label_indices) == 1:
                # 如果该类别只有一张图片，将其放入训练集
                if label_indices[0] in val_idx:
                    val_idx = np.delete(val_idx, np.where(val_idx == label_indices[0])[0])
                    train_idx = np.append(train_idx, label_indices[0])

        self.train_paths = self.image_paths[train_idx]
        self.train_labels = self.labels[train_idx]
        self.val_paths = self.image_paths[val_idx]
        self.val_labels = self.labels[val_idx]

        # 计算每个epoch的批次数
        self.train_batches = len(self.train_paths) // batch_size
        self.val_batches = len(self.val_paths) // batch_size

        # 启动预加载线程
        self.start_preload_threads()

    def start_preload_threads(self):
        """启动数据预加载线程"""
        for _ in range(self.num_workers):
            t = threading.Thread(target=self._preload_worker)
            t.daemon = True
            t.start()

    def _preload_worker(self):
        """预加载工作线程"""
        while not self.stop_event.is_set():
            try:
                # 随机选择一个训练样本进行预加载
                idx = np.random.randint(0, len(self.train_paths))
                if idx not in self.cache:
                    img = self._load_single_image(self.train_paths[idx])
                    if len(self.cache) >= self.cache_size:
                        # 如果缓存已满，移除最旧的项
                        self.cache.pop(next(iter(self.cache)))
                    self.cache[idx] = img
            except Exception as e:
                print(f"Error in preload worker: {e}")
            time.sleep(0.001)  # 避免CPU过载

    def _load_single_image(self, path):
        """加载单张图片"""
        img = Image.open(path).convert('RGB')
        img = img.resize(self.img_size)
        img = np.array(img) / 255.0
        return img.transpose(2, 0, 1)

    def load_batch(self, paths, labels, indices):
        """加载一个批次的数据"""
        batch_images = []
        batch_labels = []

        for idx in indices:
            try:
                if idx in self.cache:
                    img = self.cache[idx]
                else:
                    img = self._load_single_image(paths[idx])
                    if len(self.cache) < self.cache_size:
                        self.cache[idx] = img
                batch_images.append(img)
                batch_labels.append(labels[idx])
            except Exception as e:
                print(f"Error loading {paths[idx]}: {e}")

        return np.array(batch_images), np.array(batch_labels)

    def get_train_batch(self, batch_idx):
        """获取训练集的一个批次"""
        start_idx = batch_idx * self.batch_size
        end_idx = start_idx + self.batch_size
        indices = np.arange(start_idx, end_idx)
        return self.load_batch(self.train_paths, self.train_labels, indices)

    def get_val_batch(self, batch_idx):
        """获取验证集的一个批次"""
        start_idx = batch_idx * self.batch_size
        end_idx = start_idx + self.batch_size
        indices = np.arange(start_idx, end_idx)
        return self.load_batch(self.val_paths, self.val_labels, indices)

    def __del__(self):
        """清理资源"""
        self.stop_event.set()


class CosineAnnealingLR:
    def __init__(self, initial_lr, T_max, eta_min=0, warmup_epochs=5):
        self.initial_lr = initial_lr
        self.T_max = T_max
        self.eta_min = eta_min
        self.current_step = 0
        self.warmup_epochs = warmup_epochs
        self.warmup_steps = warmup_epochs * (T_max // 50)  # 假设50个epoch

    def step(self):
        """更新学习率"""
        self.current_step += 1

        # 预热阶段
        if self.current_step < self.warmup_steps:
            lr = self.initial_lr * (self.current_step / self.warmup_steps)
        else:
            # 余弦退火阶段
            progress = (self.current_step - self.warmup_steps) / (self.T_max - self.warmup_steps)
            lr = self.eta_min + 0.5 * (self.initial_lr - self.eta_min) * \
                 (1 + np.cos(np.pi * progress))

        return lr


class SGD:
    def __init__(self, params, lr=0.1, momentum=0.9, weight_decay=1e-4):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = {}

        # 初始化速度
        for layer in params:
            if hasattr(layer, 'W'):
                self.velocities[layer] = {
                    'W': np.zeros_like(layer.W),
                    'b': np.zeros_like(layer.b)
                }

    def step(self):
        """更新参数"""
        for layer in self.params:
            if hasattr(layer, 'W'):
                # 应用权重衰减
                layer.dW += self.weight_decay * layer.W

                # 更新速度
                self.velocities[layer]['W'] = self.momentum * self.velocities[layer]['W'] - self.lr * layer.dW
                self.velocities[layer]['b'] = self.momentum * self.velocities[layer]['b'] - self.lr * layer.db

                # 更新参数
                layer.W += self.velocities[layer]['W']
                layer.b += self.velocities[layer]['b']


def cross_entropy_loss(y_pred, y_true):
    """
    计算交叉熵损失
    y_pred: 预测概率 (N, C)
    y_true: 真实标签 (N,)
    """
    # 添加数值稳定性处理
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # 防止log(0)
    n_samples = y_true.shape[0]

    # 计算每个样本的损失
    losses = -np.log(y_pred[np.arange(n_samples), y_true])

    # 返回平均损失
    return np.mean(losses)


def accuracy(y_pred, y_true):
    """计算准确率"""
    return np.mean(np.argmax(y_pred, axis=1) == y_true)


class TrainingConfig:
    def __init__(self):
        # 数据相关参数
        self.data_dir = 'lfw'
        self.img_size = (64, 64)
        self.batch_size = 64
        self.num_workers = 4

        # 训练相关参数
        self.epochs = 50
        self.initial_learning_rate = 0.1
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.max_grad_norm = 1.0

        # 学习率调度器参数
        self.warmup_epochs = 3
        self.min_lr = 1e-6

        # 早停参数
        self.max_train_drops = 3
        self.max_val_drops = 5


class Trainer:
    def __init__(self, config):
        self.config = config
        self.data_loader = None
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None

        # 训练状态
        self.best_train_acc = 0.0
        self.best_val_acc = 0.0
        self.train_acc_drops = 0
        self.val_acc_drops = 0
        self.prev_train_acc = 0.0
        self.prev_val_acc = 0.0

    def setup_data(self):
        """初始化数据加载器"""
        print("Initializing data loader...")
        self.data_loader = LFWDataLoader(
            self.config.data_dir,
            img_size=self.config.img_size,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers
        )
        print(f"Found {len(self.data_loader.classes)} classes")
        print(f"Training samples: {len(self.data_loader.train_paths)}")
        print(f"Validation samples: {len(self.data_loader.val_paths)}")

    def setup_model(self):
        """初始化模型、优化器和学习率调度器"""
        self.model = CNN(self.data_loader.num_classes)
        self.optimizer = SGD(
            self.model.layers,
            lr=self.config.initial_learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay
        )
        self.lr_scheduler = CosineAnnealingLR(
            self.config.initial_learning_rate,
            T_max=self.config.epochs * self.data_loader.train_batches,
            eta_min=self.config.min_lr,
            warmup_epochs=self.config.warmup_epochs
        )

    def train_epoch(self):
        """训练一个epoch"""
        train_loss = 0
        train_correct = 0
        train_total = 0

        # 设置为训练模式
        for layer in self.model.layers:
            if hasattr(layer, 'train'):
                layer.train()

        for batch_idx in tqdm(range(self.data_loader.train_batches), desc="Training"):
            # 获取当前学习率
            current_lr = self.lr_scheduler.step()
            self.optimizer.lr = current_lr

            # 加载批次数据
            x_batch, y_batch = self.data_loader.get_train_batch(batch_idx)

            # 前向传播
            y_pred = self.model.forward(x_batch)

            # 计算损失和准确率
            loss = cross_entropy_loss(y_pred, y_batch)
            acc = accuracy(y_pred, y_batch)

            # 反向传播
            grad = y_pred.copy()
            grad[np.arange(len(y_batch)), y_batch] -= 1
            grad /= len(y_batch)

            # 梯度裁剪
            grad_norm = np.sqrt(np.sum(grad ** 2))
            if grad_norm > self.config.max_grad_norm:
                grad = grad * (self.config.max_grad_norm / grad_norm)

            self.model.backward(grad)
            self.optimizer.step()

            # 更新统计信息
            train_loss += loss
            train_correct += acc * len(y_batch)
            train_total += len(y_batch)

        return train_loss / self.data_loader.train_batches, train_correct / train_total

    def validate(self):
        """验证模型"""
        val_loss = 0
        val_correct = 0
        val_total = 0

        # 设置为评估模式
        for layer in self.model.layers:
            if hasattr(layer, 'eval'):
                layer.eval()

        for batch_idx in range(self.data_loader.val_batches):
            x_batch, y_batch = self.data_loader.get_val_batch(batch_idx)
            y_pred = self.model.forward(x_batch)
            val_loss += cross_entropy_loss(y_pred, y_batch)
            val_acc = accuracy(y_pred, y_batch)
            val_correct += val_acc * len(y_batch)
            val_total += len(y_batch)

        return val_loss / self.data_loader.val_batches, val_correct / val_total

    def check_early_stopping(self, train_acc, val_acc):
        """检查是否需要早停"""
        if train_acc < self.prev_train_acc:
            self.train_acc_drops += 1
            print(f"\nTraining accuracy dropped! ({self.train_acc_drops}/{self.config.max_train_drops})")
        else:
            self.train_acc_drops = 0

        if val_acc < self.prev_val_acc:
            self.val_acc_drops += 1
            print(f"Validation accuracy dropped! ({self.val_acc_drops}/{self.config.max_val_drops})")
        else:
            self.val_acc_drops = 0

        self.prev_train_acc = train_acc
        self.prev_val_acc = val_acc

        return (self.train_acc_drops >= self.config.max_train_drops or
                self.val_acc_drops >= self.config.max_val_drops)

    def save_best_models(self, epoch, train_acc, val_acc):
        """保存最佳模型"""
        if train_acc > self.best_train_acc:
            self.best_train_acc = train_acc
            print(f"\nSaving best training model at epoch {epoch + 1}")
            self.model.save(f'best_train_model.pth')

        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            print(f"\nSaving best validation model at epoch {epoch + 1}")
            self.model.save(f'best_val_model.pth')

    def train(self):
        """训练模型的主循环"""
        print("\nStarting training...")
        print("Epoch\tTrain Loss\tTrain Acc\tVal Loss\tVal Acc\tLR")
        print("-" * 70)

        for epoch in range(self.config.epochs):
            # 训练一个epoch
            train_loss, train_acc = self.train_epoch()

            # 验证
            val_loss, val_acc = self.validate()

            # 检查早停
            if epoch > 0 and self.check_early_stopping(train_acc, val_acc):
                print("\nEarly stopping triggered!")
                break

            # 保存最佳模型
            self.save_best_models(epoch, train_acc, val_acc)

            # 打印训练信息
            print(
                f"{epoch + 1:3d}\t{train_loss:.4f}\t{train_acc:.4f}\t{val_loss:.4f}\t{val_acc:.4f}\t{self.optimizer.lr:.6f}")
            print(f"Best Train Acc: {self.best_train_acc:.4f}, Best Val Acc: {self.best_val_acc:.4f}")


def main():
    # 创建配置
    config = TrainingConfig()

    # 创建训练器
    trainer = Trainer(config)

    # 设置数据和模型
    trainer.setup_data()
    trainer.setup_model()

    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main() 