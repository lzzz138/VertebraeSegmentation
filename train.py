import random
import time
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import SpineDataset
from unet import UNet
from utils.losses import CombinedLoss
from utils.metrics import calculate_metrics
from PIL import Image
from tqdm import tqdm
import os
import numpy as np

def set_seed(seed):
    random.seed(seed)  # 设置Python的随机种子
    np.random.seed(seed)  # 设置NumPy的随机种子
    torch.manual_seed(seed)  # 设置PyTorch的CPU随机种子
    torch.cuda.manual_seed(seed)  # 设置当前GPU的随机种子（如果使用GPU）
    torch.cuda.manual_seed_all(seed)  # 设置所有GPU的随机种子（如果使用多个GPU）
    torch.backends.cudnn.deterministic = True  # 确保每次卷积操作结果一致
    torch.backends.cudnn.benchmark = False  # 禁用CUDNN的自动优化

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, save_dir='./models'):
    best_dice = 0.0

    # 创建模型保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0
        running_dice_1 = 0.0
        running_dice_2 = 0.0

        # 训练阶段
        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as t:
            for inputs, labels in t:
                inputs, labels = inputs.to(device), labels.to(device)
                # 清零梯度
                optimizer.zero_grad()
                # 前向传播
                outputs = model(inputs)
                # 计算损失
                loss = criterion(outputs, labels)
                # 反向传播和优化
                loss.backward()
                optimizer.step()
                # 更新进度条
                # 计算Dice分数和IoU
                pred = torch.argmax(outputs, dim=1)
                running_dices,_ = calculate_metrics(pred, labels)
                running_dice_1 += running_dices[0]
                running_dice_2 += running_dices[1]
                running_loss += loss.item()
                t.set_postfix(loss=running_loss / (t.n + 1), dice_1=running_dice_1 / (t.n + 1), dice_2=running_dice_2 / (t.n + 1))

        # 更新学习率
        scheduler.step()

        # 记录损失和准确率
        epoch_loss = running_loss / len(train_loader)
        epoch_dice_1 = running_dice_1 / len(train_loader)
        epoch_dice_2 = running_dice_2 / len(train_loader)

        # 验证阶段
        model.eval()  # 设置模型为评估模式
        val_loss = 0.0
        val_dice_1 = 0.0
        val_dice_2 = 0.0
        with torch.no_grad():  # 禁用梯度计算
            for inputs, labels in tqdm(val_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                pred = torch.argmax(outputs, dim=1)
                val_dices,_ = calculate_metrics(pred, labels)
                val_dice_1 += val_dices[0]
                val_dice_2 += val_dices[1]
                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)
        val_dice_1 = val_dice_1 / len(val_loader)
        val_dice_2 = val_dice_2 / len(val_loader)

        tqdm.write(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Dice1: {epoch_dice_1:.4f}, Train Dice2: {epoch_dice_2:.4f}')
        tqdm.write(f'Validation Loss: {val_loss:.4f}, Validation Dice1: {val_dice_1:.4f}, Validation Dice2: {val_dice_2:.4f}')

        # 保存最优模型
        if val_dice_1+val_dice_2 > best_dice:
            best_dice = val_dice_1+val_dice_2
            torch.save(model.state_dict(), os.path.join(save_dir, 'best.pth'))
            tqdm.write(f"Best model saved with Dice {best_dice:.4f}")
        time.sleep(0.5)
