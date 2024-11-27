import numpy as np
import torch
import torch.optim as optim
from numpy.core._simd import targets
from modules.yolov5.utils.loss import ComputeLoss  # 引入 ComputeLoss 类
import yaml  # 用于加载超参数配置
from modules.yolov5 import models  # 导入YOLOv5模型
from modules import DIPModule, CNNPP, ImageLoader
from modules.loss_functions import combined_loss
import os
import glob
import matplotlib.pyplot as plt
from modules.yolov5.models.yolo import Model
from tqdm import tqdm

from modules.yolov5.hubconf import yolov5s

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模块
dip_module = DIPModule().to(device)
cnn_pp = CNNPP().to(device)

# yolov5_model = torch.hub.load('ultralytics/yolov5','yolov5s')  # 加载YOLOv5
# yolov5_model=yolov5_model.to(device)
yolov5_model = Model('modules/yolov5/models/yolov5s.yaml').to(device)


with open('modules/yolov5/models/yolov5s.yaml') as f:
    cfg = yaml.load(f, Loader=yaml.SafeLoader)
    yolov5_model.nc = cfg['nc']  # 从配置文件中获取类别数量
    print(f"Number of classes in the model: {yolov5_model.nc}")
    label_files = glob.glob('data/labels/*.txt')
    for file in label_files:
        with open(file, 'r') as f:
            for line in f:
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                if class_id < 0 or class_id >= yolov5_model.nc:
                    print(f"Invalid class_id {class_id} in {file}")
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                    print(f"Invalid bbox values in {file}: {line}")

with open('modules/yolov5/data/hyps/hyp.scratch-med.yaml') as f:
    yolov5_model.hyp = yaml.load(f, Loader=yaml.SafeLoader)  # 加载超参数字典
# 加载数据集
image_loader = ImageLoader('data/train/', image_size=(640, 640))
train_images, train_names = image_loader.load_images()

# 优化器
optimizer = optim.Adam(list(dip_module.parameters()) + list(cnn_pp.parameters()) + list(yolov5_model.parameters()), lr=1e-4)
compute_loss = ComputeLoss(yolov5_model)


def get_targets(img_name, model, device):
    """
    处理并加载目标标签，确保它们与模型预测输出的维度匹配。
    :param img_name: 图像名称
    :param model: YOLOv5 模型实例
    :param device: 设备（CPU/GPU）
    :return: 处理后的 targets 张量
    """
    # 加载标签文件
    label_path = os.path.join('data/labels', img_name.replace('.jpg', '.txt'))
    targets = np.loadtxt(label_path).reshape(-1, 6)  # [num_boxes, 6]

    # 转换为张量并移动到设备
    targets = torch.tensor(targets, dtype=torch.float32).to(device)

    # 获取锚框数量和类别数
    num_anchors = model.model[-1].anchors.shape[0]  # 获取模型中的锚框数量
    num_classes = model.nc  # 获取模型的类别数量
    batch_size = 1  # 假设每次只处理一张图像

    # 创建新的 targets 张量，形状为 [batch_size, num_anchors, num_boxes, num_classes + 5]
    num_boxes = targets.shape[0]
    targets_out = torch.zeros((batch_size, num_anchors, num_boxes, num_classes + 5), device=device)

    # 填充 targets_out
    for i in range(num_anchors):
        # 复制原始 targets 数据到 targets_out 的前 6 个维度
        targets_out[0, i, :, :6] = targets  # 前 6 个通道是 class_id, x_center, y_center, width, height, confidence
        # 添加锚框索引到 targets_out 的第 6 个通道
        targets_out[0, i, :, 6] = i  # 将锚框索引填入

    return targets_out  # 返回调整后的 targets 张量


# def get_targets(img_name):
#     label_path = os.path.join('data/labels', img_name.replace('.jpg', '.txt'))
#     targets = []
#     with open(label_path, 'r') as f:
#         for line in f:
#             class_id, x_center, y_center, width, height = map(float, line.strip().split())
#             # 检查 class_id 是否在有效范围内
#             if class_id < 0 or class_id >= yolov5_model.nc:
#                 print(f"Invalid class_id {class_id} in {label_path}, skipping this line.")
#                 continue
#             # 检查目标框坐标是否有效
#             if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
#                 print(f"Invalid bbox values in {label_path}: {line}, skipping this line.")
#                 continue
#             targets.append([class_id, x_center, y_center, width, height, 1.0])
#
#     targets = torch.tensor(targets, dtype=torch.float32).to(device)
#
#     # 确保 targets 是形状 [1, num_boxes, 6]，在 batch_size = 1 时
#     if targets.shape[0] == 1:  # For a single image
#         targets = targets.unsqueeze(0)  # Add batch dimension
#         print(f"Targets shape after unsqueeze: {targets.shape}")
#
#     return targets



# def get_targets(img_name):
#     # 读取标签文件
#     label_path = os.path.join('data/labels/', img_name.replace('.jpg', '.txt'))
#     targets = []
#     with open(label_path, 'r') as f:
#         for line in f:
#             class_id, x_center, y_center, width, height = map(float, line.strip().split())
#             if class_id < 0 or x_center < 0 or y_center < 0 or width <= 0 or height <= 0:
#                 print(f"Invalid label in {label_path}: {line}")
#                 continue
#             targets.append([class_id, x_center, y_center, width, height, 1.0])
#             class_id = int(line.split()[0])
#             if class_id >= yolov5_model.nc:
#                 print(f"Warning: class_id {class_id} in {file} is out of range!")
#
#     # 转换为 Tensor 并移动到设备上
#     targets = torch.tensor(targets, dtype=torch.float32).to(device)
#     # Ensure the target's grid location is valid
#     if (targets[:, 1] < 0 or targets[:, 1] >= 1) or (targets[:, 2] < 0 or targets[:, 2] >= 1):
#         print(f"Invalid center for bbox: {targets[:, 1:3]}")  # print invalid target center
#
#     # 确保类别索引有效
#     if (targets[:, 0] >= yolov5_model.nc).any():
#         raise ValueError(f"Invalid class index in {label_path}")
#
#     return targets


# def get_targets(img_name):
#     # 假设标签存储在 'data/labels/' 文件夹中，与图像同名
#     label_path = os.path.join('data/labels', img_name.replace('.jpg', '.txt'))
#     targets = []
#     with open(label_path, 'r') as f:
#         for line in f:
#             class_id, x_center, y_center, width, height = map(float, line.strip().split())
#             targets.append([class_id, x_center, y_center, width, height, 1.0])  # 添加置信度值 1.0
#
#     # 转换为 Tensor 并移动到设备上
#     targets = torch.tensor(targets, dtype=torch.float32).to(device)
#     return targets
# 训练过程
def train():
    epoch_losses = []
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(80):  # 假设训练50个epoch
        epoch_loss = 0
        rec_loss_epoch = 0
        detect_loss_epoch = 0

        yolov5_model.train()  # 设置 YOLOv5 为训练模式

        for img_tensor, img_name in zip(train_images, train_names):
            img_tensor = img_tensor.unsqueeze(0).to(device)  # 添加 batch 维度

            # CNN-PP 预测增强参数并应用 DIP 增强
            gamma, wb, contrast, sharpen_strength = cnn_pp(img_tensor)
            enhanced_img = dip_module(img_tensor, gamma, wb, contrast, sharpen_strength)

            # 获取标签（根据你的数据集加载目标）
            targets = get_targets(img_name,yolov5_model,device)

            # 计算联合损失（增强损失 + YOLOv5 目标检测损失）
            total_loss, rec_loss, detect_loss = combined_loss(enhanced_img, img_tensor, yolov5_model, targets, compute_loss)

            # 反向传播与优化
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            rec_loss_epoch += rec_loss.item()
            detect_loss_epoch += detect_loss.item()

        # 输出每个 epoch 的损失值
        avg_loss = epoch_loss / len(train_images)
        avg_rec_loss = rec_loss_epoch / len(train_images)
        avg_detect_loss = detect_loss_epoch / len(train_images)
        print(
            f"Epoch [{epoch + 1}/50], Total Loss: {avg_loss:.4f}, Recovery Loss: {avg_rec_loss:.4f}, Detection Loss: {avg_detect_loss:.4f}")

        epoch_losses.append((avg_loss, avg_rec_loss, avg_detect_loss))
        save_model_checkpoint(epoch)  # 保存每个 epoch 的模型

    return epoch_losses
# def train():
#     epoch_losses = []
#     for epoch in range(50):  # 假设训练50个epoch
#         epoch_loss = 0
#         rec_loss_epoch = 0
#         detect_loss_epoch = 0
#         yolov5_model.train()  # YOLOv5 设置为训练模式
#         for img_tensor, img_name in zip(train_images, train_names):
#             img_tensor = img_tensor.unsqueeze(0).to(device)  # 添加batch维度
#
#             # CNN-PP预测增强参数并应用DIP增强
#             gamma, wb, contrast, sharpen_strength = cnn_pp(img_tensor)
#             enhanced_img = dip_module(img_tensor, gamma, wb, contrast, sharpen_strength)
#
#             # 获取标签（需要自己定义数据集标签格式，示例中直接假设）
#             targets = get_targets(img_name)  # 根据图像名加载目标检测标签
#
#             # 计算联合损失（增强损失 + YOLOv5目标检测损失）
#             total_loss, rec_loss, detect_loss = combined_loss(enhanced_img, img_tensor, yolov5_model, targets)
#
#             # 反向传播与优化
#             optimizer.zero_grad()
#             total_loss.backward()
#             optimizer.step()
#
#             epoch_loss += total_loss.item()
#             rec_loss_epoch += rec_loss.item()
#             detect_loss_epoch += detect_loss.item()
#
#         # 输出每个epoch的损失值
#         avg_loss = epoch_loss / len(train_images)
#         avg_rec_loss = rec_loss_epoch / len(train_images)
#         avg_detect_loss = detect_loss_epoch / len(train_images)
#         print(f"Epoch [{epoch + 1}/50], Total Loss: {avg_loss:.4f}, Recovery Loss: {avg_rec_loss:.4f}, Detection Loss: {avg_detect_loss:.4f}")
#
#         epoch_losses.append((avg_loss, avg_rec_loss, avg_detect_loss))
#         save_model_checkpoint(epoch)  # 保存每个epoch的模型
#
#     return epoch_losses




def save_model_checkpoint(epoch):
    # 保存模型权重
    torch.save(yolov5_model.state_dict(), f"results/checkpoints/yolov5_epoch_{epoch}.pt")


# 训练并保存损失
epoch_losses = train()

# 绘制损失曲线
losses = list(zip(*epoch_losses))
plt.plot(losses[0], label='Total Loss')
plt.plot(losses[1], label='Recovery Loss')
plt.plot(losses[2], label='Detection Loss')
plt.legend()
plt.savefig("results/metrics/loss_curve.png")
