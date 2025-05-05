import sys
import re
import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from config import ckpt_dir

def extract_losses_from_checkpoints():
    """从检查点文件中提取训练损失和验证损失"""
    if not os.path.exists(ckpt_dir):
        print(f"检查点目录 {ckpt_dir} 不存在")
        return None, None
    
    checkpoints = glob.glob(f"{ckpt_dir}/epoch_*.tar")
    if not checkpoints:
        print("没有找到任何epoch检查点")
        return None, None
    
    # 提取所有epoch号
    epochs = []
    for ckpt in checkpoints:
        match = re.search(r'epoch_(\d+)\.tar', ckpt)
        if match:
            epochs.append(int(match.group(1)))
    
    if not epochs:
        print("无法从检查点文件名中提取epoch信息")
        return None, None
    
    # 对epoch排序
    epochs.sort()
    
    # 提取每个epoch的验证损失
    valid_losses = []
    for epoch in epochs:
        ckpt_path = f"{ckpt_dir}/epoch_{epoch}.tar"
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
            valid_losses.append(checkpoint['loss'])
    
    # 读取最佳检查点的损失
    best_loss = None
    if os.path.exists(f"{ckpt_dir}/best_ckpt.tar"):
        checkpoint = torch.load(f"{ckpt_dir}/best_ckpt.tar", map_location=torch.device('cpu'))
        best_loss = checkpoint['loss']
        best_epoch = checkpoint.get('epoch', None)
        
    return epochs, valid_losses, best_epoch, best_loss

def plot_training_progress():
    """绘制训练进度曲线"""
    epochs, valid_losses, best_epoch, best_loss = extract_losses_from_checkpoints()
    
    if not epochs or not valid_losses:
        print("没有可用数据来绘制训练进度")
        return
    
    plt.figure(figsize=(12, 6))
    
    # 绘制验证损失曲线
    plt.plot(epochs, valid_losses, 'b-', marker='o', label='验证损失')
    
    # 标记最佳损失点
    if best_epoch is not None and best_loss is not None:
        plt.plot(best_epoch, best_loss, 'r*', markersize=15, label=f'最佳损失 ({best_loss:.4f})')
    
    # 添加标题和标签
    plt.title('训练进度')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.grid(True)
    plt.legend()
    
    # 保存图表
    plt.savefig(f"{ckpt_dir}/training_progress.png")
    print(f"训练进度图表已保存到 {ckpt_dir}/training_progress.png")
    
    # 显示改进百分比
    if len(valid_losses) > 1:
        improvement = (valid_losses[0] - valid_losses[-1]) / valid_losses[0] * 100
        print(f"从开始到最后一个epoch的验证损失改善: {improvement:.2f}%")
    
    if best_loss is not None:
        improvement = (valid_losses[0] - best_loss) / valid_losses[0] * 100
        print(f"从开始到最佳损失的改善: {improvement:.2f}%")
    
    # 显示图表
    plt.show()

def plot_comparative_losses():
    """绘制损失对比图，展示不同训练阶段的改进情况"""
    epochs, valid_losses, best_epoch, best_loss = extract_losses_from_checkpoints()
    
    if not epochs or not valid_losses:
        print("没有可用数据来绘制损失对比图")
        return
        
    # 将训练过程分为三个阶段
    num_epochs = len(epochs)
    early_stage = valid_losses[:max(1, num_epochs//3)]
    mid_stage = valid_losses[max(1, num_epochs//3):max(2, num_epochs*2//3)]
    late_stage = valid_losses[max(2, num_epochs*2//3):]
    
    # 创建柱状图
    plt.figure(figsize=(10, 6))
    
    # 计算每个阶段的平均损失
    stages = ['初始阶段', '中间阶段', '后期阶段']
    avg_losses = [np.mean(early_stage), np.mean(mid_stage), np.mean(late_stage)]
    
    # 绘制柱状图
    bars = plt.bar(stages, avg_losses, color=['#FF9999', '#66B2FF', '#99FF99'])
    
    # 在柱子上添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.4f}', ha='center', va='bottom')
    
    # 添加标题和标签
    plt.title('训练阶段损失对比')
    plt.ylabel('平均验证损失')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # 保存图表
    plt.savefig(f"{ckpt_dir}/loss_comparison.png")
    print(f"损失对比图已保存到 {ckpt_dir}/loss_comparison.png")
    
    # 显示图表
    plt.show()

if __name__ == "__main__":
    print("生成训练进度可视化...")
    plot_training_progress()
    plot_comparative_losses() 