import sys
import re
import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from config import ckpt_dir
import matplotlib as mpl
from logger import logger
import seaborn as sns
import pandas as pd
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def extract_metrics_from_checkpoints():
    """从检查点文件中提取训练指标"""
    metrics = []
    try:
        # 获取所有检查点文件
        checkpoint_files = []
        if os.path.exists(ckpt_dir):
            # 获取所有epoch检查点
            epoch_files = [f for f in os.listdir(ckpt_dir) if f.startswith('epoch_') and f.endswith('.tar')]
            # 获取best检查点
            if os.path.exists(os.path.join(ckpt_dir, 'best_ckpt.tar')):
                epoch_files.append('best_ckpt.tar')
            checkpoint_files = [os.path.join(ckpt_dir, f) for f in epoch_files]
        
        if not checkpoint_files:
            logger.warning(f"在 {ckpt_dir} 中未找到检查点文件")
            return []
            
        # 从每个检查点提取指标
        for ckpt_file in checkpoint_files:
            try:
                # 加载检查点
                checkpoint = torch.load(ckpt_file, map_location='cpu')
                
                # 提取指标
                metric = {
                    'epoch': int(checkpoint.get('epoch', 0)),
                    'global_step': int(checkpoint.get('global_step', 0)),
                    'loss': float(checkpoint.get('loss', float('inf'))),
                    'bleu': float(checkpoint.get('bleu', 0.0)),
                    'best_loss': float(checkpoint.get('best_loss', float('inf'))),
                    'best_bleu': float(checkpoint.get('best_bleu', 0.0)),
                    'timestamp': checkpoint.get('timestamp', ''),
                    'learning_rate': float(checkpoint.get('config', {}).get('learning_rate', 0.0)),
                    'batch_size': int(checkpoint.get('config', {}).get('batch_size', 0)),
                    'is_best': ckpt_file.endswith('best_ckpt.tar')
                }
                
                # 验证必要字段
                if all(v is not None for v in metric.values()):
                    metrics.append(metric)
                else:
                    logger.warning(f"检查点文件 {ckpt_file} 缺少必要字段")
                
            except Exception as e:
                logger.warning(f"处理检查点文件 {ckpt_file} 时出错: {str(e)}")
                continue
                
        # 按epoch排序
        metrics.sort(key=lambda x: x['epoch'])
        
        if not metrics:
            logger.warning("没有找到有效的训练指标数据")
        else:
            logger.info(f"成功加载 {len(metrics)} 个检查点的训练指标")
            
        return metrics
        
    except Exception as e:
        logger.error(f"提取训练指标时出错: {str(e)}")
        return []

def plot_training_progress(metrics):
    """绘制训练进度图"""
    if not metrics:
        logger.warning("没有可用的训练指标数据")
        return
        
    try:
        # 创建图表
        plt.figure(figsize=(15, 10))
        
        # 1. 损失曲线
        plt.subplot(2, 2, 1)
        epochs = [m['epoch'] for m in metrics]
        losses = [m['loss'] for m in metrics]
        best_losses = [m['best_loss'] for m in metrics]
        
        plt.plot(epochs, losses, 'b-', label='验证损失')
        plt.plot(epochs, best_losses, 'r--', label='最佳损失')
        plt.scatter(epochs, losses, c='blue', alpha=0.5)
        plt.scatter(epochs, best_losses, c='red', alpha=0.5)
        
        # 标记最佳点
        best_idx = np.argmin(best_losses)
        plt.scatter(epochs[best_idx], best_losses[best_idx], c='red', s=100, marker='*', label='最佳点')
        
        plt.title('训练损失变化')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True)
        
        # 2. BLEU分数曲线
        plt.subplot(2, 2, 2)
        bleu_scores = [m['bleu'] for m in metrics]
        best_bleu = [m['best_bleu'] for m in metrics]
        
        plt.plot(epochs, bleu_scores, 'g-', label='BLEU分数')
        plt.plot(epochs, best_bleu, 'y--', label='最佳BLEU')
        plt.scatter(epochs, bleu_scores, c='green', alpha=0.5)
        plt.scatter(epochs, best_bleu, c='yellow', alpha=0.5)
        
        # 标记最佳点
        best_bleu_idx = np.argmax(best_bleu)
        plt.scatter(epochs[best_bleu_idx], best_bleu[best_bleu_idx], c='yellow', s=100, marker='*', label='最佳点')
        
        plt.title('BLEU分数变化')
        plt.xlabel('Epoch')
        plt.ylabel('BLEU分数')
        plt.legend()
        plt.grid(True)
        
        # 3. 学习率变化
        plt.subplot(2, 2, 3)
        learning_rates = [m['learning_rate'] for m in metrics]
        plt.plot(epochs, learning_rates, 'm-', label='学习率')
        plt.scatter(epochs, learning_rates, c='magenta', alpha=0.5)
        plt.title('学习率变化')
        plt.xlabel('Epoch')
        plt.ylabel('学习率')
        plt.legend()
        plt.grid(True)
        
        # 4. 训练进度
        plt.subplot(2, 2, 4)
        global_steps = [m['global_step'] for m in metrics]
        plt.plot(epochs, global_steps, 'c-', label='全局步数')
        plt.scatter(epochs, global_steps, c='cyan', alpha=0.5)
        plt.title('训练进度')
        plt.xlabel('Epoch')
        plt.ylabel('全局步数')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(ckpt_dir, 'training_progress.png'))
        plt.close()
        
        # 计算并打印改进百分比
        if len(metrics) > 1:
            start_loss = metrics[0]['loss']
            last_loss = metrics[-1]['loss']
            best_loss = min(m['loss'] for m in metrics)
            
            start_bleu = metrics[0]['bleu']
            last_bleu = metrics[-1]['bleu']
            best_bleu = max(m['bleu'] for m in metrics)
            
            loss_improvement = (start_loss - best_loss) / start_loss * 100
            bleu_improvement = (best_bleu - start_bleu) / start_bleu * 100 if start_bleu > 0 else 0
            
            logger.info(f"从开始到最佳点的改进:")
            logger.info(f"损失改进: {loss_improvement:.2f}%")
            logger.info(f"BLEU改进: {bleu_improvement:.2f}%")
            
    except Exception as e:
        logger.error(f"绘制训练进度图时出错: {str(e)}")

def plot_comparative_metrics(metrics):
    """绘制比较性指标图"""
    if not metrics:
        return
        
    try:
        # 将数据分为早期、中期和晚期
        total_epochs = len(metrics)
        early_idx = total_epochs // 3
        mid_idx = 2 * total_epochs // 3
        
        early_metrics = metrics[:early_idx]
        mid_metrics = metrics[early_idx:mid_idx]
        late_metrics = metrics[mid_idx:]
        
        # 计算每个阶段的平均指标
        stages = {
            '早期': early_metrics,
            '中期': mid_metrics,
            '晚期': late_metrics
        }
        
        # 创建图表
        plt.figure(figsize=(15, 10))
        
        # 1. 损失比较
        plt.subplot(2, 2, 1)
        stage_losses = [np.mean([m['loss'] for m in stage]) for stage in stages.values()]
        plt.bar(stages.keys(), stage_losses, color=['blue', 'green', 'red'])
        plt.title('各阶段平均损失比较')
        plt.ylabel('平均损失')
        plt.grid(True, axis='y')
        
        # 2. BLEU分数比较
        plt.subplot(2, 2, 2)
        stage_bleu = [np.mean([m['bleu'] for m in stage]) for stage in stages.values()]
        plt.bar(stages.keys(), stage_bleu, color=['blue', 'green', 'red'])
        plt.title('各阶段平均BLEU分数比较')
        plt.ylabel('平均BLEU分数')
        plt.grid(True, axis='y')
        
        # 3. 学习率变化
        plt.subplot(2, 2, 3)
        stage_lr = [np.mean([m['learning_rate'] for m in stage]) for stage in stages.values()]
        plt.bar(stages.keys(), stage_lr, color=['blue', 'green', 'red'])
        plt.title('各阶段平均学习率比较')
        plt.ylabel('平均学习率')
        plt.grid(True, axis='y')
        
        # 4. 训练速度（每epoch的步数）
        plt.subplot(2, 2, 4)
        stage_steps = [np.mean([m['global_step'] / m['epoch'] for m in stage if m['epoch'] > 0]) for stage in stages.values()]
        plt.bar(stages.keys(), stage_steps, color=['blue', 'green', 'red'])
        plt.title('各阶段训练速度比较')
        plt.ylabel('平均步数/epoch')
        plt.grid(True, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(ckpt_dir, 'comparative_metrics.png'))
        plt.close()
        
    except Exception as e:
        logger.error(f"绘制比较性指标图时出错: {str(e)}")

def plot_training_heatmap(metrics):
    """绘制训练热力图"""
    if not metrics:
        return
        
    try:
        # 准备数据
        df = pd.DataFrame(metrics)
        
        # 选择要可视化的指标
        metrics_to_plot = ['loss', 'bleu', 'learning_rate', 'global_step']
        
        # 创建图表
        plt.figure(figsize=(15, 10))
        
        # 计算相关性矩阵
        corr_matrix = df[metrics_to_plot].corr()
        
        # 绘制热力图
        sns.heatmap(corr_matrix, 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   fmt='.2f',
                   square=True)
        
        plt.title('训练指标相关性热力图')
        plt.tight_layout()
        plt.savefig(os.path.join(ckpt_dir, 'training_heatmap.png'))
        plt.close()
        
    except Exception as e:
        logger.error(f"绘制训练热力图时出错: {str(e)}")

def plot_training_summary(metrics):
    """绘制训练总结图"""
    if not metrics:
        return
        
    try:
        # 创建图表
        plt.figure(figsize=(15, 10))
        
        # 1. 训练时间线
        plt.subplot(2, 2, 1)
        timestamps = [datetime.fromisoformat(m['timestamp']) for m in metrics if m['timestamp']]
        if timestamps:
            time_diffs = [(t - timestamps[0]).total_seconds() / 3600 for t in timestamps]  # 转换为小时
            plt.plot(time_diffs, [m['loss'] for m in metrics], 'b-', label='损失')
            plt.plot(time_diffs, [m['bleu'] for m in metrics], 'g-', label='BLEU')
            plt.title('训练时间线')
            plt.xlabel('训练时间（小时）')
            plt.ylabel('指标值')
            plt.legend()
            plt.grid(True)
        
        # 2. 训练效率
        plt.subplot(2, 2, 2)
        steps_per_epoch = [m['global_step'] / m['epoch'] for m in metrics if m['epoch'] > 0]
        if steps_per_epoch:  # 确保有数据再绘图
            plt.plot(range(len(steps_per_epoch)), steps_per_epoch, 'r-')
            plt.title('训练效率（每epoch步数）')
            plt.xlabel('Epoch')
            plt.ylabel('步数/epoch')
            plt.grid(True)
        
        # 3. 指标分布
        plt.subplot(2, 2, 3)
        if len(metrics) > 1:  # 确保有足够的数据进行分布分析
            sns.histplot(data=[m['loss'] for m in metrics], bins=min(20, len(metrics)), color='blue', alpha=0.5, label='损失')
            sns.histplot(data=[m['bleu'] for m in metrics], bins=min(20, len(metrics)), color='green', alpha=0.5, label='BLEU')
            plt.title('指标分布')
            plt.xlabel('指标值')
            plt.ylabel('频率')
            plt.legend()
        
        # 4. 训练稳定性
        plt.subplot(2, 2, 4)
        if len(metrics) > 1:  # 确保有足够的数据计算变化
            loss_changes = np.diff([m['loss'] for m in metrics])
            bleu_changes = np.diff([m['bleu'] for m in metrics])
            plt.plot(range(len(loss_changes)), loss_changes, 'b-', label='损失变化')
            plt.plot(range(len(bleu_changes)), bleu_changes, 'g-', label='BLEU变化')
            plt.title('训练稳定性')
            plt.xlabel('检查点')
            plt.ylabel('变化量')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(ckpt_dir, 'training_summary.png'))
        plt.close()
        
    except Exception as e:
        logger.error(f"绘制训练总结图时出错: {str(e)}")
        # 打印更详细的错误信息
        import traceback
        logger.error(traceback.format_exc())

def visualize_training():
    """可视化训练过程"""
    try:
        # 提取训练指标
        metrics = extract_metrics_from_checkpoints()
        
        if not metrics:
            logger.warning("没有找到可用的训练指标数据")
            return
            
        # 绘制各种图表
        plot_training_progress(metrics)  # 训练进度图
        plot_comparative_metrics(metrics)  # 比较性指标图
        plot_training_heatmap(metrics)  # 训练热力图
        plot_training_summary(metrics)  # 训练总结图
        
        logger.info("训练可视化完成，图表已保存到检查点目录")
        
    except Exception as e:
        logger.error(f"训练可视化过程出错: {str(e)}")

if __name__ == "__main__":
    print("生成训练指标可视化...")
    visualize_training() 