# Transformer 翻译模型

基于Transformer架构的高性能神经机器翻译模型，支持多种优化技术。

## 项目特点

- 完整实现了《Attention Is All You Need》论文中的Transformer架构
- 支持多种现代优化技术（混合精度训练、Transformer Engine、Flash Attention等）
- 高度可配置，可根据不同硬件进行优化
- 内置数据预处理和分词工具（基于SentencePiece）
- 支持Beam Search和Greedy Search解码策略

## 快速开始

### 安装

请查看 [INSTALL.md](INSTALL.md) 获取详细安装说明。

基础安装:
```bash
pip install -r requirements_core.txt
```

### 训练模型

```bash
python main.py --mode train
```

### 使用模型进行推理

```bash
python main.py --mode inference --input "这是一个测试句子。" --decode beam
```

## 系统要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (推荐使用CUDA 12.1)
- 至少6GB GPU内存（推荐8GB+用于大批量训练）

## 主要组件

- **main.py**: 程序入口点
- **config.py**: 配置参数管理
- **transformer.py**: Transformer模型实现
- **layers.py**: 注意力机制和网络层实现
- **train_manager.py**: 训练流程管理
- **inference_manager.py**: 推理流程管理
- **custom_data.py**: 数据加载和预处理

## 性能优化

本项目实现了多种现代深度学习优化技术:

1. **混合精度训练**: 降低内存使用并提高计算速度
2. **内存优化注意力机制**: 减少内存占用
3. **Flash Attention**: 显著提高注意力计算效率
4. **Transformer Engine**: 利用NVIDIA优化的Transformer实现
5. **模型编译**: 使用torch.compile加速推理和训练
6. **融合操作**: 使用融合的Adam优化器和LayerNorm
7. **梯度累积**: 支持更大的有效批量大小

## 自定义和扩展

要调整模型超参数或优化选项，请编辑 `config.py` 文件中的 `MODEL_CONFIG` 字典。


## 致谢

- PyTorch团队
- 《Attention Is All You Need》论文作者
- SentencePiece团队 