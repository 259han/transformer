# Transformer Translator (PyTorch)
# Transformer 翻译器 (PyTorch实现)

[English](#english) | [中文](#chinese)

<a name="english"></a>
## English

A neural machine translation model based on the Transformer architecture, specifically designed for English-to-French translation. This implementation features various optimization techniques for training and inference.

### Features

- Complete implementation of the Transformer architecture from "Attention Is All You Need" paper
- Multiple modern optimization techniques (mixed precision training, gradient checkpointing, etc.)
- Highly configurable to optimize for different hardware setups
- Built-in data preprocessing and tokenization using SentencePiece
- Support for both Beam Search and Greedy Search decoding strategies
- Training with learning rate scheduling and early stopping
- Evaluation using BLEU score
- Interactive translation mode for easy usage
- Batch file translation for processing large documents

### Project Structure

```
transformer-translator-pytorch/
├── data/                      # Data directory
│   ├── cache/                 # Cached processed data
│   ├── corpus/                # Raw parallel corpus files
│   ├── processed/             # Processed train/valid data
│   └── sp/                    # SentencePiece models and vocabularies
├── saved_model/               # Saved model checkpoints
├── config.py                  # Configuration parameters
├── custom_data.py             # Data loading and preprocessing
├── data_structure.py          # Data structures for beam search
├── data_validation.py         # Data validation utilities
├── inference_manager.py       # Inference pipeline
├── layers.py                  # Transformer sublayers implementation
├── logger.py                  # Logging utilities
├── main.py                    # Main entry point
├── sentencepiece_train.py     # SentencePiece model training
├── train_manager.py           # Training pipeline
├── transformer.py             # Transformer model implementation
├── bleu_score.py              # BLEU score calculation
└── visualize_training.py      # Training visualization utilities
```

### Quick Start

#### Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

#### Prepare Data

The system expects parallel corpus files in the `data/corpus/` directory. The provided model is trained on English-to-French translation data.

#### Train a Model

```bash
python main.py --mode train
```

Additional training options:
- `--ckpt_name NAME`: Resume training from a specific checkpoint
- `--batch_size SIZE`: Set custom batch size
- `--optimize`: Enable additional performance optimizations
- `--validate_data`: Validate the training data before starting

#### Translation with the Model

The model supports three inference modes: single sentence translation, interactive translation, and batch file translation.

**Currently supported language pair:** English → French

**1. Single Sentence Translation**
```bash
python main.py --mode inference --input "This is a test sentence." --decode beam
```

**2. Interactive Translation Mode**
```bash
python main.py --mode inference
```

In interactive mode, you can:
- Enter any English text to translate it to French
- Type `switch` to toggle between beam and greedy decoding methods
- Type `q`, `exit` or `quit` to exit the program
- Type `help` or `?` to view help information

**3. Batch File Translation**
```bash
python main.py --mode inference --file input.txt --output translated.txt --decode beam
```

Translation options:
- `--decode beam`: Use beam search decoding (default)
- `--decode greedy`: Use greedy search decoding
- `--ckpt_name NAME`: Use a specific checkpoint (defaults to best_ckpt.tar)
- `--file PATH`: Specify input file path for batch translation
- `--output PATH`: Specify output file path for batch translation (optional)

### System Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (CUDA 12.0+ recommended)
- At least 6GB GPU memory (8GB+ recommended for larger batch sizes)

### Performance Optimization

This project implements several modern deep learning optimization techniques:

1. **Mixed Precision Training**: Reduces memory usage and increases computational speed
2. **Gradient Checkpointing**: Trades computation for memory to allow larger models
3. **TensorFloat-32 (TF32)**: Enhanced precision for RTX 30+ series GPUs
4. **JIT Scripting**: Optimizes certain model components
5. **Gradient Accumulation**: Supports larger effective batch sizes
6. **Channels Last Memory Format**: Improves performance on CUDA devices
7. **Label Smoothing**: Improves model generalization

### Configuration

To adjust model hyperparameters or optimization options, edit the `MODEL_CONFIG` and `TRAIN_CONFIG` dictionaries in the `config.py` file.

<a name="chinese"></a>
## 中文

基于Transformer架构的神经机器翻译模型，专为英法翻译设计。该实现包含多种训练和推理优化技术。

### 特点

- 完整实现了"Attention Is All You Need"论文中的Transformer架构
- 支持多种现代优化技术（混合精度训练、梯度检查点等）
- 高度可配置，可根据不同硬件设置进行优化
- 内置基于SentencePiece的数据预处理和分词
- 支持束搜索(Beam Search)和贪婪搜索(Greedy Search)解码策略
- 训练过程包含学习率调度和早停机制
- 使用BLEU分数进行评估
- 交互式翻译模式，便于日常使用
- 批量文件翻译功能，适用于处理大型文档

### 项目结构

```
transformer-translator-pytorch/
├── data/                      # 数据目录
│   ├── cache/                 # 缓存的处理数据
│   ├── corpus/                # 原始平行语料文件
│   ├── processed/             # 处理后的训练/验证数据
│   └── sp/                    # SentencePiece模型和词表
├── saved_model/               # 保存的模型检查点
├── config.py                  # 配置参数
├── custom_data.py             # 数据加载和预处理
├── data_structure.py          # 束搜索的数据结构
├── data_validation.py         # 数据验证工具
├── inference_manager.py       # 推理流程管理
├── layers.py                  # Transformer子层实现
├── logger.py                  # 日志工具
├── main.py                    # 主入口点
├── sentencepiece_train.py     # SentencePiece模型训练
├── train_manager.py           # 训练流程管理
├── transformer.py             # Transformer模型实现
├── bleu_score.py              # BLEU分数计算
└── visualize_training.py      # 训练可视化工具
```

### 快速开始

#### 安装

安装所需依赖：

```bash
pip install -r requirements.txt
```

#### 准备数据

系统期望在`data/corpus/`目录中有平行语料文件。提供的模型是在英译法数据上训练的。

#### 训练模型

```bash
python main.py --mode train
```

其他训练选项：
- `--ckpt_name 名称`：从特定检查点恢复训练
- `--batch_size 大小`：设置自定义批量大小
- `--optimize`：启用额外的性能优化
- `--validate_data`：在开始前验证训练数据

#### 使用模型进行翻译

模型支持三种推理模式：单句翻译、交互式翻译和批量文件翻译。

**当前支持的语言对：** 英语 → 法语

**1. 单句翻译**
```bash
python main.py --mode inference --input "This is a test sentence." --decode beam
```

**2. 交互式翻译模式**
```bash
python main.py --mode inference
```

在交互式模式中，您可以：
- 输入任意英语文本将其翻译为法语
- 输入`switch`在束搜索和贪婪搜索解码方法之间切换
- 输入`q`、`exit`或`quit`退出程序
- 输入`help`或`?`查看帮助信息

**3. 批量文件翻译**
```bash
python main.py --mode inference --file input.txt --output translated.txt --decode beam
```

翻译选项：
- `--decode beam`：使用束搜索解码（默认）
- `--decode greedy`：使用贪婪搜索解码
- `--ckpt_name 名称`：使用特定检查点（默认为best_ckpt.tar）
- `--file 路径`：指定输入文件路径，用于批量翻译
- `--output 路径`：指定输出文件路径（可选）

### 系统需求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+（推荐CUDA 12.0+）
- 至少6GB GPU内存（推荐8GB+用于更大批量大小）

### 性能优化

本项目实现了多种现代深度学习优化技术：

1. **混合精度训练**：减少内存使用并提高计算速度
2. **梯度检查点**：用计算换内存，允许更大的模型
3. **TensorFloat-32 (TF32)**：为RTX 30+系列GPU提供增强精度
4. **JIT脚本化**：优化某些模型组件
5. **梯度累积**：支持更大的有效批量大小
6. **Channels Last内存格式**：提高CUDA设备上的性能
7. **标签平滑**：改善模型泛化能力

### 配置

要调整模型超参数或优化选项，请编辑`config.py`文件中的`MODEL_CONFIG`和`TRAIN_CONFIG`字典。

## 主要组件

- **main.py**: 程序入口点
- **config.py**: 配置参数管理
- **transformer.py**: Transformer模型实现
- **layers.py**: 注意力机制和网络层实现
- **train_manager.py**: 训练流程管理
- **inference_manager.py**: 推理流程管理
- **custom_data.py**: 数据加载和预处理

## 自定义和扩展

- 要调整模型超参数或优化选项，请编辑 `config.py` 文件中的 `MODEL_CONFIG` 字典
- 要扩展到其他语言对，需要准备新的平行语料库并重新训练SentencePiece模型
- 要实现双向翻译，请参考issue或相关讨论获取详细指南

## 致谢

- PyTorch团队
- 《Attention Is All You Need》论文作者
- SentencePiece团队 