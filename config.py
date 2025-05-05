import torch
import torch.backends.cudnn as cudnn

# 路径相关配置
DATA_CONFIG = {
    'data_dir': 'data',  # 数据主目录
    'sp_dir': 'data/sp',  # SentencePiece模型目录
    'src_dir': 'data/processed/src',  # 处理后源语言数据目录
    'trg_dir': 'data/processed/trg',  # 处理后目标语言数据目录
    'src_raw_data': 'data/corpus/e.fr-en.src',  # 原始源语言语料
    'trg_raw_data': 'data/corpus/e.fr-en.trg',  # 原始目标语言语料
    'train_file': 'train.txt',  # 训练集文件名
    'valid_file': 'valid.txt',  # 验证集文件名
    'test_file': 'test.txt',    # 测试集文件名
    'ckpt_dir': 'saved_model',  # 检查点保存目录
    'cache_dir': 'data/cache',  # 数据缓存目录
    'log_dir': 'logs'           # 日志目录
}

# Token ID配置
TOKEN_CONFIG = {
    'pad_id': 0,                # <pad> token id
    'sos_id': 1,                # <s> (start of sentence) token id
    'eos_id': 2,                # </s> (end of sentence) token id
    'unk_id': 3,                # <unk> token id
    'src_model_prefix': 'src_sp', # 源语言分词模型前缀
    'trg_model_prefix': 'trg_sp', # 目标语言分词模型前缀
    'sp_vocab_size': 16000,     # SentencePiece词表大小
    'character_coverage': 1.0,  # 覆盖字符比例
    'model_type': 'unigram'     # SentencePiece模型类型
}

# 模型与训练超参数配置
MODEL_CONFIG = {
    'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'), # 训练设备
    'learning_rate': 3e-4,        # 降低初始学习率至1e-4
    'batch_size': 32,             # 增加批量大小以提高效率
    'seq_len': 170,               # 最大序列长度
    'num_heads': 8,               # 多头注意力头数
    'num_layers': 6,              # 编码器/解码器层数
    'd_model': 512,               # Transformer隐藏层维度
    'd_ff': 2048,                 # 前馈层维度
    'drop_out_rate': 0.15,        # Dropout比例
    'num_epochs': 50,             # 训练轮数
    'beam_size': 5,               # Beam Search宽度
    'num_workers': 4,             # DataLoader工作线程数
    'pin_memory': True,           # DataLoader是否锁页内存
    'benchmark_cudnn': True,      # 是否启用cudnn.benchmark
    'clip_grad_norm': 1.0,        # 将梯度裁剪降低到1.0
    'mixed_precision': 'bf16',    # 使用BF16混合精度训练，精度更高且训练更稳定(需要PyTorch>=1.12)
    'gradient_accumulation': 3,   # 增加梯度累积步数以允许更大的有效批量
    'weight_decay': 1e-4,         # 添加权重衰减参数
    'label_smoothing': 0.1,       # 添加标签平滑参数
    'warmup_steps': 4000,         # 添加warmup步数
    'lr_scheduler': 'inverse_sqrt', # 逆平方根衰减（原Transformer论文中的策略）
    'validate_every_steps': 2000, # 每隔多少步验证一次
    'early_stopping_patience': 5, # 早停容忍轮数
    'early_stopping_min_delta': 0.001, # 早停最小提升值
    'prefetch_to_gpu': True,      # 开启预取数据到GPU
    'save_epoch_checkpoints': True, # 是否保存每个epoch的检查点
    'save_epoch_frequency': 1,     # 每隔多少个epoch保存一次检查点
    'keep_last_n_checkpoints': 3,  # 保留最近的N个epoch检查点
    'use_tf32': True,             # 启用TensorFloat-32，对RTX 30系列有加速
    'channels_last_memory_format': True, # channels_last内存格式
    'deterministic': False,        # 是否启用确定性计算
    'enable_cudnn_autotuner': True, # 启用CUDNN自动调优
    'optimize_memory_usage': True,  # 优化内存使用
    'jit_script_encoder': True,     # 使用JIT脚本优化编码器
    'jit_script_decoder': True,     # 使用JIT脚本优化解码器
}

# 日志配置
LOG_CONFIG = {
    'level': 'warning',           # 日志级别: debug, info, warning, error, critical
    'console': True,           # 是否输出到控制台
    'log_file': True,          # 是否输出到文件
    'max_files': 10,            # 保留最大日志文件数
    'rotation': '1 day'        # 日志轮换周期
}

# 运行/推理/数据相关配置
TRAIN_CONFIG = {
    'ckpt_name': 'best_ckpt.tar',  # 检查点文件名
    'decode_method': 'beam',       # 解码方法: 'greedy' 或 'beam'
    'auto_resume': True,           # 是否自动查找最新检查点并继续训练
    'resume_epoch': None,          # 从哪个epoch继续训练
    'clear_cache': False,          # 是否在启动时清除缓存
    'data_limit': {                # 数据量限制
        'train': None,             # 训练集不再限制数据量
        'valid': 10000             # 验证集最大行数增加到10000
    },
    'sentencepiece': {             # SentencePiece相关配置
        'raw_data_limit': 300000,    # 原始语料不再限制
        'train_frac': 0.8,         # 训练集比例
        'regenerate': False        # 是否重新生成分词模型
    },
    'evaluation': {                # 评估相关配置
        'calculate_bleu': True,    # 是否计算BLEU分数
        'bleu_sample_size': 1000,   # BLEU评估样本数增加
        'save_translations': True, # 是否保存翻译结果
        'translation_output': 'translations.txt' # 翻译结果输出文件
    },
    'optimization': {              # 优化相关配置
        'enable_jit': True,         # 启用JIT编译优化
        'optimize_memory': True,    # 优化内存使用
        'use_gradient_checkpointing': True, # 使用梯度检查点
        'compile_model': False,     # 不启用compile_model优化（需要较新版PyTorch）
    }
}

# 是否启用断点续训功能
ENABLE_RESUME = True

# 导出变量供其它模块使用
DATA_DIR = DATA_CONFIG['data_dir']
SP_DIR = DATA_CONFIG['sp_dir']
SRC_DIR = DATA_CONFIG['src_dir']
TRG_DIR = DATA_CONFIG['trg_dir']
SRC_RAW_DATA_NAME = DATA_CONFIG['src_raw_data']
TRG_RAW_DATA_NAME = DATA_CONFIG['trg_raw_data']
TRAIN_NAME = DATA_CONFIG['train_file']
VALID_NAME = DATA_CONFIG['valid_file']
TEST_NAME = DATA_CONFIG['test_file']
ckpt_dir = DATA_CONFIG['ckpt_dir']

pad_id = TOKEN_CONFIG['pad_id']
sos_id = TOKEN_CONFIG['sos_id']
eos_id = TOKEN_CONFIG['eos_id']
unk_id = TOKEN_CONFIG['unk_id']
src_model_prefix = TOKEN_CONFIG['src_model_prefix']
trg_model_prefix = TOKEN_CONFIG['trg_model_prefix']
sp_vocab_size = TOKEN_CONFIG['sp_vocab_size']
character_coverage = TOKEN_CONFIG['character_coverage']
model_type = TOKEN_CONFIG['model_type']

device = MODEL_CONFIG['device']
learning_rate = MODEL_CONFIG['learning_rate']
batch_size = MODEL_CONFIG['batch_size']
seq_len = MODEL_CONFIG['seq_len']
num_heads = MODEL_CONFIG['num_heads']
num_layers = MODEL_CONFIG['num_layers']
d_model = MODEL_CONFIG['d_model']
d_ff = MODEL_CONFIG['d_ff']
d_k = MODEL_CONFIG['d_model'] // MODEL_CONFIG['num_heads']
drop_out_rate = MODEL_CONFIG['drop_out_rate']
num_epochs = MODEL_CONFIG['num_epochs']
beam_size = MODEL_CONFIG['beam_size']
num_workers = MODEL_CONFIG['num_workers']
pin_memory = MODEL_CONFIG['pin_memory']
benchmark_cudnn = MODEL_CONFIG['benchmark_cudnn']
clip_grad_norm = MODEL_CONFIG['clip_grad_norm']
mixed_precision = MODEL_CONFIG['mixed_precision']
gradient_accumulation = MODEL_CONFIG['gradient_accumulation']
weight_decay = MODEL_CONFIG['weight_decay']
label_smoothing = MODEL_CONFIG['label_smoothing']
warmup_steps = MODEL_CONFIG['warmup_steps']
validate_every_steps = MODEL_CONFIG['validate_every_steps']
early_stopping_patience = MODEL_CONFIG['early_stopping_patience']
early_stopping_min_delta = MODEL_CONFIG['early_stopping_min_delta']
prefetch_to_gpu = MODEL_CONFIG['prefetch_to_gpu']
save_epoch_checkpoints = MODEL_CONFIG['save_epoch_checkpoints']
save_epoch_frequency = MODEL_CONFIG['save_epoch_frequency']
keep_last_n_checkpoints = MODEL_CONFIG['keep_last_n_checkpoints']
use_tf32 = MODEL_CONFIG['use_tf32']
channels_last_memory_format = MODEL_CONFIG['channels_last_memory_format']
deterministic = MODEL_CONFIG['deterministic']
enable_cudnn_autotuner = MODEL_CONFIG.get('enable_cudnn_autotuner', False)
optimize_memory_usage = MODEL_CONFIG.get('optimize_memory_usage', False)
jit_script_encoder = MODEL_CONFIG.get('jit_script_encoder', False)
jit_script_decoder = MODEL_CONFIG.get('jit_script_decoder', False)

# 根据配置参数设置cudnn.benchmark
if torch.cuda.is_available():
    if benchmark_cudnn:
        cudnn.benchmark = True
    # 启用CUDNN自动调优
    if enable_cudnn_autotuner:
        cudnn.enabled = True
    # 对于RTX 30系列GPU，启用TF32精度
    if use_tf32 and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("启用TensorFloat-32精度")

# 根据配置设置确定性计算
if deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

# 辅助函数
def get_config_value(path, default=None):
    """
    从配置中获取嵌套路径的值
    
    参数:
    - path: 点分隔的配置路径，例如 'performance.batch_size'
    - default: 如果路径不存在，返回的默认值
    
    返回:
    - 配置值或默认值
    """
    try:
        parts = path.split('.')
        if len(parts) == 1:
            return TRAIN_CONFIG.get(parts[0], default)
        
        current = TRAIN_CONFIG
        for part in parts:
            if part not in current:
                return default
            current = current[part]
        return current
    except:
        return default 