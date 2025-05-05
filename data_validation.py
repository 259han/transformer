from config import *
from logger import logger
import os
import torch
import sentencepiece as spm
import random

def setup_data_directories():
    """创建项目所需的所有数据目录"""
    required_dirs = [
        DATA_DIR,
        SP_DIR,
        SRC_DIR,
        TRG_DIR,
        f"{DATA_DIR}/cache",
        f"{DATA_DIR}/corpus",
        ckpt_dir,
        LOG_CONFIG.get('log_dir', 'logs')
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            logger.info(f"创建目录: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
    
    logger.info("数据目录结构设置完成")

def validate_directory_structure():
    """验证项目所需的目录结构是否存在，如果不存在则创建"""
    required_dirs = [
        DATA_DIR,
        SP_DIR,
        SRC_DIR,
        TRG_DIR,
        f"{DATA_DIR}/cache",
        ckpt_dir
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
            logger.warning(f"目录不存在，将创建: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
    
    if missing_dirs:
        logger.info(f"已创建 {len(missing_dirs)} 个缺失的目录")
    else:
        logger.info("所有必需的目录已存在")
    
    return len(missing_dirs) == 0

def validate_raw_data():
    """验证原始数据文件是否存在和可读"""
    src_exists = os.path.exists(SRC_RAW_DATA_NAME)
    trg_exists = os.path.exists(TRG_RAW_DATA_NAME)
    
    if not src_exists:
        logger.error(f"源语言原始数据文件不存在: {SRC_RAW_DATA_NAME}")
    
    if not trg_exists:
        logger.error(f"目标语言原始数据文件不存在: {TRG_RAW_DATA_NAME}")
    
    if not src_exists or not trg_exists:
        return False
    
    # 检查文件是否可读
    try:
        with open(SRC_RAW_DATA_NAME, 'r', encoding='utf-8') as f:
            src_lines = f.readlines(1000)  # 只读取前1000行进行检查
        
        with open(TRG_RAW_DATA_NAME, 'r', encoding='utf-8') as f:
            trg_lines = f.readlines(1000)  # 只读取前1000行进行检查
        
        # 检查行数是否匹配
        if len(src_lines) != len(trg_lines):
            logger.warning(f"前1000行中，源语言和目标语言数据行数不匹配: 源={len(src_lines)}, 目标={len(trg_lines)}")
        
        # 检查文件内容的基本格式
        for i, (src_line, trg_line) in enumerate(zip(src_lines[:10], trg_lines[:10])):
            if not src_line.strip() or not trg_line.strip():
                logger.warning(f"第{i+1}行包含空内容: src='{src_line.strip()}', trg='{trg_line.strip()}'")
        
        logger.info(f"原始数据文件可读，已检查前1000行")
        return True
    
    except Exception as e:
        logger.error(f"读取原始数据文件时出错: {str(e)}")
        return False

def validate_sp_models():
    """验证SentencePiece模型是否存在和可加载"""
    src_model_path = f"{SP_DIR}/{src_model_prefix}.model"
    trg_model_path = f"{SP_DIR}/{trg_model_prefix}.model"
    src_vocab_path = f"{SP_DIR}/{src_model_prefix}.vocab"
    trg_vocab_path = f"{SP_DIR}/{trg_model_prefix}.vocab"
    
    src_model_exists = os.path.exists(src_model_path) and os.path.exists(src_vocab_path)
    trg_model_exists = os.path.exists(trg_model_path) and os.path.exists(trg_vocab_path)
    
    if not src_model_exists:
        logger.warning(f"源语言SentencePiece模型或词汇表不存在: {src_model_path}, {src_vocab_path}")
    
    if not trg_model_exists:
        logger.warning(f"目标语言SentencePiece模型或词汇表不存在: {trg_model_path}, {trg_vocab_path}")
    
    if not src_model_exists or not trg_model_exists:
        return False
    
    # 尝试加载模型
    try:
        src_sp = spm.SentencePieceProcessor()
        src_sp.Load(src_model_path)
        
        trg_sp = spm.SentencePieceProcessor()
        trg_sp.Load(trg_model_path)
        
        # 测试分词功能
        test_sentence = "This is a test sentence."
        src_tokens = src_sp.EncodeAsIds(test_sentence)
        
        logger.info(f"SentencePiece模型加载成功，词汇表大小: 源={src_sp.GetPieceSize()}, 目标={trg_sp.GetPieceSize()}")
        
        return True
    except Exception as e:
        logger.error(f"加载SentencePiece模型时出错: {str(e)}")
        return False

def validate_processed_data():
    """验证处理后的数据文件是否存在和格式是否正确"""
    src_train_path = f"{SRC_DIR}/{TRAIN_NAME}"
    src_valid_path = f"{SRC_DIR}/{VALID_NAME}"
    trg_train_path = f"{TRG_DIR}/{TRAIN_NAME}"
    trg_valid_path = f"{TRG_DIR}/{VALID_NAME}"
    
    required_files = [
        (src_train_path, "源语言训练数据"),
        (src_valid_path, "源语言验证数据"),
        (trg_train_path, "目标语言训练数据"),
        (trg_valid_path, "目标语言验证数据")
    ]
    
    all_exist = True
    for file_path, desc in required_files:
        if not os.path.exists(file_path):
            logger.warning(f"{desc}文件不存在: {file_path}")
            all_exist = False
    
    if not all_exist:
        return False
    
    # 抽样检查数据格式
    try:
        # 检查文件行数是否匹配
        src_train_lines = sum(1 for _ in open(src_train_path, 'r', encoding='utf-8'))
        trg_train_lines = sum(1 for _ in open(trg_train_path, 'r', encoding='utf-8'))
        
        if src_train_lines != trg_train_lines:
            logger.warning(f"训练集行数不匹配: 源={src_train_lines}, 目标={trg_train_lines}")
        
        src_valid_lines = sum(1 for _ in open(src_valid_path, 'r', encoding='utf-8'))
        trg_valid_lines = sum(1 for _ in open(trg_valid_path, 'r', encoding='utf-8'))
        
        if src_valid_lines != trg_valid_lines:
            logger.warning(f"验证集行数不匹配: 源={src_valid_lines}, 目标={trg_valid_lines}")
        
        logger.info(f"处理后的数据文件存在，行数: 训练集={src_train_lines}行, 验证集={src_valid_lines}行")
        
        # 随机抽取几行检查格式
        def sample_lines(file_path, count=5):
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if len(lines) <= count:
                return lines
            else:
                return random.sample(lines, count)
        
        src_train_samples = sample_lines(src_train_path)
        trg_train_samples = sample_lines(trg_train_path)
        
        logger.debug("训练集样本:")
        for i, (src, trg) in enumerate(zip(src_train_samples, trg_train_samples)):
            logger.debug(f"  样本 {i+1}:")
            logger.debug(f"    源: {src.strip()}")
            logger.debug(f"    目标: {trg.strip()}")
        
        return True
    except Exception as e:
        logger.error(f"验证处理后的数据文件时出错: {str(e)}")
        return False

def run_all_validations():
    """运行所有验证检查，返回总体结果"""
    logger.info("开始数据验证...")
    
    validations = [
        ("目录结构", validate_directory_structure),
        ("原始数据", validate_raw_data),
        ("SentencePiece模型", validate_sp_models),
        ("处理后的数据", validate_processed_data)
    ]
    
    results = []
    for name, validation_func in validations:
        logger.info(f"验证 {name}...")
        try:
            result = validation_func()
            status = "通过" if result else "失败"
            logger.info(f"验证 {name}: {status}")
            results.append(result)
        except Exception as e:
            logger.error(f"验证 {name} 时发生错误: {str(e)}")
            results.append(False)
    
    all_passed = all(results)
    logger.info(f"数据验证完成: {'全部通过' if all_passed else '存在问题'}")
    
    return all_passed

if __name__ == "__main__":
    run_all_validations() 