from config import *
from tqdm import tqdm
from logger import logger  # 导入日志记录器
from data_validation import validate_directory_structure

import os
import sentencepiece as spm
import traceback

# 从配置中获取训练集比例
train_frac = TRAIN_CONFIG.get('sentencepiece', {}).get('train_frac', 0.8)

def train_sp(is_src=True):
    """
    训练SentencePiece模型
    
    参数:
    - is_src: 是否为源语言模型，True为源语言，False为目标语言
    
    返回:
    - bool: 训练是否成功
    """
    try:
        template = "--input={} --pad_id={} --bos_id={} --eos_id={} --unk_id={} --model_prefix={} --vocab_size={} --character_coverage={} --model_type={}"

        if is_src:
            this_input_file = SRC_RAW_DATA_NAME
            this_model_prefix = f"{SP_DIR}/{src_model_prefix}"
            lang_desc = "源语言"
        else:
            this_input_file = TRG_RAW_DATA_NAME
            this_model_prefix = f"{SP_DIR}/{trg_model_prefix}"
            lang_desc = "目标语言"
        
        logger.info(f"开始训练{lang_desc} SentencePiece模型...")
        
        # 检查输入文件是否存在
        if not os.path.exists(this_input_file):
            logger.error(f"{lang_desc}输入文件不存在: {this_input_file}")
            return False

        # 应用数据限制（如果有）
        raw_data_limit = TRAIN_CONFIG.get('sentencepiece', {}).get('raw_data_limit')
        limited_file = None
        
        if raw_data_limit is not None:
            try:
                # 创建一个临时的限制数据文件
                limited_file = f"{this_input_file}_limited"
                with open(this_input_file, 'r', encoding='utf-8') as f_in:
                    lines = f_in.readlines()
                
                if len(lines) > raw_data_limit:
                    logger.info(f"应用原始数据限制: 从 {len(lines)} 行减少到 {raw_data_limit} 行")
                    lines = lines[:raw_data_limit]
                
                with open(limited_file, 'w', encoding='utf-8') as f_out:
                    for line in lines:
                        f_out.write(line)
                
                # 使用限制后的文件进行SentencePiece训练
                this_input_file = limited_file
                logger.info(f"已创建限制数据文件: {limited_file}")
            except Exception as e:
                logger.error(f"创建限制数据文件时出错: {str(e)}")
                # 清理临时文件
                if limited_file and os.path.exists(limited_file):
                    os.remove(limited_file)
                return False

        config = template.format(this_input_file,
                                pad_id,
                                sos_id,
                                eos_id,
                                unk_id,
                                this_model_prefix,
                                sp_vocab_size,
                                character_coverage,
                                model_type)

        logger.debug(f"SentencePiece训练配置: {config}")

        # 确保输出目录存在
        if not os.path.isdir(SP_DIR):
            os.makedirs(SP_DIR, exist_ok=True)
            logger.info(f"已创建SentencePiece模型目录: {SP_DIR}")

        # 训练模型
        logger.info(f"开始训练{lang_desc} SentencePiece模型...")
        spm.SentencePieceTrainer.Train(config)
        
        # 验证模型是否已创建
        model_path = f"{this_model_prefix}.model"
        vocab_path = f"{this_model_prefix}.vocab"
        
        if not os.path.exists(model_path) or not os.path.exists(vocab_path):
            logger.error(f"{lang_desc} SentencePiece模型或词汇表未创建: {model_path}, {vocab_path}")
            return False
        
        logger.info(f"{lang_desc} SentencePiece模型训练完成: {model_path}")
        
        # 如果创建了临时文件，删除它
        if raw_data_limit is not None and limited_file and os.path.exists(limited_file):
            os.remove(limited_file)
            logger.info(f"已删除临时文件 {limited_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"训练{lang_desc} SentencePiece模型时出错: {str(e)}")
        logger.error(traceback.format_exc())
        
        # 清理临时文件
        if 'limited_file' in locals() and limited_file and os.path.exists(limited_file):
            os.remove(limited_file)
        
        return False


def split_data_aligned(src_raw_data_name, trg_raw_data_name, src_data_dir, trg_data_dir):
    """
    对齐分割源/目标数据为训练集和验证集，过滤空行
    
    参数:
    - src_raw_data_name: 源语言原始数据文件路径
    - trg_raw_data_name: 目标语言原始数据文件路径
    - src_data_dir: 源语言处理后数据保存目录
    - trg_data_dir: 目标语言处理后数据保存目录
    
    返回:
    - bool: 操作是否成功
    """
    try:
        logger.info(f"开始对齐分割数据: {src_raw_data_name} & {trg_raw_data_name}")
        
        # 检查原始数据文件是否存在
        if not os.path.exists(src_raw_data_name):
            logger.error(f"源语言原始数据文件不存在: {src_raw_data_name}")
            return False
        if not os.path.exists(trg_raw_data_name):
            logger.error(f"目标语言原始数据文件不存在: {trg_raw_data_name}")
            return False
            
        # 读取全部行
        with open(src_raw_data_name, 'r', encoding='utf-8') as f:
            src_lines = f.readlines()
        with open(trg_raw_data_name, 'r', encoding='utf-8') as f:
            trg_lines = f.readlines()
            
        # 过滤空行（两边都要非空才保留）
        filtered_src = []
        filtered_trg = []
        empty_count = 0
        for s, t in zip(src_lines, trg_lines):
            s_strip = s.strip()
            t_strip = t.strip()
            if s_strip and t_strip:
                filtered_src.append(s_strip)
                filtered_trg.append(t_strip)
            else:
                empty_count += 1
                
        min_len = min(len(filtered_src), len(filtered_trg))
        filtered_src = filtered_src[:min_len]
        filtered_trg = filtered_trg[:min_len]
        logger.info(f"过滤空行后有效样本数: {min_len}，被过滤空行对数: {empty_count}")
        
        # 应用数据限制
        raw_data_limit = TRAIN_CONFIG.get('sentencepiece', {}).get('raw_data_limit')
        if raw_data_limit is not None and min_len > raw_data_limit:
            logger.info(f"应用原始数据限制: 从 {min_len} 行减少到 {raw_data_limit} 行")
            filtered_src = filtered_src[:raw_data_limit]
            filtered_trg = filtered_trg[:raw_data_limit]
            min_len = raw_data_limit
            
        # 分割为训练集和验证集
        train_num = int(train_frac * min_len)
        src_train = filtered_src[:train_num]
        src_valid = filtered_src[train_num:]
        trg_train = filtered_trg[:train_num]
        trg_valid = filtered_trg[train_num:]
        
        # 确保目标目录存在
        os.makedirs(src_data_dir, exist_ok=True)
        os.makedirs(trg_data_dir, exist_ok=True)
        logger.info(f"训练集数据量: {len(src_train)} 行，验证集数据量: {len(src_valid)} 行")
        
        # 保存源语言训练和验证数据
        logger.info(f"保存源语言(英语)数据到: {src_data_dir}")
        with open(f"{src_data_dir}/{TRAIN_NAME}", 'w', encoding='utf-8') as f:
            for line in tqdm(src_train, desc="保存源语言训练集"):
                f.write(line + '\n')
                
        with open(f"{src_data_dir}/{VALID_NAME}", 'w', encoding='utf-8') as f:
            for line in tqdm(src_valid, desc="保存源语言验证集"):
                f.write(line + '\n')
        
        # 保存目标语言训练和验证数据
        logger.info(f"保存目标语言(法语)数据到: {trg_data_dir}")
        with open(f"{trg_data_dir}/{TRAIN_NAME}", 'w', encoding='utf-8') as f:
            for line in tqdm(trg_train, desc="保存目标语言训练集"):
                f.write(line + '\n')
                
        with open(f"{trg_data_dir}/{VALID_NAME}", 'w', encoding='utf-8') as f:
            for line in tqdm(trg_valid, desc="保存目标语言验证集"):
                f.write(line + '\n')
        
        logger.info(f"源语言数据保存至: {src_data_dir}")
        logger.info(f"目标语言数据保存至: {trg_data_dir}")
        logger.info("数据分割完成")
        return True
        
    except Exception as e:
        logger.error(f"对齐分割数据时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return False


if __name__=='__main__':
    try:
        logger.info("========== 开始数据准备流程 ==========")
        
        # 确保目录结构存在
        validate_directory_structure()
        
        # 检查是否需要重新生成SentencePiece模型
        regenerate = TRAIN_CONFIG.get('sentencepiece', {}).get('regenerate', False)
        
        # 如果需要重新生成，清除之前的缓存数据
        if regenerate:
            try:
                # 导入清除缓存函数
                from custom_data import clear_cache
                logger.info("清除缓存数据...")
                clear_cache() # 清除所有缓存
                logger.info("缓存数据已清除")
            except Exception as e:
                logger.warning(f"清除缓存失败: {str(e)}")
        
        # 检查SentencePiece模型是否已存在
        src_model_exists = os.path.exists(f"{SP_DIR}/{src_model_prefix}.model") and os.path.exists(f"{SP_DIR}/{src_model_prefix}.vocab")
        trg_model_exists = os.path.exists(f"{SP_DIR}/{trg_model_prefix}.model") and os.path.exists(f"{SP_DIR}/{trg_model_prefix}.vocab")
        
        # 如果模型不存在或需要重新生成
        if not src_model_exists or not trg_model_exists or regenerate:
            logger.info("正在生成SentencePiece模型...")
            
            src_success = train_sp(is_src=True)
            trg_success = train_sp(is_src=False)
            
            if not src_success or not trg_success:
                logger.error("SentencePiece模型生成失败，请检查错误日志")
                exit(1)
        else:
            logger.info("SentencePiece模型已存在，跳过生成步骤。如需重新生成，请在配置中设置'regenerate'为True。")
        
        # 检查分割后的数据文件是否存在
        src_train_exists = os.path.exists(f"{SRC_DIR}/{TRAIN_NAME}")
        src_valid_exists = os.path.exists(f"{SRC_DIR}/{VALID_NAME}")
        trg_train_exists = os.path.exists(f"{TRG_DIR}/{TRAIN_NAME}")
        trg_valid_exists = os.path.exists(f"{TRG_DIR}/{VALID_NAME}")
        
        # 如果数据文件不存在或需要重新生成
        if not src_train_exists or not src_valid_exists or not trg_train_exists or not trg_valid_exists or regenerate:
            logger.info("正在分割数据...")
            
            split_success = split_data_aligned(SRC_RAW_DATA_NAME, TRG_RAW_DATA_NAME, SRC_DIR, TRG_DIR)
            
            if not split_success:
                logger.error("数据分割失败，请检查错误日志")
                exit(1)
        else:
            logger.info("分割后的数据文件已存在，跳过分割步骤。如需重新分割，请在配置中设置'regenerate'为True。")
        
        logger.info("========== 数据准备流程结束 ==========")
        
    except Exception as e:
        logger.error(f"数据准备过程中发生未处理的异常: {str(e)}")
        logger.error(traceback.format_exc())
        exit(1)
