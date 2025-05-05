from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from config import *
from logger import logger
from data_validation import validate_directory_structure

import torch
import sentencepiece as spm
import numpy as np
import os
import pickle
import traceback
import datetime

# 添加缓存目录
CACHE_DIR = f"{DATA_DIR}/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# 全局变量，用于标记SentencePiece模型是否已加载
_SP_LOADED = False

# worker初始化函数，确保每个worker只加载一次SentencePiece模型
def worker_init_fn(worker_id):
    global _SP_LOADED
    if not _SP_LOADED:
        # 只在需要时加载模型
        try:
            src_sp = spm.SentencePieceProcessor()
            trg_sp = spm.SentencePieceProcessor()
            
            src_model_path = f"{SP_DIR}/{src_model_prefix}.model"
            trg_model_path = f"{SP_DIR}/{trg_model_prefix}.model"
            
            if os.path.exists(src_model_path):
                src_sp.Load(src_model_path)
            
            if os.path.exists(trg_model_path):
                trg_sp.Load(trg_model_path)
                
            _SP_LOADED = True
        except Exception as e:
            print(f"Worker {worker_id} 加载SentencePiece模型失败: {e}")

# 加载SentencePiece处理器
try:
    # SentencePiece processors are loaded here, ensure their model files are also UTF-8 if they contain non-ASCII chars
    src_sp = spm.SentencePieceProcessor()
    trg_sp = spm.SentencePieceProcessor()
    # Assuming .model files are handled correctly by sentencepiece library regarding encoding
    
    src_model_path = f"{SP_DIR}/{src_model_prefix}.model"
    trg_model_path = f"{SP_DIR}/{trg_model_prefix}.model"
    
    # 检查模型文件是否存在
    if not os.path.exists(src_model_path):
        logger.warning(f"源语言SentencePiece模型不存在: {src_model_path}")
    else:
        src_sp.Load(src_model_path)
        logger.info(f"加载源语言SentencePiece模型: {src_model_path}")
    
    if not os.path.exists(trg_model_path):
        logger.warning(f"目标语言SentencePiece模型不存在: {trg_model_path}")
    else:
        trg_sp.Load(trg_model_path)
        logger.info(f"加载目标语言SentencePiece模型: {trg_model_path}")
        
    # 标记模型已加载
    _SP_LOADED = True
except Exception as e:
    logger.error(f"加载SentencePiece模型时出错: {str(e)}")
    logger.error(traceback.format_exc())


def get_data_loader(file_name, shuffle=True):
    """
    加载数据并创建DataLoader
    
    参数:
    - file_name: 数据文件名
    - shuffle: 是否随机打乱数据
    
    返回:
    - DataLoader对象或None（如果出错）
    """
    try:
        logger.info(f"开始准备数据加载器: {file_name}")
        # 统一使用config模块导出的参数
        actual_batch_size = batch_size
        use_pin_memory = pin_memory
        actual_workers = num_workers
        
        # 获取数据限制配置
        data_limit = None
        if file_name == TRAIN_NAME:
            data_limit = TRAIN_CONFIG.get('data_limit', {}).get('train', None)
        elif file_name == VALID_NAME:
            data_limit = TRAIN_CONFIG.get('data_limit', {}).get('valid', None)
            
        # 添加数据处理版本号，当预处理逻辑改变时，能够重新生成缓存
        proc_version = "v2"  # 每次修改预处理逻辑时应递增
        
        # 根据是否有数据限制生成不同的缓存文件名
        limit_suffix = f"_limit{data_limit}" if data_limit is not None else ""
        cache_file = f"{CACHE_DIR}/{file_name}{limit_suffix}_proc{proc_version}.pkl"
        logger.debug(f"数据限制: {data_limit if data_limit else '无限制'}, 缓存文件: {cache_file}")
        
        # 使用mmap进行快速缓存加载
        def load_cache_with_mmap(cache_file):
            try:
                import mmap
                with open(cache_file, 'rb') as f:
                    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                    cached_data = pickle.load(mm)
                    mm.close()
                    return cached_data
            except:
                # 如果mmap失败，回退到常规方法
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)

        # 检查是否存在缓存
        if os.path.exists(cache_file):
            logger.info(f"找到缓存数据 {cache_file}，直接加载...")
            try:
                # 尝试使用更快的加载方式
                cached_data = load_cache_with_mmap(cache_file)
                
                # 检查缓存数据的完整性
                required_keys = ['src_list', 'input_trg_list', 'output_trg_list', 'metadata']
                if not all(key in cached_data for key in required_keys):
                    logger.warning("缓存数据格式不完整，将重新生成")
                    raise ValueError("缓存数据不完整")
                
                # 获取缓存的元数据
                metadata = cached_data.get('metadata', {})
                cache_seq_len = metadata.get('seq_len')
                
                # 检查关键参数是否变化
                if cache_seq_len != seq_len:
                    logger.warning(f"序列长度已变更: 缓存={cache_seq_len}, 当前={seq_len}，将重新生成缓存")
                    raise ValueError("序列长度不匹配")
                
                src_list = cached_data['src_list']
                input_trg_list = cached_data['input_trg_list']
                output_trg_list = cached_data['output_trg_list']
                
                logger.info(f"缓存加载完成! src数据形状: {np.shape(src_list)}, input_trg形状: {np.shape(input_trg_list)}, output_trg形状: {np.shape(output_trg_list)}")
            except Exception as e:
                logger.error(f"加载缓存文件失败: {str(e)}")
                logger.info(f"将重新处理数据并创建新的缓存...")
                try:
                    os.remove(cache_file)  # 删除可能损坏的缓存文件
                    logger.debug(f"已删除无效的缓存文件: {cache_file}")
                except Exception as e2:
                    logger.warning(f"删除缓存文件失败: {str(e2)}，将尝试覆盖它")
                # 继续执行下面的代码，重新处理数据
        else:
            logger.info(f"未找到缓存，开始处理 {file_name}...")
        
        # 如果没有缓存数据，则从文件读取并处理
        if 'src_list' not in locals():
            # 明确指定编码为 'utf-8'
            src_file_path = f"{SRC_DIR}/{file_name}"
            trg_file_path = f"{TRG_DIR}/{file_name}"
            
            # 检查文件是否存在
            if not os.path.exists(src_file_path):
                logger.error(f"源语言文件不存在: {src_file_path}")
                raise FileNotFoundError(f"源语言文件不存在: {src_file_path}")
            
            if not os.path.exists(trg_file_path):
                logger.error(f"目标语言文件不存在: {trg_file_path}")
                raise FileNotFoundError(f"目标语言文件不存在: {trg_file_path}")
                
            try:
                logger.info(f"开始读取源语言文件: {src_file_path}")
                with open(src_file_path, 'r', encoding='utf-8') as f:
                    src_text_list = f.readlines()
            except Exception as e:
                logger.error(f"读取源语言文件失败: {str(e)}")
                raise

            try:
                logger.info(f"开始读取目标语言文件: {trg_file_path}")
                with open(trg_file_path, 'r', encoding='utf-8') as f:
                    trg_text_list = f.readlines()
            except Exception as e:
                logger.error(f"读取目标语言文件失败: {str(e)}")
                raise
                
            # 验证源语言和目标语言数据行数是否匹配
            if len(src_text_list) != len(trg_text_list):
                logger.warning(f"源语言和目标语言数据行数不匹配: 源={len(src_text_list)}, 目标={len(trg_text_list)}")
                # 取较小值作为共同长度
                min_len = min(len(src_text_list), len(trg_text_list))
                src_text_list = src_text_list[:min_len]
                trg_text_list = trg_text_list[:min_len]
                logger.info(f"截断到共同长度: {min_len}行")
            
            # 如果设置了数据限制，则只使用前data_limit行数据
            if data_limit is not None:
                if len(src_text_list) > data_limit:
                    logger.info(f"应用数据限制: 从 {len(src_text_list)} 行减少到 {data_limit} 行")
                    src_text_list = src_text_list[:data_limit]
                    trg_text_list = trg_text_list[:data_limit]

            logger.info("开始处理源语言数据: 分词和填充...")
            src_list = process_src(src_text_list) # (sample_num, L)
            logger.info(f"源语言数据处理完成，形状: {np.shape(src_list)}")

            logger.info("开始处理目标语言数据: 分词和填充...")
            input_trg_list, output_trg_list = process_trg(trg_text_list) # (sample_num, L)
            logger.info(f"目标语言数据处理完成，输入形状: {np.shape(input_trg_list)}, 输出形状: {np.shape(output_trg_list)}")
            
            # 创建元数据用于缓存校验
            metadata = {
                'created_at': datetime.datetime.now().isoformat(),
                'seq_len': seq_len,
                'proc_version': proc_version,
                'data_limit': data_limit,
                'pad_id': pad_id,
                'eos_id': eos_id,
                'sos_id': sos_id,
            }
            
            # 保存缓存
            try:
                logger.info(f"保存数据到缓存: {cache_file}")
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'src_list': src_list,
                        'input_trg_list': input_trg_list,
                        'output_trg_list': output_trg_list,
                        'metadata': metadata  # 添加元数据
                    }, f)
                logger.info("缓存保存完成!")
            except Exception as e:
                logger.warning(f"保存缓存文件失败: {str(e)}, 将继续执行但不保存缓存")

        # 创建数据集，如果to_device不为None，则将数据移到指定设备
        try:
            dataset = CustomDataset(src_list, input_trg_list, output_trg_list)
            logger.info(f"创建DataLoader: 批量大小={actual_batch_size}, 工作线程数={actual_workers}, pin_memory={use_pin_memory}")
            
            # 创建预取数据加载器，提高数据加载效率
            from torch.utils.data import DataLoader
            
            # 使用持久化工作器以减少worker初始化开销
            persistent_workers = actual_workers > 0
            
            dataloader = DataLoader(
                dataset, 
                batch_size=actual_batch_size, 
                shuffle=shuffle,
                pin_memory=use_pin_memory,
                num_workers=actual_workers,
                worker_init_fn=worker_init_fn if actual_workers > 0 else None,
                persistent_workers=persistent_workers,
                prefetch_factor=2 if actual_workers > 0 else None,  # 预取因子
            )
            logger.info(f"DataLoader创建成功，每批数据形状: [{actual_batch_size}, {seq_len}]")
            return dataloader
        except Exception as e:
            logger.error(f"创建DataLoader失败: {str(e)}")
            raise
            
    except Exception as e:
        logger.error(f"准备数据加载器时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return None


def pad_or_truncate(tokenized_text):
    """
    填充或截断序列至指定长度
    
    参数:
    - tokenized_text: 待处理的token列表
    
    返回:
    - 处理后的等长token列表
    """
    original_len = len(tokenized_text)
    
    if original_len < seq_len:
        left = seq_len - original_len
        padding = [pad_id] * left
        tokenized_text += padding
    else:
        # 对长句保留开头和结尾部分，提高语义完整性
        if original_len > seq_len + 10:  # 只对明显超长的句子应用智能截断
            # 保留前2/3和后1/3的内容
            keep_front = int(seq_len * 0.7)
            keep_end = seq_len - keep_front
            tokenized_text = tokenized_text[:keep_front] + tokenized_text[-keep_end:]
        else:
            # 对稍微超长的简单截断
            tokenized_text = tokenized_text[:seq_len]
    
    return tokenized_text


def clean_text(text, default_text="."):
    """
    清洗文本，处理常见问题
    
    参数:
    - text: 待清洗的文本
    - default_text: 文本为空时的默认值
    
    返回:
    - 清洗后的文本
    """
    # 基本清洗
    text = text.strip()
    # 规范化空白字符
    text = ' '.join(text.split())
    # 移除控制字符和不可打印字符
    text = ''.join(c for c in text if c.isprintable())
    
    # 对空文本的处理
    if not text:
        return default_text
        
    return text


def process_src(text_list):
    """
    处理源语言文本列表
    
    参数:
    - text_list: 源语言文本列表
    
    返回:
    - 处理后的token列表列表
    """
    try:
        tokenized_list = []
        total = len(text_list)
        
        # 确认分词器已加载
        if not hasattr(src_sp, 'EncodeAsIds'):
            logger.error("源语言分词器未正确加载，无法进行分词")
            raise RuntimeError("源语言分词器未正确加载")
        
        for i, text in enumerate(tqdm(text_list, desc="处理源语言")):
            try:
                # 使用改进的清洗函数
                text = clean_text(text)
                
                # SentencePiece分词
                tokenized = src_sp.EncodeAsIds(text)
                
                # 记录长句子
                if len(tokenized) > seq_len:
                    logger.debug(f"发现长句子 ({len(tokenized)} > {seq_len})，将进行智能截断，行号: {i+1}")
                
                tokenized_list.append(pad_or_truncate(tokenized))
                
                # 定期报告进度
                if (i+1) % 50000 == 0 or i+1 == total:
                    logger.info(f"源语言处理进度: {i+1}/{total} 行 ({(i+1)/total*100:.1f}%)")
            except Exception as e:
                logger.error(f"处理源语言文本时出错，行号 {i+1}: {str(e)}")
                # 使用空序列替代，但添加unk_id避免纯填充
                empty_seq = [unk_id] + [pad_id] * (seq_len - 1)
                tokenized_list.append(empty_seq)
        
        return tokenized_list
    
    except Exception as e:
        logger.error(f"源语言处理过程出错: {str(e)}")
        logger.error(traceback.format_exc())
        return []


def process_trg(text_list):
    """
    处理目标语言文本列表
    
    参数:
    - text_list: 目标语言文本列表
    
    返回:
    - (input_tokenized_list, output_tokenized_list) 输入和输出token列表列表的元组
    """
    try:
        input_tokenized_list = []
        output_tokenized_list = []
        total = len(text_list)
        
        # 确认分词器已加载
        if not hasattr(trg_sp, 'EncodeAsIds'):
            logger.error("目标语言分词器未正确加载，无法进行分词")
            raise RuntimeError("目标语言分词器未正确加载")
        
        for i, text in enumerate(tqdm(text_list, desc="处理目标语言")):
            try:
                # 使用改进的清洗函数
                text = clean_text(text)
                
                # SentencePiece分词
                tokenized = trg_sp.EncodeAsIds(text)
                
                # 记录长句子
                if len(tokenized) >= seq_len - 1:  # 减1是因为还要加上sos或eos
                    logger.debug(f"发现长句子 ({len(tokenized)} > {seq_len-1})，将进行智能截断，行号: {i+1}")
                
                # 添加开始和结束标记
                trg_input = [sos_id] + tokenized
                trg_output = tokenized + [eos_id]
                
                input_tokenized_list.append(pad_or_truncate(trg_input))
                output_tokenized_list.append(pad_or_truncate(trg_output))
                
                # 定期报告进度
                if (i+1) % 50000 == 0 or i+1 == total:
                    logger.info(f"目标语言处理进度: {i+1}/{total} 行 ({(i+1)/total*100:.1f}%)")
            except Exception as e:
                logger.error(f"处理目标语言文本时出错，行号 {i+1}: {str(e)}")
                # 使用具有基本结构的空序列替代
                empty_input = [sos_id] + [pad_id] * (seq_len - 1)
                empty_output = [unk_id] + [eos_id] + [pad_id] * (seq_len - 2)
                input_tokenized_list.append(empty_input)
                output_tokenized_list.append(empty_output)
        
        return input_tokenized_list, output_tokenized_list
    
    except Exception as e:
        logger.error(f"目标语言处理过程出错: {str(e)}")
        logger.error(traceback.format_exc())
        return [], []


class CustomDataset(Dataset):
    """
    自定义数据集类，用于批量加载翻译数据
    """
    def __init__(self, src_list, input_trg_list, output_trg_list):
        """
        初始化数据集
        
        参数:
        - src_list: 源语言输入序列列表
        - input_trg_list: 目标语言输入序列列表（已添加sos和eos）
        - output_trg_list: 目标语言输出序列列表（已添加sos和eos）
        """
        super().__init__()
        try:
            # 验证数据对齐
            if len(src_list) != len(input_trg_list) or len(src_list) != len(output_trg_list):
                error_msg = f"源语言和目标语言数据长度不匹配: 源={len(src_list)}, 输入目标={len(input_trg_list)}, 输出目标={len(output_trg_list)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # 验证数据样本
            self.valid_indices = []
            invalid_count = 0
            max_check = min(1000, len(src_list))  # 检查更多样本但不超过1000个
            
            for i in range(max_check):
                is_valid = True
                
                # 检查序列长度
                if len(src_list[i]) != seq_len or len(input_trg_list[i]) != seq_len or len(output_trg_list[i]) != seq_len:
                    logger.warning(f"序列长度不匹配，索引 {i}: 源={len(src_list[i])}, 输入目标={len(input_trg_list[i])}, 输出目标={len(output_trg_list[i])}, 期望={seq_len}")
                    is_valid = False
                
                # 检查特殊标记
                # 检查目标语言输入是否以SOS开始
                if input_trg_list[i][0] != sos_id:
                    logger.warning(f"目标语言输入未以SOS开始，索引 {i}")
                    is_valid = False
                
                # 跳过无效样本
                if is_valid:
                    self.valid_indices.append(i)
                else:
                    invalid_count += 1
            
            # 假设剩余未检查的数据都是有效的
            if max_check < len(src_list):
                self.valid_indices.extend(range(max_check, len(src_list)))
            
            if invalid_count > 0:
                logger.warning(f"发现 {invalid_count}/{max_check} 个无效样本，将在训练中跳过")
            
            # 转换为张量
            logger.debug("转换数据为PyTorch张量...")
            self.src_data = torch.LongTensor(src_list)
            self.input_trg_data = torch.LongTensor(input_trg_list)
            self.output_trg_data = torch.LongTensor(output_trg_list)
            
            # 统计其他指标
            has_pad = torch.sum(self.src_data == pad_id).item() > 0
            has_eos = torch.sum(self.output_trg_data == eos_id).item() > 0
            
            logger.info(f"数据集创建完成，共 {len(src_list)} 条数据，有效样本 {len(self.valid_indices)} 条")
            logger.debug(f"数据集统计: 包含填充={has_pad}, 包含EOS标记={has_eos}")
        
        except Exception as e:
            logger.error(f"创建数据集时出错: {str(e)}")
            logger.error(traceback.format_exc())
            # 确保至少有一些数据可用，即使出错
            self.src_data = torch.LongTensor(src_list) if isinstance(src_list, list) else src_list
            self.input_trg_data = torch.LongTensor(input_trg_list) if isinstance(input_trg_list, list) else input_trg_list
            self.output_trg_data = torch.LongTensor(output_trg_list) if isinstance(output_trg_list, list) else output_trg_list
            self.valid_indices = list(range(len(self.src_data)))
            raise

    def make_mask(self):
        """
        创建源语言和目标语言的注意力掩码
        
        返回:
        - (e_mask, d_mask) 注意力掩码元组
        """
        try:
            # 编码器掩码：将填充部分标记为False
            e_mask = (self.src_data != pad_id).unsqueeze(1) # (num_samples, 1, L)
            
            # 解码器掩码：将填充部分和未来位置标记为False
            d_mask = (self.input_trg_data != pad_id).unsqueeze(1) # (num_samples, 1, L)

            # 创建三角掩码，确保解码器不能看到未来位置
            nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool) # (1, L, L)
            # 确保掩码在正确的设备上
            nopeak_mask = torch.tril(nopeak_mask).to(self.input_trg_data.device) # (1, L, L) to triangular shape
            d_mask = d_mask & nopeak_mask # (num_samples, L, L) padding false

            return e_mask, d_mask
            
        except Exception as e:
            logger.error(f"创建注意力掩码时出错: {str(e)}")
            raise

    def __getitem__(self, idx):
        """
        获取数据集中的一个样本
        
        参数:
        - idx: 样本索引
        
        返回:
        - (src, input_trg, output_trg) 数据元组
        """
        try:
            # 使用有效索引访问数据
            real_idx = self.valid_indices[idx] if idx < len(self.valid_indices) else idx
            
            # 尝试获取样本，添加基本错误检查
            src = self.src_data[real_idx]
            input_trg = self.input_trg_data[real_idx]
            output_trg = self.output_trg_data[real_idx]
            
            # 简单验证样本格式是否正确
            if input_trg[0] != sos_id:
                logger.warning(f"获取到的样本不以SOS开始，索引 {real_idx}，将修复")
                input_trg[0] = sos_id
                
            return src, input_trg, output_trg
            
        except Exception as e:
            logger.error(f"获取数据样本时出错，索引: {idx}，映射索引: {real_idx if 'real_idx' in locals() else '未知'}, 错误: {str(e)}")
            # 返回一个空样本，避免训练中断
            device = self.src_data.device
            empty_src = torch.full((seq_len,), pad_id, dtype=torch.long, device=device)
            empty_input_trg = torch.full((seq_len,), pad_id, dtype=torch.long, device=device)
            empty_output_trg = torch.full((seq_len,), pad_id, dtype=torch.long, device=device)
            # 设置起始和结束标记
            empty_src[0] = unk_id
            empty_input_trg[0] = sos_id
            empty_output_trg[0] = unk_id
            empty_output_trg[1] = eos_id
            return empty_src, empty_input_trg, empty_output_trg

    def __len__(self):
        """
        获取数据集的样本数量
        
        返回:
        - 样本数量
        """
        # 使用有效样本的数量
        return len(self.valid_indices)

def clear_cache(file_name=None):
    """
    清除数据缓存
    
    参数:
    - file_name: 指定要清除的文件缓存，如果为None则清除所有缓存
    
    返回:
    - bool: 操作是否成功
    """
    try:
        if not os.path.exists(CACHE_DIR):
            logger.info("缓存目录不存在，无需清除")
            return True
            
        if file_name is not None:
            # 清除特定文件的所有缓存（包括各种限制条件下的缓存）
            cache_pattern = f"{file_name}"
            cache_files = [f for f in os.listdir(CACHE_DIR) if f.startswith(cache_pattern)]
            
            if not cache_files:
                logger.info(f"未找到{file_name}相关的缓存文件")
                return True
                
            for cf in cache_files:
                try:
                    cache_path = os.path.join(CACHE_DIR, cf)
                    os.remove(cache_path)
                    logger.debug(f"已删除缓存文件: {cache_path}")
                except Exception as e:
                    logger.warning(f"删除缓存文件失败: {cache_path}, 错误: {str(e)}")
            
            logger.info(f"已清除{len(cache_files)}个与{file_name}相关的缓存文件")
        else:
            # 清除所有缓存
            cache_files = os.listdir(CACHE_DIR)
            if not cache_files:
                logger.info("缓存目录为空，无需清除")
                return True
                
            for cf in cache_files:
                try:
                    cache_path = os.path.join(CACHE_DIR, cf)
                    os.remove(cache_path)
                    logger.debug(f"已删除缓存文件: {cache_path}")
                except Exception as e:
                    logger.warning(f"删除缓存文件失败: {cache_path}, 错误: {str(e)}")
            
            logger.info(f"已清除所有{len(cache_files)}个缓存文件")
        
        return True
    
    except Exception as e:
        logger.error(f"清除缓存时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return False


# 添加模块初始化日志
logger.info(f"custom_data模块已加载，缓存目录: {CACHE_DIR}")
