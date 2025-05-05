import math
import collections
import numpy as np
from typing import List, Dict, Tuple, Optional

def get_ngrams(segment: List[str], max_order: int) -> Dict[Tuple[str, ...], int]:
    """
    从一个序列中提取所有可能的n元组（n-grams）
    
    参数:
    - segment: 单词或标记序列
    - max_order: 最大n-gram长度
    
    返回:
    - n-grams计数字典
    """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts

def bleu_score(references: List[List[str]], hypotheses: List[List[str]], 
              max_order: int = 4, smooth: bool = False) -> Dict:
    """
    计算BLEU分数
    
    参数:
    - references: 参考翻译序列列表
    - hypotheses: 模型生成的翻译序列列表
    - max_order: 最大n-gram长度
    - smooth: 是否使用平滑算法
    
    返回:
    - BLEU分数及相关数据
    """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    hypothesis_length = 0
    
    for (references_i, hypothesis_i) in zip(references, hypotheses):
        reference_length += min(len(r) for r in references_i)
        hypothesis_length += len(hypothesis_i)
        
        # 合并当前示例的所有参考翻译的n-grams
        merged_ref_ngram_counts = collections.Counter()
        for reference in references_i:
            merged_ref_ngram_counts |= get_ngrams(reference, max_order)
            
        # 获取当前生成的翻译的n-grams
        hypothesis_ngram_counts = get_ngrams(hypothesis_i, max_order)
        
        # 统计共同的n-grams
        overlap = hypothesis_ngram_counts & merged_ref_ngram_counts
        
        # 计算每个阶数的n-gram匹配数和可能匹配数
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
            
        for ngram in hypothesis_ngram_counts:
            possible_matches_by_order[len(ngram) - 1] += hypothesis_ngram_counts[ngram]
    
    # 计算精确率
    precisions = [0] * max_order
    for i in range(max_order):
        if possible_matches_by_order[i] > 0:
            if smooth:
                precisions[i] = (matches_by_order[i] + 1.0) / (possible_matches_by_order[i] + 1.0)
            else:
                precisions[i] = matches_by_order[i] / possible_matches_by_order[i]
        else:
            precisions[i] = 0.0
    
    # 如果所有精确率为0，则BLEU分数为0
    if min(precisions) > 0:
        p_log_sum = sum(math.log(p) for p in precisions) / max_order
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0
    
    # 计算长度比率
    ratio = hypothesis_length / reference_length if reference_length > 0 else 0
    
    # 如果生成的翻译比参考翻译长，惩罚因子为1
    # 否则，惩罚因子为e^(1-1/ratio)
    if ratio > 1.0:
        bp = 1.0
    else:
        bp = math.exp(1 - 1.0 / ratio) if ratio > 0 else 0.0
    
    # 最终BLEU分数
    bleu = geo_mean * bp
    
    return {
        'bleu': bleu * 100,  # 转换为百分比
        'precisions': [p * 100 for p in precisions],  # 各阶n-gram精确率
        'brevity_penalty': bp,  # 简短惩罚因子
        'length_ratio': ratio,  # 长度比
        'reference_length': reference_length,  # 参考翻译总长度
        'hypothesis_length': hypothesis_length  # 生成翻译总长度
    }

def compute_bleu_for_corpus(reference_file: str, hypothesis_file: str, 
                           max_order: int = 4, smooth: bool = False) -> Dict:
    """
    计算整个语料库的BLEU分数
    
    参数:
    - reference_file: 参考翻译文件路径
    - hypothesis_file: 模型生成的翻译文件路径
    - max_order: 最大n-gram长度
    - smooth: 是否使用平滑算法
    
    返回:
    - BLEU分数及相关数据
    """
    # 读取参考翻译
    with open(reference_file, 'r', encoding='utf-8') as f:
        references = [line.strip().split() for line in f]
    
    # 读取模型生成的翻译
    with open(hypothesis_file, 'r', encoding='utf-8') as f:
        hypotheses = [line.strip().split() for line in f]
    
    # 确保两个文件的行数相同
    if len(references) != len(hypotheses):
        raise ValueError(f"参考翻译和生成翻译的行数不匹配: {len(references)} vs {len(hypotheses)}")
    
    # 将每个参考翻译包装在列表中以匹配bleu_score函数的格式要求
    wrapped_references = [[ref] for ref in references]
    
    return bleu_score(wrapped_references, hypotheses, max_order, smooth)

def compute_bleu_from_lists(references: List[str], hypotheses: List[str], 
                          max_order: int = 4, smooth: bool = False) -> Dict:
    """
    从字符串列表计算BLEU分数
    
    参数:
    - references: 参考翻译字符串列表
    - hypotheses: 模型生成的翻译字符串列表
    - max_order: 最大n-gram长度
    - smooth: 是否使用平滑算法
    
    返回:
    - BLEU分数及相关数据
    """
    # 将字符串分割为标记
    tokenized_refs = [ref.strip().split() for ref in references]
    tokenized_hyps = [hyp.strip().split() for hyp in hypotheses]
    
    # 确保两个列表的长度相同
    if len(tokenized_refs) != len(tokenized_hyps):
        raise ValueError(f"参考翻译和生成翻译的数量不匹配: {len(tokenized_refs)} vs {len(tokenized_hyps)}")
    
    # 将每个参考翻译包装在列表中以匹配bleu_score函数的格式要求
    wrapped_references = [[ref] for ref in tokenized_refs]
    
    return bleu_score(wrapped_references, tokenized_hyps, max_order, smooth)

def format_bleu_result(result: Dict) -> str:
    """
    格式化BLEU分数结果为可读字符串
    
    参数:
    - result: bleu_score函数返回的结果字典
    
    返回:
    - 格式化后的BLEU分数信息
    """
    precisions_str = '/'.join([f"{p:.1f}" for p in result['precisions']])
    return (f"BLEU = {result['bleu']:.2f}, "
            f"{precisions_str} "
            f"(BP = {result['brevity_penalty']:.3f}, "
            f"ratio = {result['length_ratio']:.3f}, "
            f"hyp_len = {result['hypothesis_length']}, "
            f"ref_len = {result['reference_length']})")

# 测试用例
if __name__ == "__main__":
    # 示例用法
    references = ["这是一个测试句子。", "这是另一个句子。"]
    hypotheses = ["这是测试句子。", "这是一个句子。"]
    
    result = compute_bleu_from_lists(references, hypotheses)
    print(format_bleu_result(result)) 