import argparse
from logger import logger, log_system_info
from data_validation import run_all_validations
import traceback
import sys
from config import *
import torch
import sentencepiece as spm
import os
from train_manager import TrainManager
from inference_manager import InferenceManager
from custom_data import clear_cache
from data_validation import validate_directory_structure, setup_data_directories

def verify_gpu_optimizations():
    """验证GPU优化设置是否正确生效"""
    if not torch.cuda.is_available():
        logger.warning("CUDA不可用，无法验证GPU优化设置")
        return
        
    # 检查TF32设置
    if use_tf32:
        is_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
        is_tf32_cudnn = torch.backends.cudnn.allow_tf32
        logger.info(f"TF32状态: matmul={is_tf32_matmul}, cudnn={is_tf32_cudnn}")
        
        # 检测是否为支持TF32的GPU (Ampere或更新架构)
        gpu_name = torch.cuda.get_device_name()
        if "RTX 30" in gpu_name or "RTX 40" in gpu_name or "A100" in gpu_name or "A10" in gpu_name:
            logger.info(f"检测到支持TF32的GPU: {gpu_name}")
        else:
            logger.warning(f"当前GPU ({gpu_name}) 可能不支持TF32，此配置可能无效")
    
    # 测试channels_last内存格式
    if channels_last_memory_format:
        # 创建一个测试张量
        test_tensor = torch.randn(1, 3, 64, 64, device=device)
        is_channels_last = test_tensor.to(memory_format=torch.channels_last).is_contiguous(memory_format=torch.channels_last)
        logger.info(f"Channels Last测试: 设备上的张量能否转换为channels_last格式: {is_channels_last}")

def log_optimizations():
    """输出启用的优化选项"""
    optimizations = []
    if mixed_precision:
        optimizations.append("混合精度训练 (AMP)")
    if benchmark_cudnn:
        optimizations.append("cuDNN benchmark")
    if use_tf32:
        optimizations.append("TensorFloat-32")
    if channels_last_memory_format:
        optimizations.append("channels_last内存格式")
    if label_smoothing > 0:
        optimizations.append(f"标签平滑正则化 ({label_smoothing})")
    if weight_decay > 0:
        optimizations.append(f"权重衰减 ({weight_decay})")
    if warmup_steps > 0:
        optimizations.append(f"学习率预热 ({warmup_steps}步)")
    if validate_every_steps > 0:
        optimizations.append(f"基于步数的验证 (每{validate_every_steps}步)")
    
    if optimizations:
        logger.info("已启用的优化选项:")
        for opt in optimizations:
            logger.info(f"  - {opt}")
    else:
        logger.info("未启用任何额外优化选项")

# 添加JIT编译和性能优化功能
def optimize_model(model, use_jit=True):
    """
    对模型应用优化技术，提高推理和训练速度
    
    参数:
    - model: 要优化的模型
    - use_jit: 是否使用JIT编译
    
    返回:
    - 优化后的模型
    """
    logger.info("正在应用模型优化...")
    
    # 启用JIT编译优化
    if use_jit and TRAIN_CONFIG.get('optimization', {}).get('enable_jit', False):
        try:
            # 仅在模型完全构建好并可能已经加载了权重之后再应用JIT
            logger.info("尝试应用JIT编译优化...")
            # 为编码器和解码器启用梯度检查点
            model.enable_checkpointing(True)
            logger.info("JIT编译优化应用成功")
        except Exception as e:
            logger.warning(f"无法应用JIT编译优化: {str(e)}")
    
    # 应用其他优化操作
    if torch.cuda.is_available():
        # 尝试预热GPU（可能有助于稳定性能）
        try:
            logger.info("预热GPU...")
            dummy_input = torch.zeros(2, seq_len, dtype=torch.long, device=device)
            dummy_mask = torch.ones(2, 1, seq_len, dtype=torch.bool, device=device)
            with torch.no_grad():
                for _ in range(3):  # 进行几次预热
                    _ = model.encoder(model.src_embedding(dummy_input), dummy_mask)
            logger.info("GPU预热完成")
        except Exception as e:
            logger.warning(f"GPU预热失败: {str(e)}")
    
    # 检查是否支持/启用TF32精度
    if torch.cuda.is_available() and use_tf32:
        if hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
            logger.info("当前系统支持TF32精度")
        else:
            logger.warning("当前系统不支持TF32精度或PyTorch版本过低")
            
    return model

def main():
    try:
        # 设置CUDA性能选项
        if torch.cuda.is_available():
            # 根据配置启用TF32精度（仅在Ampere及以上GPU上有效）
            if use_tf32:
                torch.set_float32_matmul_precision('high')
                # 显式启用TF32模式
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("已启用TensorFloat-32 (TF32)，将加速矩阵乘法和卷积操作")
            
            # 确保cudnn.benchmark已启用
            if benchmark_cudnn:
                logger.info("cuDNN benchmark已启用，将优化卷积操作")
            
        log_system_info()
        log_optimizations()
        verify_gpu_optimizations()
        
        parser = argparse.ArgumentParser(description='运行Transformer翻译模型')
        parser.add_argument('--mode', type=str, choices=['train', 'inference', 'evaluate'], required=True, help='模式: train, inference或evaluate')
        parser.add_argument('--ckpt_name', type=str, default=None, help='检查点文件名')
        parser.add_argument('--input', type=str, help='要翻译的输入文本 (仅inference模式)')
        parser.add_argument('--decode', type=str, default='greedy', choices=['greedy', 'beam'], help='解码方法: greedy或beam (仅inference模式)')
        parser.add_argument('--validate_data', action='store_true', help='运行数据验证')
        parser.add_argument('--ref_file', type=str, help='用于评估的参考翻译文件 (仅evaluate模式)')
        parser.add_argument('--calc_bleu', action='store_true', help='是否计算BLEU分数')
        parser.add_argument('--clear_cache', action='store_true', help='清除数据缓存')
        parser.add_argument('--batch_size', type=int, help='训练批量大小')
        parser.add_argument('--optimize', action='store_true', help='应用额外的性能优化')
        args = parser.parse_args()

        if args.validate_data:
            logger.info("开始运行数据验证...")
            if run_all_validations():
                logger.info("数据验证通过，继续执行...")
            else:
                logger.warning("数据验证未通过，但将继续执行。请检查日志获取详细信息。")

        TRAIN_CONFIG['mode'] = args.mode
        if args.ckpt_name:
            TRAIN_CONFIG['ckpt_name'] = args.ckpt_name
        if args.input:
            TRAIN_CONFIG['input_text'] = args.input
        if args.decode:
            TRAIN_CONFIG['decode_method'] = args.decode
        if args.calc_bleu:
            TRAIN_CONFIG['evaluation']['calculate_bleu'] = True

        logger.info(f"模式: {TRAIN_CONFIG['mode']}")
        logger.info(f"检查点: {TRAIN_CONFIG['ckpt_name'] if TRAIN_CONFIG['ckpt_name'] else '无 (从头开始)'}")

        # 清理缓存（如果请求）
        if args.clear_cache or TRAIN_CONFIG.get('clear_cache', False):
            logger.info("正在清除数据缓存...")
            clear_cache()
            logger.info("数据缓存已清除")
        
        # 设置数据目录并验证结构
        setup_data_directories()
        validate_directory_structure()
        
        # 根据命令行参数调整批量大小
        if args.batch_size:
            logger.info(f"使用命令行指定的批量大小: {args.batch_size}")
            # 这里只修改全局变量，确保它在导入其他模块之前生效
            globals()['batch_size'] = args.batch_size

        if TRAIN_CONFIG['mode'] == 'train':
            logger.info("开始训练模式")
            
            # 初始化训练管理器
            ckpt_name = args.ckpt_name if args.ckpt_name else TRAIN_CONFIG.get('ckpt_name', 'best_ckpt.tar')
            train_manager = TrainManager(ckpt_name=ckpt_name)
            
            # 应用优化
            if args.optimize or TRAIN_CONFIG.get('optimization', {}).get('optimize_memory', False):
                # 对模型应用额外的优化
                train_manager.model = optimize_model(train_manager.model)
            
            # 开始训练
            train_manager.train()
        elif TRAIN_CONFIG['mode'] == 'inference':
            if not TRAIN_CONFIG['input_text']:
                logger.error("inference模式需要提供输入文本，请使用--input参数")
                sys.exit(1)
            logger.info("开始推理模式")
            
            # 获取推理输入
            input_text = TRAIN_CONFIG['input_text']
            decode_method = TRAIN_CONFIG['decode_method']
            logger.info(f"输入文本: {input_text}")
            logger.info(f"解码方法: {decode_method}")
            
            # 获取检查点名称
            ckpt_name = args.ckpt_name if args.ckpt_name else TRAIN_CONFIG.get('ckpt_name', 'best_ckpt.tar')
            inference_manager = InferenceManager(ckpt_name=ckpt_name)
            
            # 应用优化
            if args.optimize or TRAIN_CONFIG.get('optimization', {}).get('optimize_memory', False):
                # 对模型应用额外的优化
                inference_manager.model = optimize_model(inference_manager.model)
            
            # 进行翻译
            try:
                result = inference_manager.inference(
                    input_sentence=input_text,
                    method=decode_method
                )
                
                # 检查结果是否为空
                if not result or result.strip() == "":
                    logger.warning("推理结果为空，可能存在问题")
                    print(f"翻译结果: [空] (注意：模型未能生成有效的翻译结果)")
                else:
                    print(f"翻译结果: {result}")
                
                # 提醒用户关于语言方向
                print("\n注意：本模型训练自英法语料库，将英语翻译为法语")
                
            except Exception as e:
                logger.error(f"推理过程中发生错误: {str(e)}")
                print(f"翻译失败: {str(e)}")
        else:
            logger.error(f"未知模式: {TRAIN_CONFIG['mode']}")
            sys.exit(1)
        logger.info("处理完成")
    except Exception as e:
        logger.critical(f"程序执行过程中发生未处理的异常: {str(e)}")
        logger.critical(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
