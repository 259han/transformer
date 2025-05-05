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
import time
from tqdm import tqdm
import gc  # 用于手动垃圾回收

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
    # 确保使用全局的torch模块
    import torch
    from config import device, seq_len, use_tf32, benchmark_cudnn
    from config import TRAIN_CONFIG
    
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
    
    # 检查是否支持批处理推理
    try:
        # 尝试进行批处理推理测试（确认模型可以处理批次输入）
        if hasattr(model, 'encoder') and hasattr(model, 'src_embedding'):
            batch_size = 2
            dummy_batch = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
            dummy_mask = torch.ones(batch_size, 1, seq_len, dtype=torch.bool, device=device)
            with torch.no_grad():
                _ = model.encoder(model.src_embedding(dummy_batch), dummy_mask)
            logger.info(f"批处理推理测试成功，支持批量大小: {batch_size}")
    except Exception as e:
        logger.warning(f"批处理推理测试失败: {str(e)}")
    
    # 应用量化（如果启用）
    if TRAIN_CONFIG.get('optimization', {}).get('quantize_model', False):
        try:
            import torch.quantization
            # 应用动态量化
            if hasattr(torch.quantization, 'quantize_dynamic'):
                logger.info("尝试应用动态量化...")
                # 注意：量化可能会影响模型精度
                model_quantized = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
                logger.info("动态量化应用成功")
                return model_quantized
        except Exception as e:
            logger.warning(f"无法应用量化: {str(e)}")
    
    # 应用内存优化（如果启用）
    if TRAIN_CONFIG.get('optimization', {}).get('optimize_memory', False):
        # 在推理模式下，可以释放未使用的缓冲区和参数
        for param in model.parameters():
            if hasattr(param, 'grad') and param.grad is not None:
                param.grad = None  # 释放梯度
        
        # 确保模型处于评估模式
        model.eval()
        
        logger.info("已应用内存优化")
    
    # 返回优化后的模型
    return model

def translate_file(inference_manager, input_file, output_file, decode_method):
    """
    批量翻译文件内容
    
    参数:
    - inference_manager: 推理管理器实例
    - input_file: 输入文件路径
    - output_file: 输出文件路径
    - decode_method: 解码方法
    
    返回:
    - 成功翻译的句子数量
    """
    logger.info(f"开始批量翻译文件: {input_file} -> {output_file}")
    
    try:
        # 检查是否启用并行处理
        use_parallel = (decode_method == 'greedy' and 
                       TRAIN_CONFIG.get('optimization', {}).get('parallel_file_processing', False))
        
        # 并行处理设置
        if use_parallel:
            import multiprocessing as mp
            from concurrent.futures import ThreadPoolExecutor
            
            num_workers = min(mp.cpu_count(), 
                             TRAIN_CONFIG.get('optimization', {}).get('max_workers', 4))
            logger.info(f"启用并行处理: {num_workers}个工作线程")
        
        # 优化CUDA性能（如果可用）
        if torch.cuda.is_available():
            # 清理GPU缓存
            torch.cuda.empty_cache()
            
            # 应用额外的CUDA优化
            if hasattr(torch.cuda, "amp") and torch.cuda.is_available():
                logger.info("应用CUDA优化...")
                
                # 设置CUDA流同步模式
                torch.cuda.set_stream(torch.cuda.Stream())
                
                # 如果应用其他GPU优化...
                if benchmark_cudnn:
                    torch.backends.cudnn.benchmark = True
            
            # 获取当前GPU使用情况
            if torch.cuda.is_available():
                gpu_total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**2
                gpu_used_mem = torch.cuda.memory_allocated(0) / 1024**2
                gpu_free_mem = gpu_total_mem - gpu_used_mem
                logger.info(f"GPU内存使用情况: 已使用={gpu_used_mem:.1f}MB, 空闲={gpu_free_mem:.1f}MB, 总计={gpu_total_mem:.1f}MB")
                
                # 根据剩余内存调整批量大小
                memory_per_sample = 60.0  # 假定每个样本需要约60MB的GPU内存
                memory_overhead = 512.0  # 系统开销
                calculated_batch = int((gpu_free_mem - memory_overhead) / memory_per_sample)
        
        # 读取输入文件
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        logger.info(f"共读取 {total_lines} 行文本")
        
        # 批处理大小
        if torch.cuda.is_available():
            # 基于GPU内存大小动态调整批处理大小
            batch_size = max(1, min(32, calculated_batch))
        else:
            # CPU模式下使用较小的批处理大小
            batch_size = min(16, total_lines)
            
        logger.info(f"使用批处理大小: {batch_size}")
        
        # 对于特别大的文件，使用分块处理
        chunk_size = 1000  # 每个分块的行数
        use_chunking = total_lines > 5000  # 超过5000行时使用分块
        
        # 创建输出文件
        with open(output_file, 'w', encoding='utf-8') as f:
            # 如果使用分块处理
            if use_chunking:
                logger.info(f"文件较大，使用分块处理，每块{chunk_size}行")
                
                # 处理每个分块
                for chunk_start in range(0, total_lines, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, total_lines)
                    chunk_lines = lines[chunk_start:chunk_end]
                    
                    logger.info(f"处理分块 {chunk_start//chunk_size + 1}/{(total_lines+chunk_size-1)//chunk_size}, 行 {chunk_start+1}-{chunk_end}")
                    
                    # 如果开启并行处理，则分割任务给多个线程
                    if use_parallel:
                        # 将分块进一步分割为子块
                        sub_chunks = [chunk_lines[i:i+batch_size] for i in range(0, len(chunk_lines), batch_size)]
                        
                        # 处理子块的函数
                        def process_sub_chunk(sub_chunk):
                            results = []
                            for line in sub_chunk:
                                line = line.strip()
                                if not line:
                                    results.append("")
                                else:
                                    try:
                                        result = inference_manager.inference(line, method=decode_method)
                                        results.append(result)
                                    except Exception as e:
                                        logger.error(f"翻译出错: {str(e)}")
                                        results.append(f"[翻译错误] {str(e)}")
                            return results
                        
                        # 使用线程池并行处理子块
                        with ThreadPoolExecutor(max_workers=num_workers) as executor:
                            futures = [executor.submit(process_sub_chunk, sub_chunk) for sub_chunk in sub_chunks]
                            
                            # 收集结果并写入文件
                            for future in tqdm(futures, desc=f"分块{chunk_start//chunk_size + 1}进度"):
                                sub_results = future.result()
                                for result in sub_results:
                                    f.write(result + "\n")
                    else:
                        # 不使用并行，按正常批处理模式处理分块
                        process_chunk(inference_manager, f, chunk_lines, batch_size, decode_method)
                    
                    # 每处理完一个分块，清理缓存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
            else:
                # 不分块，直接处理整个文件
                process_chunk(inference_manager, f, lines, batch_size, decode_method)
        
        logger.info(f"翻译完成，结果已保存到 {output_file}")
        
        # 翻译结束后，再次清理缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return total_lines
    
    except Exception as e:
        logger.error(f"批量翻译文件时出错: {str(e)}")
        # 出错时也要清理缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return 0

def process_chunk(inference_manager, output_file, lines, batch_size, decode_method):
    """处理文件的一个分块"""
    # 分批处理文本行
    for i in tqdm(range(0, len(lines), batch_size), desc="翻译批次"):
        batch_lines = lines[i:i+batch_size]
        valid_lines = []
        empty_line_indices = []
        
        # 识别非空行
        for j, line in enumerate(batch_lines):
            line = line.strip()
            if line:
                valid_lines.append(line)
            else:
                empty_line_indices.append(j)
        
        # 如果批次中有有效行
        if valid_lines:
            try:
                # 批量翻译
                results = batch_translate(inference_manager, valid_lines, decode_method)
                
                # 将结果写入文件，同时还原空行
                result_index = 0
                for j in range(len(batch_lines)):
                    if j in empty_line_indices:
                        output_file.write('\n')
                    else:
                        output_file.write(results[result_index] + '\n')
                        result_index += 1
            except Exception as e:
                logger.error(f"翻译批次 {i//batch_size + 1} 时出错: {str(e)}")
                # 发生错误时，逐行翻译以保证尽可能多的翻译成功
                for j, line in enumerate(batch_lines):
                    line = line.strip()
                    if not line:
                        output_file.write('\n')
                        continue
                    
                    try:
                        result = inference_manager.inference(line, method=decode_method)
                        output_file.write(result + '\n')
                    except Exception as e2:
                        logger.error(f"翻译第 {i+j+1} 行时出错: {str(e2)}")
                        output_file.write(f"[翻译错误] {str(e2)}\n")
        else:
            # 如果批次中全是空行
            for _ in range(len(batch_lines)):
                output_file.write('\n')
        
        # 每处理完一个批次，主动清理缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()  # 手动触发垃圾回收

def batch_translate(inference_manager, sentences, decode_method):
    """
    批量翻译多个句子
    
    参数:
    - inference_manager: 推理管理器实例
    - sentences: 句子列表
    - decode_method: 解码方法
    
    返回:
    - 翻译结果列表
    """
    if not sentences:
        return []
    
    # 如果只有一个句子，直接调用单句翻译
    if len(sentences) == 1:
        return [inference_manager.inference(sentences[0], method=decode_method)]
    
    # 使用推理管理器的批量推理功能
    return inference_manager.batch_inference(sentences, method=decode_method)

def interactive_translation(inference_manager, decode_method):
    """
    交互式翻译模式
    
    参数:
    - inference_manager: 推理管理器实例
    - decode_method: 解码方法
    """
    print("\n===== 交互式翻译模式 =====")
    print("模式：英语 -> 法语翻译")
    print("提示: 输入英语文本进行翻译，输入'q'或'exit'退出，输入'switch'切换解码方法")
    print(f"当前解码方法: {decode_method}")
    
    current_method = decode_method
    
    while True:
        try:
            # 获取用户输入
            user_input = input("\n> ")
            
            # 检查退出命令
            if user_input.lower() in ['q', 'exit', 'quit']:
                print("退出交互式翻译模式")
                break
            
            # 检查切换解码方法命令
            if user_input.lower() == 'switch':
                current_method = 'greedy' if current_method == 'beam' else 'beam'
                print(f"已切换解码方法: {current_method}")
                continue
                
            # 检查帮助命令
            if user_input.lower() in ['help', '?', 'h']:
                print("\n可用命令:")
                print("  'q' 或 'exit' - 退出交互式翻译")
                print("  'switch' - 切换解码方法 (beam/greedy)")
                print("  'help' 或 '?' - 显示此帮助信息")
                continue
                
            # 如果输入为空，跳过
            if not user_input.strip():
                continue
                
            # 翻译用户输入
            start_time = time.time()
            result = inference_manager.inference(user_input, method=current_method)
            end_time = time.time()
            
            # 显示翻译结果和用时
            print(f"翻译结果 ({current_method}, {(end_time-start_time)*1000:.0f}ms): {result}")
            
        except KeyboardInterrupt:
            print("\n已中断，退出交互式翻译模式")
            break
        except Exception as e:
            print(f"翻译出错: {str(e)}")

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
        parser.add_argument('--decode', type=str, default=TRAIN_CONFIG.get('decode_method', 'beam'), choices=['greedy', 'beam'], help='解码方法: greedy或beam (仅inference模式)')
        parser.add_argument('--validate_data', action='store_true', help='运行数据验证')
        parser.add_argument('--ref_file', type=str, help='用于评估的参考翻译文件 (仅evaluate模式)')
        parser.add_argument('--calc_bleu', action='store_true', help='是否计算BLEU分数')
        parser.add_argument('--clear_cache', action='store_true', help='清除数据缓存')
        parser.add_argument('--batch_size', type=int, help='训练批量大小')
        parser.add_argument('--optimize', action='store_true', help='应用额外的性能优化')
        parser.add_argument('--file', type=str, help='输入文件路径')
        parser.add_argument('--output', type=str, help='输出文件路径')
        args = parser.parse_args()

        if args.validate_data:
            logger.info("开始运行数据验证...")
            if run_all_validations():
                logger.info("数据验证通过，继续执行...")
            else:
                logger.warning("数据验证未通过，但将继续执行。请检查日志获取详细信息。")

        # 创建模式临时变量，而不是修改TRAIN_CONFIG
        mode = args.mode
        
        # 根据命令行参数更新配置，但不更改原始TRAIN_CONFIG
        run_config = {}
        if args.ckpt_name:
            run_config['ckpt_name'] = args.ckpt_name
        else:
            run_config['ckpt_name'] = TRAIN_CONFIG.get('ckpt_name', 'best_ckpt.tar')
            
        if args.input:
            run_config['input_text'] = args.input
            
        if args.decode:
            run_config['decode_method'] = args.decode
        else:
            run_config['decode_method'] = TRAIN_CONFIG.get('decode_method', 'beam')
            
        if args.calc_bleu:
            run_config['calculate_bleu'] = True

        logger.info(f"模式: {mode}")
        logger.info(f"检查点: {run_config['ckpt_name']}")

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

        if mode == 'train':
            logger.info("开始训练模式")
            
            # 初始化训练管理器
            train_manager = TrainManager(ckpt_name=run_config.get('ckpt_name'))
            
            # 应用优化
            if args.optimize or TRAIN_CONFIG.get('optimization', {}).get('optimize_memory', False):
                # 对模型应用额外的优化
                train_manager.model = optimize_model(train_manager.model)
            
            # 开始训练
            train_manager.train()
        elif mode == 'inference':
            # 检查是否指定了输入
            has_input = run_config.get('input_text') is not None
            has_input_file = args.file is not None
            
            # 如果没有指定输入也没有指定文件，则进入交互模式
            if not has_input and not has_input_file:
                # 如果命令行没有提供输入，进入交互式模式
                logger.info("未指定输入文本或文件，启动交互式翻译模式")
                
                # 获取推理管理器和参数
                ckpt_name = run_config.get('ckpt_name')
                decode_method = run_config.get('decode_method')
                logger.info(f"检查点: {ckpt_name}")
                logger.info(f"默认解码方法: {decode_method}")
                
                # 初始化推理管理器
                inference_manager = InferenceManager(ckpt_name=ckpt_name)
                
                # 应用优化
                if args.optimize or TRAIN_CONFIG.get('optimization', {}).get('optimize_memory', False):
                    inference_manager.model = optimize_model(inference_manager.model)
                
                # 进入交互式翻译模式
                interactive_translation(inference_manager, decode_method)
            
            # 文件批量翻译模式
            elif has_input_file:
                input_file = args.file
                output_file = args.output or f"{input_file}.translated.txt"
                
                # 确保输入文件存在
                if not os.path.exists(input_file):
                    logger.error(f"输入文件不存在: {input_file}")
                    sys.exit(1)
                
                logger.info(f"文件翻译模式: {input_file} -> {output_file}")
                logger.info(f"解码方法: {run_config.get('decode_method')}")
                
                # 初始化推理管理器
                inference_manager = InferenceManager(ckpt_name=run_config.get('ckpt_name'))
                
                # 应用优化
                if args.optimize or TRAIN_CONFIG.get('optimization', {}).get('optimize_memory', False):
                    inference_manager.model = optimize_model(inference_manager.model)
                
                # 批量翻译文件
                translate_file(inference_manager, input_file, output_file, run_config.get('decode_method'))
                
                print(f"\n翻译完成，结果已保存至: {output_file}")
            
            # 单句翻译模式
            else:
                logger.info("开始推理模式")
                
                # 获取推理输入
                input_text = run_config.get('input_text')
                decode_method = run_config.get('decode_method')
                logger.info(f"输入文本: {input_text}")
                logger.info(f"解码方法: {decode_method}")
                
                # 获取检查点名称
                inference_manager = InferenceManager(ckpt_name=run_config.get('ckpt_name'))
                
                # 应用优化
                if args.optimize or TRAIN_CONFIG.get('optimization', {}).get('optimize_memory', False):
                    inference_manager.model = optimize_model(inference_manager.model)
                
                # 执行推理并输出结果
                result = inference_manager.inference(input_text, method=decode_method)
                print(f"\n原文(英语): {input_text}")
                print(f"翻译(法语): {result}")
                print("\n注意：当前模型将英语翻译为法语。")
        else:
            logger.error(f"未知模式: {mode}")
            sys.exit(1)
        logger.info("处理完成")
    except Exception as e:
        logger.critical(f"程序执行过程中发生未处理的异常: {str(e)}")
        logger.critical(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
