import torch
import numpy as np
import os
import datetime
from tqdm import tqdm
from torch import nn
from torch.amp import autocast, GradScaler
from config import *
from custom_data import get_data_loader, pad_or_truncate
from transformer import Transformer
from logger import logger
from bleu_score import compute_bleu_from_lists  # 导入BLEU评分计算函数

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

class TrainManager:
    def __init__(self, ckpt_name=None):
        print(f"当前使用设备: {device}")
        print(f"PyTorch是否可以使用CUDA: {torch.cuda.is_available()}")
        
        # 1. 词表加载
        self.src_i2w = {}
        self.trg_i2w = {}
        self.src_w2i = {}
        self.trg_w2i = {}
        with open(f"{SP_DIR}/{src_model_prefix}.vocab", 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            word = line.strip().split('\t')[0]
            self.src_i2w[i] = word
            self.src_w2i[word] = i
        with open(f"{SP_DIR}/{trg_model_prefix}.vocab", 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            word = line.strip().split('\t')[0]
            self.trg_i2w[i] = word
            self.trg_w2i[word] = i
            
        # 参数验证：确保模型配置的合理性
        if d_model % num_heads != 0:
            error_msg = f"d_model({d_model})必须能被num_heads({num_heads})整除"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # 2. 模型、优化器、损失、AMP
        self.model = Transformer(src_vocab_size=len(self.src_i2w), trg_vocab_size=len(self.trg_i2w)).to(device)
        
        # 启用梯度检查点以节省内存并允许更大批量
        self.model.enable_checkpointing(True)
        logger.info("启用梯度检查点以节省GPU内存")
        
        # 3. 应用channels_last内存格式优化（对于卷积和大型张量操作更高效）
        if channels_last_memory_format:
            self.model = self.model.to(memory_format=torch.channels_last)
            logger.info("已转换模型为channels_last内存格式")
        
        # 设置混合精度训练
        self.use_amp = mixed_precision is not None and mixed_precision is not False
        self.use_bf16 = isinstance(mixed_precision, str) and mixed_precision.lower() == 'bf16'

        # 检查是否支持BF16
        if self.use_bf16 and torch.cuda.is_available():
            if torch.__version__ >= '1.12.0' and torch.cuda.is_bf16_supported():
                logger.info("使用BF16混合精度训练")
                # BF16模式下不需要scaler，精度更稳定
                self.scaler = None
            else:
                logger.warning("当前环境不支持BF16，回退到FP16混合精度")
                self.use_bf16 = False
                self.scaler = GradScaler(device='cuda', enabled=True)
        elif self.use_amp:
            # FP16混合精度
            logger.info("使用FP16混合精度训练")
            self.scaler = GradScaler(device='cuda', enabled=True)
        else:
            logger.info("未启用混合精度训练")
            self.scaler = None
        
        # 使用Adam优化器，添加权重衰减
        self.optim = torch.optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate,
            betas=(0.9, 0.98),  # 原始Transformer论文中的推荐值
            eps=1e-9,  # 提高数值稳定性
            weight_decay=weight_decay  # 从配置中获取权重衰减值
        )
        logger.info(f"使用AdamW优化器，学习率={learning_rate}，权重衰减={weight_decay}")
        
        # 加载数据
        self.train_loader = get_data_loader(TRAIN_NAME, shuffle=True)
        self.valid_loader = get_data_loader(VALID_NAME, shuffle=False)
        
        # 计算每个周期的总步数，用于学习率调度器
        steps_per_epoch = len(self.train_loader) // gradient_accumulation
        total_steps = steps_per_epoch * num_epochs
        
        # 添加学习率调度器
        lr_scheduler_type = MODEL_CONFIG.get('lr_scheduler', 'inverse_sqrt')
        logger.info(f"使用学习率调度器类型: {lr_scheduler_type}")

        if lr_scheduler_type == 'inverse_sqrt':
            # 原始Transformer论文中的学习率调度策略
            # lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
            def lr_lambda(current_step):
                current_step = max(1, current_step)
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return max(0.0, float(warmup_steps ** 0.5) * float(current_step ** -0.5))
            
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda)
            logger.info(f"使用逆平方根学习率调度，包含{warmup_steps}步的预热阶段，初始学习率={learning_rate}")
        else:
            # 默认使用线性衰减
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optim,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=total_steps
            )
            logger.info(f"使用线性学习率调度，无预热阶段，总步数={total_steps}")
            
        # 使用带标签平滑的交叉熵损失
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=pad_id,
            label_smoothing=label_smoothing  # 添加标签平滑
        )
        logger.info(f"使用带标签平滑(smoothing={label_smoothing})的CrossEntropyLoss损失函数")
        
        self.best_loss = float('inf')
        self.best_bleu = 0.0  # 添加最佳BLEU分数记录
        self.last_epoch = 0
        self.global_step = 0  # 添加全局步数计数器，用于基于步数的验证
        
        # 创建一个三角掩码，在训练过程中重复使用而不是每次创建
        self.nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool).to(device)
        self.nopeak_mask = torch.tril(self.nopeak_mask)
        
        # 3. 断点续训
        self.ckpt_name = ckpt_name if ckpt_name is not None else TRAIN_CONFIG.get('ckpt_name', 'best_ckpt.tar')
        self.auto_resume = TRAIN_CONFIG.get('auto_resume', True)
        
        # 确保检查点目录存在
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
            logger.info(f"创建检查点目录: {ckpt_dir}")
            
        if self.auto_resume and os.path.exists(f"{ckpt_dir}/{self.ckpt_name}"):
            logger.info(f"检测到检查点 {ckpt_dir}/{self.ckpt_name}，自动恢复训练...")
            checkpoint = torch.load(f"{ckpt_dir}/{self.ckpt_name}", map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optim.load_state_dict(checkpoint['optim_state_dict'])
            
            # 加载混合精度设置
            if 'mixed_precision' in checkpoint:
                mixed_precision_info = checkpoint['mixed_precision']
                if mixed_precision_info.get('enabled', False):
                    # 检查是否需要转换BF16设置
                    checkpoint_bf16 = mixed_precision_info.get('bf16', False)
                    if checkpoint_bf16 != self.use_bf16:
                        logger.warning(f"检查点的BF16设置 ({checkpoint_bf16}) 与当前设置 ({self.use_bf16}) 不一致!")
            
            # 根据混合精度设置加载scaler
            if self.use_amp and not self.use_bf16 and self.scaler is not None:
                if 'scaler' in checkpoint and checkpoint['scaler'] is not None:
                    self.scaler.load_state_dict(checkpoint['scaler'])
                    logger.info("已恢复混合精度scaler状态")
                else:
                    logger.warning("检查点中未包含scaler状态, 将使用新的scaler")
            
            self.last_epoch = checkpoint.get('epoch', 0)
            self.global_step = checkpoint.get('global_step', 0)  # 恢复全局步数
            self.best_loss = checkpoint.get('best_loss', checkpoint.get('loss', float('inf')))
            self.best_bleu = checkpoint.get('best_bleu', checkpoint.get('bleu', 0.0))
            # 恢复学习率调度器状态
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info(f"恢复到 epoch {self.last_epoch}，step {self.global_step}，best_loss={self.best_loss}，best_bleu={self.best_bleu}")
        else:
            logger.info("未检测到可用检查点，将从头开始训练。")

    def train(self):
        early_stopper = EarlyStopping(patience=early_stopping_patience,
                                      min_delta=early_stopping_min_delta)
        logger.info(f"训练开始，最大epoch={num_epochs}")
        grad_accum_steps = gradient_accumulation
        clip_grad_norm_val = clip_grad_norm
        
        # 预计算总批次数，用于进度显示
        total_steps = len(self.train_loader)
        logger.info(f"每个epoch包含 {total_steps} 批数据，批大小={batch_size}")
        
        # 添加CUDA事件以进行性能分析
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
        
        # 禁用梯度的异步计算，提高性能
        torch.set_grad_enabled(False)
        
        for epoch in range(self.last_epoch + 1, num_epochs + 1):
            self.model.train()
            train_losses = []
            # 输出当前学习率
            current_lr = self.scheduler.get_last_lr()[0]
            logger.info(f"Epoch {epoch} 开始，当前学习率: {current_lr:.6f}")
            
            # 创建空的梯度累积张量
            if grad_accum_steps > 1:
                logger.info(f"使用梯度累积，每 {grad_accum_steps} 步更新一次权重")
            
            self.optim.zero_grad(set_to_none=True)
            
            # 计时开始
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                epoch_start_time = torch.cuda.Event(enable_timing=True)
                epoch_end_time = torch.cuda.Event(enable_timing=True)
                epoch_start_time.record()
            
            with tqdm(total=total_steps, desc=f"Epoch {epoch} 训练进度", dynamic_ncols=True) as pbar:
                for batch_idx, batch in enumerate(self.train_loader):
                    # 启用梯度计算，但仅在前向和反向传播期间
                    torch.set_grad_enabled(True)
                    
                    # 计时开始
                    if torch.cuda.is_available():
                        start_event.record()
                    
                    # 移动数据到设备并创建掩码 (批处理以提高效率)
                    src_input, trg_input, trg_output = [x.to(device, non_blocking=True) for x in batch]
                    e_mask, d_mask = self.make_batch_mask(src_input, trg_input)
                    
                    # 混合精度训练 - 减少GPU内存使用并提高速度
                    if self.use_amp:
                        if self.use_bf16:
                            # 使用BF16混合精度，更高精度和稳定性
                            with autocast('cuda', dtype=torch.bfloat16):
                                output = self.model(src_input, trg_input, e_mask, d_mask)
                                loss = self.criterion(output.view(-1, sp_vocab_size), trg_output.view(-1))
                            loss = loss / grad_accum_steps
                            loss.backward()
                        else:
                            # 使用FP16混合精度，需要梯度缩放
                            with autocast('cuda', dtype=torch.float16):
                                output = self.model(src_input, trg_input, e_mask, d_mask)
                                loss = self.criterion(output.view(-1, sp_vocab_size), trg_output.view(-1))
                            loss = loss / grad_accum_steps
                            self.scaler.scale(loss).backward()
                    else:
                        output = self.model(src_input, trg_input, e_mask, d_mask)
                        loss = self.criterion(output.view(-1, sp_vocab_size), trg_output.view(-1))
                        loss = loss / grad_accum_steps
                        loss.backward()
                    
                    # 再次禁用梯度计算以节省内存
                    torch.set_grad_enabled(False)
                    
                    train_losses.append(loss.item() * grad_accum_steps)
                    
                    # 计时结束
                    if torch.cuda.is_available():
                        end_event.record()
                        torch.cuda.synchronize()
                        step_time = start_event.elapsed_time(end_event) / 1000  # 转换为秒
                    
                    # 梯度累积实现 - 减少内存使用并允许更大批量
                    if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == total_steps:
                        # 梯度裁剪 - 防止梯度爆炸
                        if clip_grad_norm_val is not None and clip_grad_norm_val > 0:
                            if self.use_amp and not self.use_bf16 and self.scaler is not None:
                                self.scaler.unscale_(self.optim)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad_norm_val)
                        
                        # 更新参数
                        if self.use_amp and not self.use_bf16 and self.scaler is not None:
                            self.scaler.step(self.optim)
                            self.scaler.update()
                        else:
                            self.optim.step()
                        self.optim.zero_grad(set_to_none=True)  # 使用set_to_none=True释放梯度内存
                        
                        # 更新学习率 - 放在梯度更新后
                        self.scheduler.step()
                        
                        # 增加全局步数计数器
                        self.global_step += 1
                        
                        # 基于步数进行验证
                        if validate_every_steps > 0 and self.global_step % validate_every_steps == 0:
                            # 在GPU上清理缓存
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            
                            logger.info(f"全局步数 {self.global_step}，开始基于步数的定期验证...")
                            val_loss, val_bleu = self.validation(epoch, step=self.global_step)
                            
                            # 模型选择策略 - 同时考虑损失和BLEU
                            bleu_improved = val_bleu > self.best_bleu
                            loss_improved = val_loss < self.best_loss
                            
                            if loss_improved or (bleu_improved and val_loss <= self.best_loss * 1.01):
                                improvement_msg = []
                                if loss_improved:
                                    improvement_msg.append(f"验证损失: {self.best_loss:.4f} -> {val_loss:.4f}")
                                    self.best_loss = val_loss
                                if bleu_improved:
                                    improvement_msg.append(f"BLEU分数: {self.best_bleu:.2f} -> {val_bleu:.2f}")
                                    self.best_bleu = val_bleu
                                
                                logger.info(f"模型性能提升: {', '.join(improvement_msg)}")
                                self.save_checkpoint(epoch, val_loss, val_bleu, self.global_step)
                                
                            # 早停机制
                            if early_stopper.step(val_loss):
                                logger.info(f"早停触发，验证损失 {val_loss:.4f}，最佳损失 {self.best_loss:.4f}")
                                return  # 提前结束训练
                                
                            # 恢复训练模式
                            self.model.train()
                            
                            # 验证后清理内存
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                    
                    # 更新进度条，但减少更新频率以提高性能
                    if (batch_idx+1) % 20 == 0:
                        avg_loss = np.mean(train_losses[-20:]) if len(train_losses) >= 20 else np.mean(train_losses)
                        # 添加显示当前学习率和批处理时间
                        current_lr = self.scheduler.get_last_lr()[0]
                        
                        # 如果CUDA可用，显示批处理时间
                        if torch.cuda.is_available():
                            pbar.set_postfix({
                                "loss": f"{avg_loss:.4f}", 
                                "lr": f"{current_lr:.6f}",
                                "time/batch": f"{step_time:.3f}s"
                            })
                        else:
                            pbar.set_postfix({
                                "loss": f"{avg_loss:.4f}", 
                                "lr": f"{current_lr:.6f}"
                            })
                    pbar.update(1)
            
            # 记录epoch耗时
            if torch.cuda.is_available():
                epoch_end_time.record()
                torch.cuda.synchronize()
                epoch_time = epoch_start_time.elapsed_time(epoch_end_time) / 1000  # 转换为秒
                logger.info(f"Epoch {epoch} 耗时: {epoch_time:.2f} 秒")
            
            # 计算整个epoch的平均损失
            mean_train_loss = np.mean(train_losses)
            logger.info(f"Epoch {epoch} 训练损失: {mean_train_loss:.4f}")
            
            # 保存每个Epoch的检查点（与验证策略无关）
            if save_epoch_checkpoints:
                # 如果使用基于步数验证，此处可能还没有当前epoch的验证结果
                # 因此需要进行一次验证以获取最新指标
                epoch_val_loss, epoch_val_bleu = self.validation(epoch, step=self.global_step)
                self.save_epoch_checkpoint(epoch, epoch_val_loss, epoch_val_bleu, self.global_step)
                logger.info(f"已完成Epoch {epoch}的检查点保存")
                
                # 如果这次验证表明模型性能提升，也保存最佳模型
                bleu_improved = epoch_val_bleu > self.best_bleu
                loss_improved = epoch_val_loss < self.best_loss
                
                if loss_improved or (bleu_improved and epoch_val_loss <= self.best_loss * 1.01):
                    improvement_msg = []
                    if loss_improved:
                        improvement_msg.append(f"验证损失: {self.best_loss:.4f} -> {epoch_val_loss:.4f}")
                        self.best_loss = epoch_val_loss
                    if bleu_improved:
                        improvement_msg.append(f"BLEU分数: {self.best_bleu:.2f} -> {epoch_val_bleu:.2f}")
                        self.best_bleu = epoch_val_bleu
                    
                    logger.info(f"Epoch结束验证：模型性能提升: {', '.join(improvement_msg)}")
                    self.save_checkpoint(epoch, epoch_val_loss, epoch_val_bleu, self.global_step)
                
                # 检查是否需要早停
                if early_stopper.step(epoch_val_loss):
                    logger.info(f"早停触发，验证损失 {epoch_val_loss:.4f}，最佳损失 {self.best_loss:.4f}")
                    break
            # 如果未启用基于步数的验证，且不保存每个epoch检查点，则需要单独进行验证
            elif validate_every_steps <= 0:
                val_loss, val_bleu = self.validation(epoch)
                
                # 模型选择策略 - 同时考虑损失和BLEU
                bleu_improved = val_bleu > self.best_bleu
                loss_improved = val_loss < self.best_loss
                
                if loss_improved or (bleu_improved and val_loss <= self.best_loss * 1.01):
                    improvement_msg = []
                    if loss_improved:
                        improvement_msg.append(f"验证损失: {self.best_loss:.4f} -> {val_loss:.4f}")
                        self.best_loss = val_loss
                    if bleu_improved:
                        improvement_msg.append(f"BLEU分数: {self.best_bleu:.2f} -> {val_bleu:.2f}")
                        self.best_bleu = val_bleu
                    
                    logger.info(f"模型性能提升: {', '.join(improvement_msg)}")
                    self.save_checkpoint(epoch, val_loss, val_bleu, self.global_step)
                    
                # 早停机制
                if early_stopper.step(val_loss):
                    logger.info(f"早停触发，验证损失 {val_loss:.4f}，最佳损失 {self.best_loss:.4f}")
                    break
        
        logger.info("训练结束")

    def validation(self, epoch, step=None):
        """优化的验证函数，支持基于步数的验证"""
        self.model.eval()
        valid_losses = []
        references = []
        hypotheses = []
        
        # 增加BLEU评估样本数量
        max_bleu_samples = 500  # 从原来的100增加到500
        
        step_str = f" (step {step})" if step is not None else ""
        logger.info(f"Epoch {epoch}{step_str} 开始验证...")
        
        # 禁用梯度计算以加速验证
        with torch.no_grad():
            # 使用小批量进行验证，加速处理
            for batch_idx, batch in enumerate(self.valid_loader):
                # 移动数据到设备并创建掩码 (批处理以提高效率)
                src_input, trg_input, trg_output = [x.to(device, non_blocking=True) for x in batch]
                e_mask, d_mask = self.make_batch_mask(src_input, trg_input)
                
                # 前向传播
                if self.use_amp:
                    # 根据混合精度类型选择不同的数据类型
                    with autocast('cuda', dtype=torch.bfloat16 if self.use_bf16 else torch.float16):
                        output = self.model(src_input, trg_input, e_mask, d_mask)
                        loss = self.criterion(output.view(-1, sp_vocab_size), trg_output.view(-1))
                else:
                    output = self.model(src_input, trg_input, e_mask, d_mask)
                    loss = self.criterion(output.view(-1, sp_vocab_size), trg_output.view(-1))
                    
                valid_losses.append(loss.item())
                
                # 为BLEU评估收集足够的样本
                if len(references) < max_bleu_samples:
                    # 使用GPU加速argmax操作
                    pred = output.argmax(dim=-1)
                    
                    # 只有在需要时才移至CPU
                    if len(references) + pred.size(0) > max_bleu_samples:
                        # 计算还需要多少样本
                        samples_needed = max_bleu_samples - len(references)
                        # 只处理需要的样本数量
                        pred = pred[:samples_needed].cpu().numpy()
                        target = trg_output[:samples_needed].cpu().numpy()
                    else:
                        pred = pred.cpu().numpy()
                        target = trg_output.cpu().numpy()
                    
                    for i in range(len(pred)):
                        if len(references) >= max_bleu_samples:
                            break
                            
                        pred_tokens = []
                        for token_id in pred[i]:
                            if token_id == eos_id:
                                break
                            if token_id != pad_id and token_id != sos_id:
                                pred_tokens.append(self.trg_i2w[token_id])
                                
                        target_tokens = []
                        for token_id in target[i]:
                            if token_id == eos_id:
                                break
                            if token_id != pad_id and token_id != sos_id:
                                target_tokens.append(self.trg_i2w[token_id])
                        
                        if pred_tokens and target_tokens:  # 确保序列不为空
                            hypotheses.append(" ".join(pred_tokens))
                            references.append(" ".join(target_tokens))
                
                # 移除固定批次限制，让验证更全面
                # 但为了效率，仍在达到目标样本数后退出
                if len(references) >= max_bleu_samples and batch_idx >= 100:
                    break
        
        # 计算平均损失
        mean_loss = np.mean(valid_losses)
        step_info = f"step {step}, " if step is not None else ""
        logger.info(f"Epoch {epoch}, {step_info}验证损失: {mean_loss:.4f}")
        
        # 计算BLEU分数
        bleu_score = 0.0
        if references and hypotheses and len(references) == len(hypotheses):
            try:
                bleu_result = compute_bleu_from_lists(references, hypotheses)
                bleu_score = bleu_result['bleu']
                logger.info(f"Epoch {epoch}, {step_info}BLEU分数: {bleu_score:.2f} (基于 {len(references)} 个样本)")
            except Exception as e:
                logger.error(f"计算BLEU分数时出错: {str(e)}")
        
        return mean_loss, bleu_score

    # 优化后的批量掩码创建函数
    def make_batch_mask(self, src_input, trg_input):
        """
        创建用于Transformer的注意力掩码
        
        参数:
        - src_input: 源序列输入，形状 (batch_size, src_seq_len)
        - trg_input: 目标序列输入，形状 (batch_size, trg_seq_len)
        
        返回:
        - e_mask: 编码器掩码，形状 (batch_size, 1, src_seq_len)
        - d_mask: 解码器掩码，形状 (batch_size, 1, trg_seq_len)，包含前瞻屏蔽
        """
        # 创建编码器掩码：屏蔽填充标记
        e_mask = (src_input != pad_id).unsqueeze(1)
        
        # 创建解码器掩码：屏蔽填充标记和未来位置
        d_mask = (trg_input != pad_id).unsqueeze(1)
        d_mask = d_mask & self.nopeak_mask  # 使用预计算的前瞻掩码
        
        return e_mask, d_mask

    def save_checkpoint(self, epoch, val_loss, val_bleu, global_step):
        state = {
            'epoch': epoch,
            'global_step': global_step,  # 保存全局步数
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': self.optim.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': val_loss,
            'bleu': val_bleu,
            'best_loss': self.best_loss,
            'best_bleu': self.best_bleu,
            'scaler': self.scaler.state_dict() if self.use_amp and not self.use_bf16 and self.scaler is not None else None,
            'mixed_precision': {'enabled': self.use_amp, 'bf16': self.use_bf16}
        }
        torch.save(state, f"{ckpt_dir}/best_ckpt.tar")
        logger.info(f"已保存最佳模型到 {ckpt_dir}/best_ckpt.tar (epoch {epoch}, step {global_step})")

    def save_epoch_checkpoint(self, epoch, val_loss, val_bleu, global_step):
        """保存每个epoch的检查点，用于训练过程可视化和分析"""
        # 如果配置为不保存epoch检查点，则直接返回
        if not save_epoch_checkpoints:
            return
            
        # 检查是否达到保存频率
        if epoch % save_epoch_frequency != 0:
            logger.info(f"Epoch {epoch} 未达到保存频率 (每{save_epoch_frequency}个epoch)，跳过检查点保存")
            return
        
        state = {
            'epoch': epoch,
            'global_step': global_step,  # 保存全局步数
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': self.optim.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': val_loss,
            'bleu': val_bleu,
            'scaler': self.scaler.state_dict() if self.use_amp and not self.use_bf16 and self.scaler is not None else None,
            'mixed_precision': {'enabled': self.use_amp, 'bf16': self.use_bf16}
        }
        
        checkpoint_path = f"{ckpt_dir}/epoch_{epoch}.tar"
        torch.save(state, checkpoint_path)
        logger.info(f"已保存epoch {epoch}的检查点到 {checkpoint_path}")
        
        # 清理旧检查点文件
        if keep_last_n_checkpoints > 0:
            try:
                # 获取所有epoch检查点文件
                import glob
                import os
                checkpoint_files = glob.glob(f"{ckpt_dir}/epoch_*.tar")
                # 解析文件名，提取epoch号
                epoch_files = []
                for file_path in checkpoint_files:
                    file_name = os.path.basename(file_path)
                    if file_name.startswith("epoch_") and file_name.endswith(".tar"):
                        try:
                            epoch_num = int(file_name[6:-4])  # 提取 "epoch_X.tar" 中的 X
                            epoch_files.append((epoch_num, file_path))
                        except ValueError:
                            continue
                
                # 按epoch号排序
                epoch_files.sort(reverse=True)
                
                # 删除旧文件，保留最新的N个
                if len(epoch_files) > keep_last_n_checkpoints:
                    for _, file_path in epoch_files[keep_last_n_checkpoints:]:
                        try:
                            os.remove(file_path)
                            logger.info(f"已删除旧检查点文件: {file_path}")
                        except Exception as e:
                            logger.warning(f"删除旧检查点文件失败: {file_path}, 错误: {str(e)}")
            except Exception as e:
                logger.warning(f"清理旧检查点文件时出错: {str(e)}")
                # 错误不应阻止训练继续进行 