import torch
import torch.nn.functional as F
import sentencepiece as spm
from config import *
from custom_data import pad_or_truncate
from transformer import Transformer
import numpy as np
import datetime
from data_structure import BeamNode, PriorityQueue
from logger import logger

# 添加安全全局变量以允许numpy相关类型
from numpy._core.multiarray import scalar
from numpy import dtype
torch.serialization.add_safe_globals([scalar, dtype])

# 添加更多安全类型
# 对于无法直接导入的类型，使用字符串形式添加
torch.serialization.add_safe_globals(["numpy.dtypes.Float64DType"])

class InferenceManager:
    def __init__(self, ckpt_name=None):
        # 词表加载
        self.src_i2w = {}
        self.trg_i2w = {}
        self.src_w2i = {}
        self.trg_w2i = {}
        
        # 加载源语言和目标语言词表
        self._load_vocabulary()
        
        # 初始化SentencePiece处理器（延迟加载）
        self.src_sp = None
        self.trg_sp = None
        
        # 加载模型
        self.model = Transformer(src_vocab_size=len(self.src_i2w), trg_vocab_size=len(self.trg_i2w)).to(device)
        if ckpt_name is not None:
            checkpoint = torch.load(f"{ckpt_dir}/{ckpt_name}", map_location='cpu', weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
    
    def _load_vocabulary(self):
        """加载词汇表"""
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
    
    def _ensure_tokenizers_loaded(self):
        """确保分词器已加载"""
        if self.src_sp is None:
            self.src_sp = spm.SentencePieceProcessor()
            self.src_sp.Load(f"{SP_DIR}/{src_model_prefix}.model")
        
        if self.trg_sp is None:
            self.trg_sp = spm.SentencePieceProcessor()
            self.trg_sp.Load(f"{SP_DIR}/{trg_model_prefix}.model")
            # 验证SentencePiece解码器
            self._test_sentencepiece(self.trg_sp)

    def inference(self, input_sentence, method=None):
        """
        执行推理翻译
        
        参数:
        - input_sentence: 要翻译的输入文本
        - method: 解码方法，'greedy'或'beam'，默认使用config中的decode_method设置
        
        返回:
        - 翻译后的文本
        """
        # 确保只处理一个句子的情况
        if isinstance(input_sentence, list):
            if len(input_sentence) == 1:
                input_sentence = input_sentence[0]
            else:
                return self.batch_inference(input_sentence, method)
                
        # 使用配置中的默认解码方法（如果未指定）
        if method is None:
            method = TRAIN_CONFIG.get('decode_method', 'beam')
        
        # 确保分词器已加载
        self._ensure_tokenizers_loaded()
            
        input_sentence = input_sentence.strip()
        tokenized = self.src_sp.EncodeAsIds(input_sentence)
        logger.debug(f"输入分词后的token IDs: {tokenized}")
        src_data = torch.LongTensor(pad_or_truncate(tokenized + [eos_id])).unsqueeze(0).to(device)
        logger.debug(f"输入张量形状: {src_data.shape}")
        e_mask = (src_data != pad_id).unsqueeze(1).to(device)
        with torch.no_grad():
            src_data = self.model.src_embedding(src_data)
            src_data = self.model.positional_encoder(src_data)
            e_output = self.model.encoder(src_data, e_mask)
            if method == 'greedy':
                result = self.greedy_search(e_output, e_mask, self.trg_sp)
            elif method == 'beam':
                result = self.beam_search(e_output, e_mask, self.trg_sp)
            else:
                logger.warning(f"未知的解码方法: {method}，使用默认的beam搜索")
                result = self.beam_search(e_output, e_mask, self.trg_sp)
        logger.debug(f"翻译结果: {result}")
        return result
    
    def batch_inference(self, sentences, method=None):
        """
        批量执行推理翻译
        
        参数:
        - sentences: 要翻译的输入文本列表
        - method: 解码方法，'greedy'或'beam'，默认使用config中的decode_method设置
        
        返回:
        - 翻译结果列表
        """
        if not sentences:
            return []
            
        # 使用配置中的默认解码方法（如果未指定）
        if method is None:
            method = TRAIN_CONFIG.get('decode_method', 'beam')
        
        # 确保分词器已加载
        self._ensure_tokenizers_loaded()
        
        # 对于beam search，当前只能逐个处理
        if method == 'beam':
            logger.info("束搜索模式下使用逐个句子翻译")
            results = []
            for sentence in sentences:
                results.append(self.inference(sentence, method))
            return results
        
        # 对于贪婪搜索，可以批量处理
        logger.info(f"贪婪搜索模式下进行批量翻译，句子数: {len(sentences)}")
        
        # 将句子标记化并填充到相同长度
        batch_tokens = []
        max_len = 0
        
        # 首先确定最大长度
        for sentence in sentences:
            sentence = sentence.strip()
            tokens = self.src_sp.EncodeAsIds(sentence) + [eos_id]
            batch_tokens.append(tokens)
            max_len = max(max_len, len(tokens))
        
        # 添加填充
        padded_batch = []
        for tokens in batch_tokens:
            padded = tokens + [pad_id] * (max_len - len(tokens))
            padded_batch.append(padded)
        
        # 将批处理转换为张量
        src_data = torch.LongTensor(padded_batch).to(device)
        e_mask = (src_data != pad_id).unsqueeze(1).to(device)
        
        # 编码批处理
        with torch.no_grad():
            embedded = self.model.src_embedding(src_data)
            embedded = self.model.positional_encoder(embedded)
            e_output = self.model.encoder(embedded, e_mask)
        
        # 对批处理中的每个样本执行贪婪搜索
        results = []
        for i in range(len(sentences)):
            # 提取当前样本的编码器输出和掩码
            single_e_output = e_output[i:i+1]
            single_e_mask = e_mask[i:i+1]
            
            # 执行贪婪搜索
            result = self.greedy_search(single_e_output, single_e_mask, self.trg_sp)
            results.append(result)
        
        return results

    def greedy_search(self, e_output, e_mask, trg_sp):
        """贪婪搜索解码"""
        # 初始化
        generated = [sos_id]
        min_length = 5  # 最小生成长度
        
        # 获取输入序列长度
        batch_size, _, src_len = e_mask.size()
        
        # 最大生成token数为源序列长度的一半或80，取较大值
        max_token_to_generate = max(src_len // 2, 80)
        logger.debug(f"最大生成token数: {max_token_to_generate}")
        
        consecutive_pad_limit = 3  # 连续PAD标记的上限
        consecutive_pad_count = 0
        
        logger.debug("开始贪婪搜索")
        
        for step in range(max_token_to_generate):
            # 将当前序列转换为张量并填充
            decoder_input = pad_or_truncate(generated.copy())
            decoder_input = torch.LongTensor([decoder_input]).to(device)
            
            # 创建掩码
            d_mask = (decoder_input != pad_id).unsqueeze(1).to(device)
            nopeak_mask = torch.ones([1, src_len, src_len], dtype=torch.bool).to(device)
            nopeak_mask = torch.tril(nopeak_mask)
            d_mask = d_mask & nopeak_mask
            
            # 编码解码过程
            with torch.no_grad():
                decoder_embedded = self.model.trg_embedding(decoder_input)
                decoder_embedded = self.model.positional_encoder(decoder_embedded)
                
                decoder_output = self.model.decoder(
                    decoder_embedded,
                    e_output,
                    e_mask,
                    d_mask
                )
                
                output = self.model.output_linear(decoder_output)
                # 使用F.softmax替换模型的softmax
                output = F.log_softmax(output, dim=-1)
            
            # 获取当前步的预测结果
            cur_len = len(generated)
            prob = output[0, cur_len-1]
            
            # 如果序列太短，不要生成EOS或特殊token
            if len(generated) < min_length:
                prob[eos_id] = float('-inf')
                prob[pad_id] = float('-inf')
                prob[sos_id] = float('-inf')
                prob[unk_id] = float('-inf')
            
            # 获取概率最高的token
            next_token = prob.argmax().item()
            
            # 如果是EOS标记，停止生成
            if next_token == eos_id:
                logger.debug("生成了EOS标记，停止生成")
                break
            
            # 如果是PAD标记，计数并可能停止
            if next_token == pad_id:
                consecutive_pad_count += 1
                if consecutive_pad_count >= consecutive_pad_limit:
                    logger.debug(f"连续生成了{consecutive_pad_count}个PAD标记，停止生成")
                    break
            else:
                consecutive_pad_count = 0
                
            # 添加到生成序列
            generated.append(next_token)
        
        # 去除SOS标记和所有的填充标记
        result = [token for token in generated[1:] if token != pad_id]
        
        # 如果最后一个是EOS，也去除
        if result and result[-1] == eos_id:
            result = result[:-1]
            
        logger.debug(f"贪婪搜索完成，生成了{len(result)}个token")
        
        # 解码结果
        if not result:
            return "生成失败：没有有效的翻译结果"
            
        try:
            translated_text = trg_sp.decode_ids(result)
            return translated_text
        except Exception as e:
            logger.error(f"解码出错: {str(e)}")
            return f"解码错误: {str(e)}"

    def beam_search(self, e_output, e_mask, trg_sp):
        """束搜索解码"""
        # 初始化队列
        cur_queue = PriorityQueue()
        # 初始节点概率为0（log域）
        cur_queue.put(BeamNode(sos_id, 0.0, [sos_id]))
        
        # 参数设置
        finished_count = 0
        min_length = 5  # 最小翻译长度
        max_length = seq_len - 2  # 最大长度，保留结束标记位置
        
        logger.debug("开始束搜索解码")
        
        # 逐位置生成序列
        for pos in range(max_length):
            # 创建新队列
            new_queue = PriorityQueue()
            queue_size = min(cur_queue.qsize(), beam_size)
            
            if queue_size == 0:
                break
                
            logger.debug(f"束搜索位置 {pos}, 队列大小: {queue_size}")
            
            # 处理当前队列中的每个节点
            for k in range(queue_size):
                if cur_queue.qsize() == 0:
                    break
                    
                node = cur_queue.get()
                
                # 如果节点已经完成（生成了EOS），直接放入新队列
                if node.is_finished:
                    new_queue.put(node)
                    continue
                
                # 准备输入序列
                seq_len_now = len(node.decoded)
                trg_input = torch.LongTensor(node.decoded + [pad_id] * (seq_len - seq_len_now)).to(device)
                
                # 创建掩码，与训练过程保持一致
                d_mask = (trg_input.unsqueeze(0) != pad_id).unsqueeze(1).to(device)
                nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool).to(device)
                nopeak_mask = torch.tril(nopeak_mask)
                d_mask = d_mask & nopeak_mask
                
                # 模型前向传播
                trg_embedded = self.model.trg_embedding(trg_input.unsqueeze(0))
                trg_positional_encoded = self.model.positional_encoder(trg_embedded)
                decoder_output = self.model.decoder(
                    trg_positional_encoded,
                    e_output,
                    e_mask,
                    d_mask
                )
                
                # 获取当前位置的输出
                logits = self.model.output_linear(decoder_output)[0, seq_len_now-1]
                # 使用torch.nn.functional.softmax代替模型的softmax
                output = F.log_softmax(logits, dim=-1)
                
                # 禁止短序列生成EOS和特殊标记
                if seq_len_now < min_length:
                    output[eos_id] = float('-inf')  # 不要生成EOS
                    output[pad_id] = float('-inf')  # 不要生成PAD
                    output[sos_id] = float('-inf')  # 不要生成SOS
                
                # 获取最可能的beam_size个词
                topk_probs, topk_ids = torch.topk(output, k=beam_size)
                
                # 创建新的beam节点
                for i, (prob, token_id) in enumerate(zip(topk_probs, topk_ids)):
                    token_id_val = token_id.item()
                    log_prob_val = prob.item()  # 已经是log概率
                    
                    # 新节点总分 = 当前节点总分 + 新token对数概率
                    new_log_prob = node.prob + log_prob_val
                    new_decoded = node.decoded + [token_id_val]
                    new_node = BeamNode(token_id_val, new_log_prob, new_decoded)
                    
                    # 如果生成EOS，标记完成
                    if token_id_val == eos_id and len(new_decoded) > min_length:
                        new_node.is_finished = True
                        finished_count += 1
                        logger.debug(f"发现完成序列，分数: {new_log_prob:.4f}")
                    
                    # 添加到新队列
                    new_queue.put(new_node)
            
            # 更新当前队列
            cur_queue = new_queue
            
            # 足够的序列已完成时可以提前结束
            if finished_count >= beam_size:
                logger.debug(f"束搜索提前完成，已有{finished_count}个束生成了结束标记")
                break
        
        # 处理结果：从队列中获取所有节点
        candidates = []
        while cur_queue.qsize() > 0:
            node = cur_queue.get()
            # 由于我们已经修改了比较运算符，节点按概率降序排列（高概率先出队）
            candidates.append((node.prob, node.decoded, node.is_finished))
        
        # 这里不再需要对candidates做额外排序，因为已经按概率降序排列了
        # 但为了保证一致性，我们还是显式排序一下
        candidates = sorted(candidates, key=lambda x: x[0], reverse=True)
        
        logger.debug(f"束搜索完成，找到{len(candidates)}个候选序列")
        
        # 优先选择已完成的最高分候选
        for score, sequence, is_finished in candidates:
            if is_finished:
                logger.debug(f"选择完成候选，分数: {score:.4f}")
                result_seq = sequence
                break
        else:
            # 如果没有已完成的候选，选择分数最高的
            result_seq = candidates[0][1]
            logger.debug(f"未找到完成候选，选择最高分候选，分数: {candidates[0][0]:.4f}")
        
        # 移除开始标记和结束标记（如果有）
        if result_seq[0] == sos_id:
            result_seq = result_seq[1:]
            
        if result_seq and result_seq[-1] == eos_id:
            result_seq = result_seq[:-1]
            
        # 过滤特殊标记
        filtered_seq = [t for t in result_seq if t not in [pad_id, sos_id, eos_id]]
        
        # 如果过滤后为空，返回错误消息
        if not filtered_seq:
            return "生成失败：没有有效的翻译结果"
            
        # 解码为文本
        try:
            translated_text = trg_sp.decode_ids(filtered_seq)
            return translated_text
        except Exception as e:
            logger.error(f"解码出错: {str(e)}")
            return f"解码错误: {str(e)}"

    def make_mask(self, src_input, trg_input):
        e_mask = (src_input != pad_id).unsqueeze(1)
        d_mask = (trg_input != pad_id).unsqueeze(1)
        nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool).to(device)
        nopeak_mask = torch.tril(nopeak_mask)
        d_mask = d_mask & nopeak_mask
        return e_mask, d_mask

    def _test_sentencepiece(self, sp):
        """测试SentencePiece解码器是否正常工作"""
        # 验证基本功能是否工作
        try:
            # 验证特殊token
            for name, token_id in {"PAD": pad_id, "SOS": sos_id, "EOS": eos_id, "UNK": unk_id}.items():
                token_text = sp.id_to_piece(token_id)
            
            # 验证基本解码功能
            vocab_size = sp.get_piece_size()
            
            # 简单解码测试
            test_ids = list(range(4, 8))
            test_text = sp.decode_ids(test_ids)
            
            # 编码-解码一致性测试
            test_word = "test"
            encoded = sp.encode_as_ids(test_word)
            decoded = sp.decode_ids(encoded)
            
            # 如果所有功能正常工作，返回
            return
        except Exception as e:
            logger.error(f"SentencePiece验证失败: {str(e)}")
            return 