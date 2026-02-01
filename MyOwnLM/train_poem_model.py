"""
诗歌语言模型训练脚本
基于Transformer的古诗生成模型训练
支持检查点保存和恢复训练
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
from collections import Counter
import argparse
import json


# ==================== 数据加载和预处理 ====================

def load_poetry_data(csv_path):
    """从CSV文件加载诗歌数据"""
    df = pd.read_csv(csv_path)
    poems = [str(line).strip() for line in df["text1"].tolist() if len(str(line).strip()) > 0]
    return poems


def format_poem(poems):
    """格式化诗歌：替换标点符号为空格"""
    formatted_poems = []
    for poem in poems:
        line = poem.replace('，', ' ').replace('。', ' ').replace('？', ' ').replace('！', ' ')
        line = line.strip()
        formatted_poems.append(line)
    return formatted_poems


def build_char_vocab(poems, min_freq=2, max_size=5000):
    """构建字符级词汇表"""
    char_counter = Counter()

    for poem in poems:
        # 统计每个字符
        for char in poem:
            if char.strip():  # 跳过空白字符
                char_counter[char] += 1

    # 按频率排序，保留高频字符
    sorted_chars = sorted(char_counter.items(), key=lambda x: x[1], reverse=True)

    # 构建词汇表
    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[BOS]": 2,  # 开始标记
        "[EOS]": 3,  # 结束标记
        "[SEP]": 4,  # 分隔标记（诗句间）
        "[MASK]": 5,  # 掩码标记
    }

    # 添加高频字符（跳过逗号和句号，它们已被替换为[SEP]）
    idx = len(vocab)
    for char, freq in sorted_chars:
        if char in ['，', '。']:
            continue
        if freq >= min_freq and idx < max_size:
            vocab[char] = idx
            idx += 1

    return vocab, char_counter


class Tokenizer:
    """字符级分词器（用于诗歌文本编码和解码）"""

    def __init__(self, vocab):
        self.vocab = vocab
        self.inv_vocab = {idx: char for char, idx in vocab.items()}

    def encode(self, text):
        """将文本编码为token ID序列"""
        result = [self.vocab["[BOS]"]]
        for s in text.split():
            for char in s:
                result.append(self.vocab.get(char, self.vocab["[UNK]"]))
            result.append(self.vocab["[SEP]"])
        result[-1] = self.vocab["[EOS]"]
        return result

    def decode(self, token_ids):
        """将token ID序列解码为文本"""
        chars = []
        for idx in token_ids:
            if idx < 6:  # 特殊token
                chars.append(' ')
                continue
            char = self.inv_vocab.get(idx, "[UNK]")
            chars.append(char)
        return ''.join(chars).strip()


# ==================== 数据集 ====================

def pad_sequence(tokens, pad_idx=0, max_len=1024):
    """填充或截断序列到指定长度"""
    if len(tokens) >= max_len:
        return tokens[:max_len]
    else:
        return tokens + [pad_idx] * (max_len - len(tokens))


class PoemDataset(Dataset):
    """诗歌数据集（用于语言模型训练，包含固定长度序列）"""

    def __init__(self, token_sequences, max_len=1024, pad_idx=0):
        self.sequences = []
        for tokens in token_sequences:
            padded = pad_sequence(tokens, pad_idx, max_len)
            self.sequences.append(torch.tensor(padded, dtype=torch.long))
        self.pad_idx = pad_idx

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


# ==================== 模型定义 ====================

class PositionalEncoding(nn.Module):
    """Sin/Cos 位置编码"""

    def __init__(self, embed_dim, max_seq_len=1024, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() *
                            (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        if embed_dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """多头注意力机制（支持因果掩码和填充掩码）"""

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim必须能被num_heads整除"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.linear_q = nn.Linear(embed_dim, embed_dim)
        self.linear_k = nn.Linear(embed_dim, embed_dim)
        self.linear_v = nn.Linear(embed_dim, embed_dim)
        self.linear_out = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 线性变换并分头
        Q = self.linear_q(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.linear_k(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.linear_v(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力到值
        context = torch.matmul(attn_weights, V)

        # 拼接多头结果
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.linear_out(context)

        return output


class FeedForwardNetwork(nn.Module):
    """前馈网络"""

    def __init__(self, embed_dim, ffn_dim, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerDecoderLayer(nn.Module):
    """Transformer解码器层（用于GPT式解码器模型）"""

    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim, dropout)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 自注意力 + 残差连接
        attn_output = self.attention(x, x, x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # 前馈网络 + 残差连接
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)

        return x


class TransformerModel(nn.Module):
    """Transformer语言模型（GPT式解码器架构）"""

    def __init__(self, vocab_size, embed_dim, max_seq_len, num_layers,
                 num_heads, ffn_dim, dropout=0.1, padding_idx=0):
        super(TransformerModel, self).__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.padding_idx = padding_idx

        # 嵌入层
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len, dropout)

        # Transformer解码器层
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        # 输出层
        self.output_projection = nn.Linear(embed_dim, vocab_size)

        # 初始化参数
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_causal_mask(self, seq_len, device):
        """生成因果掩码（下三角矩阵）"""
        # 创建形状为 (seq_len, seq_len) 的下三角矩阵，对角线以上为0，对角线及以下为1
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        # 调整为 (1, 1, seq_len, seq_len) 形状，以便进行广播
        return mask.view(1, 1, seq_len, seq_len)

    def forward(self, input_ids, mask=None):
        """
        Args:
            input_ids: (batch_size, seq_len)
            mask: (batch_size, 1, 1, seq_len) or None (padding mask)
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape

        # 嵌入
        x = self.token_embedding(input_ids)
        x = self.positional_encoding(x)

        # 生成因果掩码用于因果注意力（防止当前位置关注未来位置）
        causal_mask = self._generate_causal_mask(seq_len, x.device)

        # 合并因果掩码和填充掩码（如果提供填充掩码，结合两者以确保自回归特性并忽略填充token）
        if mask is not None:
            # 将填充掩码从 (batch_size, 1, 1, seq_len) 广播为 (batch_size, 1, seq_len, seq_len)
            # 以便与因果掩码结合
            # 填充掩码中，0 表示填充位置（应屏蔽），1 表示有效位置
            padding_mask_expanded = mask.expand(-1, -1, seq_len, -1)

            # 因果掩码中，0 表示未来位置（应屏蔽），1 表示过去及当前位置
            # 在注意力机制中，最终掩码计算为：scores.masked_fill(mask == 0, float('-inf'))
            # 因此，我们需要确保：
            # - 填充位置（padding_mask 为 0）和未来位置（causal_mask 为 0）都设为 -inf
            # - 过去/当前位置且非填充（两者都为 1）保持为 0
            # 通过逐元素相乘，任一为 0 时结果为 0，将在注意力中被替换为 -inf
            combined_mask = causal_mask * padding_mask_expanded
        else:
            combined_mask = causal_mask

        # 通过Transformer解码器层
        for layer in self.decoder_layers:
            x = layer(x, combined_mask)

        # 输出投影
        logits = self.output_projection(x)

        return logits


# ==================== 配置类 ====================

class Config:
    """模型配置类（用于 GPT 式解码器 Transformer 语言模型）"""

    def __init__(self, vocab_size):
        # 模型参数
        self.vocab_size = vocab_size
        self.embed_dim = 768
        self.max_seq_len = 64
        self.num_layers = 6
        self.num_heads = 8
        self.ffn_dim = 1024
        self.dropout = 0.3
        self.padding_idx = 0

        # 训练参数
        self.batch_size = 128
        self.num_epochs = 100
        self.learning_rate = 1e-4  
        self.weight_decay = 0.01
        self.grad_clip = 1.0  
        self.t_max = 100
        self.eta_min = 1e-7

        # 数据集参数
        self.train_val_split = 0.95

        # 设备
        self.device = torch.device('mps' if torch.backends.mps.is_available() else
                                   ('cuda' if torch.cuda.is_available() else 'cpu'))

    def save(self, path):
        """保存配置到文件"""
        config_dict = {k: str(v) if not isinstance(v, (int, float, bool)) else v
                      for k, v in self.__dict__.items()}
        config_dict['device'] = str(self.device)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path, vocab_size):
        """从文件加载配置"""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        config = cls(vocab_size)
        for k, v in config_dict.items():
            if k == 'device':
                continue
            if isinstance(getattr(config, k, None), int):
                setattr(config, k, int(v))
            elif isinstance(getattr(config, k, None), float):
                setattr(config, k, float(v))
            else:
                setattr(config, k, v)
        return config


# ==================== 训练函数 ====================

def train_epoch(model, train_loader, optimizer, criterion, config, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    batch_count = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)

    for batch in progress_bar:
        input_ids = batch.to(config.device)

        # 前向传播
        logits = model(input_ids)  # (batch_size, seq_len, vocab_size)

        # 计算损失（预测下一个token）
        loss = criterion(
            logits[:, :-1, :].reshape(-1, config.vocab_size),  # 使用前seq_len-1个预测
            input_ids[:, 1:].reshape(-1)  # 目标是后seq_len-1个token
        )

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1

        # 更新进度条显示
        avg_loss = total_loss / batch_count
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{avg_loss:.4f}'
        })

    avg_loss = total_loss / batch_count
    return avg_loss


def evaluate(model, val_loader, criterion, config):
    """评估模型"""
    model.eval()
    total_loss = 0
    batch_count = 0

    progress_bar = tqdm(val_loader, desc="评估中", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch.to(config.device)
            logits = model(input_ids)

            loss = criterion(
                logits[:, :-1, :].reshape(-1, config.vocab_size),
                input_ids[:, 1:].reshape(-1)
            )

            total_loss += loss.item()
            batch_count += 1
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / batch_count
    return avg_loss


def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss,
                    vocab, config, checkpoint_path, is_best=False):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'vocab': vocab,
        'config': config.__dict__
    }

    # 保存检查点
    torch.save(checkpoint, checkpoint_path)
    print(f"  已保存检查点: {checkpoint_path}")

    # 如果是最好模型，额外保存一份
    if is_best:
        best_path = checkpoint_path.replace('.pt', '_best.pt')
        torch.save(checkpoint, best_path)
        print(f"  已保存最佳模型: {best_path}")


def main():
    parser = argparse.ArgumentParser(description='训练诗歌语言模型')
    parser.add_argument('--data_dir', type=str, default='../chinese-poetry',
                       help='数据目录路径')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                       help='检查点保存目录')
    parser.add_argument('--num_epochs', type=int, default=None,
                       help='训练轮数（覆盖配置）')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='批次大小（覆盖配置）')
    parser.add_argument('--lr', type=float, default=None,
                       help='学习率（覆盖配置）')
    parser.add_argument('--resume', type=str, default=None,
                       help='从检查点恢复训练')
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置默认dtype
    torch.set_default_dtype(torch.float32)

    print("=" * 60)
    print("诗歌语言模型训练")
    print("=" * 60)

    # 加载数据
    print("\n加载诗歌数据...")
    train_csv = os.path.join(args.data_dir, 'train.csv')
    poems = load_poetry_data(train_csv)
    print(f"  总诗歌数: {len(poems)}")

    # 格式化诗歌
    poems = format_poem(poems)

    # 构建词汇表
    print("\n构建词汇表...")
    char_vocab, char_counter = build_char_vocab(poems, min_freq=2, max_size=4000)
    print(f"  词汇表大小: {len(char_vocab)}")

    # 保存词汇表
    vocab_path = os.path.join(args.output_dir, 'vocab.json')
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(char_vocab, f, ensure_ascii=False, indent=2)
    print(f"  已保存词汇表: {vocab_path}")

    # 创建分词器
    tokenizer = Tokenizer(char_vocab)

    # 编码数据
    print("\n编码训练数据...")
    train_tokens = [tokenizer.encode(poem) for poem in poems]

    # 创建配置
    config = Config(len(char_vocab))

    # 覆盖配置参数（如果指定）
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr

    print(f"\n使用设备: {config.device}")
    config.print_model_info() if hasattr(config, 'print_model_info') else None

    # 创建数据集
    print("\n创建数据集...")
    dataset = PoemDataset(train_tokens, max_len=config.max_seq_len, pad_idx=config.padding_idx)
    print(f"  数据集大小: {len(dataset)}")

    # 分割数据集
    train_size = int(config.train_val_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # 固定随机种子
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                             shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                           shuffle=False, drop_last=True)

    print(f"  训练数据: {len(train_dataset)} 样本")
    print(f"  验证数据: {len(val_dataset)} 样本")

    # 创建模型
    print("\n初始化模型...")
    model = TransformerModel(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        max_seq_len=config.max_seq_len,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        ffn_dim=config.ffn_dim,
        dropout=config.dropout,
        padding_idx=config.padding_idx
    ).to(config.device)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params / 1e6:.2f}M")
    print(f"  可训练参数: {trainable_params / 1e6:.2f}M")

    # 优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate,
                           weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=config.padding_idx)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.t_max,
                                                    eta_min=config.eta_min)

    # 恢复训练（如果需要）
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        print(f"\n从检查点恢复: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_loss']
        print(f"  从epoch {start_epoch}开始继续训练")
        print(f"  最佳验证损失: {best_val_loss:.4f}")

    # 保存配置
    config_path = os.path.join(args.output_dir, 'config.json')
    config.save(config_path)
    print(f"\n已保存配置: {config_path}")

    # 训练循环
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)

    train_losses = []
    val_losses = []

    for epoch in range(start_epoch, start_epoch + config.num_epochs):
        epoch_num = epoch + 1

        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config, epoch_num)

        # 评估
        val_loss = evaluate(model, val_loader, criterion, config)

        # 更新学习率
        scheduler.step()

        # 记录损失
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"\nEpoch {epoch_num}/{start_epoch + config.num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        # 保存检查点（每10轮保存一次）
        if epoch_num % 10 == 0 or val_loss < best_val_loss:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch_num:03d}.pt')
            save_checkpoint(model, optimizer, scheduler, epoch_num, train_loss, val_loss,
                          char_vocab, config, checkpoint_path, is_best=(val_loss < best_val_loss))

        # 更新最佳损失
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # 保存最后的训练状态
        last_checkpoint = os.path.join(args.output_dir, 'checkpoint_last.pt')
        torch.save({
            'epoch': epoch_num,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'vocab': char_vocab,
            'config': config.__dict__
        }, last_checkpoint)

    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print("=" * 60)

    # 保存最终模型
    final_model_path = os.path.join(args.output_dir, 'model_final.pt')
    torch.save(model.state_dict(), final_model_path)
    print(f"\n已保存最终模型: {final_model_path}")


if __name__ == '__main__':
    main()