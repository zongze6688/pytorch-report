"""
诗歌语言模型继续训练脚本
从检查点恢复并继续训练模型
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import argparse
import json

# 导入训练脚本中的相同模块
from train_poem_model import (
    load_poetry_data, format_poem, build_char_vocab, Tokenizer,
    pad_sequence, PoemDataset, TransformerModel, Config,
    train_epoch, evaluate, save_checkpoint
)


def load_checkpoint(checkpoint_path, device=None):
    """加载检查点"""
    if device is None:
        device = torch.device('mps' if torch.backends.mps.is_available() else
                             ('cuda' if torch.cuda.is_available() else 'cpu'))

    print(f"加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    return checkpoint, device


def create_model_from_checkpoint(checkpoint, device):
    """从检查点创建模型"""
    # 从检查点获取配置
    config_dict = checkpoint['config']
    vocab = checkpoint['vocab']

    # 创建配置对象
    config = Config(len(vocab))
    for k, v in config_dict.items():
        if k == 'device':
            continue
        if hasattr(config, k):
            if isinstance(getattr(config, k), int):
                setattr(config, k, int(v))
            elif isinstance(getattr(config, k), float):
                setattr(config, k, float(v))
            else:
                setattr(config, k, v)

    config.device = device

    # 创建模型
    model = TransformerModel(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        max_seq_len=config.max_seq_len,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        ffn_dim=config.ffn_dim,
        dropout=config.dropout,
        padding_idx=config.padding_idx
    ).to(device)

    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, config, vocab


def main():
    parser = argparse.ArgumentParser(description='继续训练诗歌语言模型')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='检查点文件路径')
    parser.add_argument('--data_dir', type=str, default='../chinese-poetry',
                       help='数据目录路径')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                       help='检查点保存目录')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='继续训练的轮数')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='批次大小（覆盖配置）')
    parser.add_argument('--lr', type=float, default=None,
                       help='学习率（覆盖配置）')
    parser.add_argument('--new_lr', action='store_true',
                       help='使用新的学习率（忽略检查点中的学习率）')
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置默认dtype
    torch.set_default_dtype(torch.float32)

    print("=" * 60)
    print("继续训练诗歌语言模型")
    print("=" * 60)

    # 加载检查点
    checkpoint, device = load_checkpoint(args.checkpoint)
    vocab = checkpoint['vocab']
    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['val_loss']

    print(f"\n检查点信息:")
    print(f"  训练轮数: {start_epoch}")
    print(f"  训练损失: {checkpoint['train_loss']:.4f}")
    print(f"  验证损失: {checkpoint['val_loss']:.4f}")

    # 创建模型和配置
    model, config, vocab = create_model_from_checkpoint(checkpoint, device)

    # 覆盖配置参数（如果指定）
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr

    print(f"\n使用设备: {config.device}")
    print(f"词汇表大小: {config.vocab_size}")

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params / 1e6:.2f}M")

    # 加载训练数据
    print("\n加载诗歌数据...")
    train_csv = os.path.join(args.data_dir, 'train.csv')
    poems = load_poetry_data(train_csv)
    poems = format_poem(poems)
    print(f"  总诗歌数: {len(poems)}")

    # 创建分词器
    tokenizer = Tokenizer(vocab)

    # 编码数据
    print("\n编码训练数据...")
    train_tokens = [tokenizer.encode(poem) for poem in poems]

    # 创建数据集
    print("\n创建数据集...")
    dataset = PoemDataset(train_tokens, max_len=config.max_seq_len, pad_idx=config.padding_idx)
    print(f"  数据集大小: {len(dataset)}")

    # 分割数据集（使用相同的随机种子）
    train_size = int(config.train_val_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                             shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                           shuffle=False, drop_last=True)

    print(f"  训练数据: {len(train_dataset)} 样本")
    print(f"  验证数据: {len(val_dataset)} 样本")

    # 优化器和损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=config.padding_idx)

    # 如果指定使用新的学习率，重新创建优化器
    if args.new_lr or args.lr:
        print(f"\n使用新学习率: {config.learning_rate}")
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate,
                               weight_decay=config.weight_decay)
    else:
        # 从检查点恢复优化器
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate,
                               weight_decay=config.weight_decay)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"\n恢复优化器状态")

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.t_max,
                                                    eta_min=config.eta_min)
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # 训练循环
    print("\n" + "=" * 60)
    print(f"继续训练 {args.num_epochs} 轮")
    print("=" * 60)

    train_losses = []
    val_losses = []

    for epoch in range(args.num_epochs):
        epoch_num = start_epoch + epoch + 1

        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config, epoch_num)

        # 评估
        val_loss = evaluate(model, val_loader, criterion, config)

        # 更新学习率
        scheduler.step()

        # 记录损失
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"\nEpoch {epoch_num}/{start_epoch + args.num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        # 保存检查点（每10轮保存一次）
        if (epoch + 1) % 10 == 0 or val_loss < best_val_loss:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch_num:03d}.pt')
            save_checkpoint(model, optimizer, scheduler, epoch_num, train_loss, val_loss,
                          vocab, config, checkpoint_path, is_best=(val_loss < best_val_loss))

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
            'vocab': vocab,
            'config': config.__dict__
        }, last_checkpoint)

    print("\n" + "=" * 60)
    print("继续训练完成!")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print("=" * 60)

    # 保存最终模型
    final_model_path = os.path.join(args.output_dir, 'model_resumed_final.pt')
    torch.save(model.state_dict(), final_model_path)
    print(f"\n已保存最终模型: {final_model_path}")


if __name__ == '__main__':
    main()