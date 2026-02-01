"""
诗歌语言模型可视化脚本
用于生成诗歌并评估模型性能
"""

import torch
import torch.nn.functional as F
import argparse
import json
import sys
import os

# 导入必要的类
from train_poem_model import TransformerModel, Config, Tokenizer, format_poem, build_char_vocab


def load_model(checkpoint_path, vocab_path, device=None):
    """加载模型和词汇表"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载词汇表
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    # 创建配置
    config = Config(len(vocab))
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

    # 加载权重
    if checkpoint_path.endswith('.pt'):
        # 加载检查点
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded from checkpoint: {checkpoint_path}")
            else:
                model.load_state_dict(checkpoint)
                print(f"Loaded model weights: {checkpoint_path}")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            print("Using random weights")

    return model, vocab, config


def generate_poem(model, vocab, config, prompt="", max_length=100,
                 temperature=0.8, top_k=50, repetition_penalty=1.0):
    """生成诗歌"""
    model.eval()
    tokenizer = Tokenizer(vocab)

    # 处理输入
    if prompt:
        input_tokens = tokenizer.encode(prompt)
    else:
        input_tokens = [vocab["[BOS]"]]

    generated = input_tokens.copy()

    # 用于重复惩罚的token计数
    token_counts = {}

    with torch.no_grad():
        for _ in range(max_length):
            # 获取当前输入
            current_input = torch.tensor(
                [generated[-config.max_seq_len:]],
                dtype=torch.long,
                device=config.device
            )

            # 前向传播
            logits = model(current_input)
            next_token_logits = logits[0, -1, :] / temperature

            # 应用重复惩罚
            if repetition_penalty != 1.0:
                for token_id in set(generated):
                    if token_counts.get(token_id, 0) > 0:
                        next_token_logits[token_id] /= repetition_penalty

            # Top-k采样
            top_k_logits, top_k_indices = torch.topk(
                next_token_logits,
                min(top_k, logits.size(-1))
            )
            next_token_probs = F.softmax(top_k_logits, dim=-1)
            next_token_idx = top_k_indices[torch.multinomial(next_token_probs, 1)].item()

            # 更新token计数
            token_counts[next_token_idx] = token_counts.get(next_token_idx, 0) + 1

            generated.append(next_token_idx)

            # 如果生成了结束标记，停止生成
            if next_token_idx == vocab.get("[EOS]", 3):
                break

    # 解码
    generated_text = tokenizer.decode(generated)

    return generated_text


def calculate_perplexity(model, test_tokens, config):
    """计算困惑度"""
    model.eval()
    total_loss = 0
    num_tokens = 0

    criterion = torch.nn.CrossEntropyLoss(ignore_index=config.padding_idx, reduction='sum')

    with torch.no_grad():
        # 批量处理
        batch_size = 64
        for i in range(0, len(test_tokens), batch_size):
            batch_tokens = test_tokens[i:i+batch_size]
            if not batch_tokens:
                continue

            # 填充批次
            max_len = min(max(len(t) for t in batch_tokens), config.max_seq_len)
            batch_tensor = torch.zeros((len(batch_tokens), max_len), dtype=torch.long)
            for j, tokens in enumerate(batch_tokens):
                if len(tokens) > max_len:
                    batch_tensor[j] = torch.tensor(tokens[:max_len], dtype=torch.long)
                else:
                    batch_tensor[j, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)

            batch_tensor = batch_tensor.to(config.device)

            # 前向传播
            logits = model(batch_tensor)

            # 计算损失
            loss = criterion(
                logits[:, :-1, :].reshape(-1, config.vocab_size),
                batch_tensor[:, 1:].reshape(-1)
            )

            # 计算有效token数量（排除padding）
            num_predictions = (batch_tensor[:, 1:] != config.padding_idx).sum().item()

            total_loss += loss.item()
            num_tokens += num_predictions

    if num_tokens > 0:
        avg_loss = total_loss / num_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        return perplexity
    else:
        return float('inf')


def main():
    parser = argparse.ArgumentParser(description='诗歌语言模型生成脚本')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/checkpoint_last.pt',
                       help='检查点文件路径')
    parser.add_argument('--vocab', type=str, default='./checkpoints/vocab.json',
                       help='词汇表文件路径')
    parser.add_argument('--prompt', type=str, default='',
                       help='生成提示（如：春 风）')
    parser.add_argument('--max_length', type=int, default=100,
                       help='最大生成长度')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='采样温度（越高越随机）')
    parser.add_argument('--top_k', type=int, default=50,
                       help='Top-k采样参数')
    parser.add_argument('--repetition_penalty', type=float, default=1.0,
                       help='重复惩罚（>1.0惩罚重复，<1.0鼓励重复）')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='生成样本数量')
    parser.add_argument('--test_file', type=str, default='',
                       help='测试文件路径（用于计算困惑度）')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载模型
    model, vocab, config = load_model(args.checkpoint, args.vocab, device)
    print(f"模型加载完成，词汇表大小: {len(vocab)}")

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params / 1e6:.2f}M")

    # 如果有测试文件，计算困惑度
    if args.test_file and os.path.exists(args.test_file):
        from train_poem_model import load_poetry_data
        print(f"\n计算测试集困惑度...")
        test_poems = load_poetry_data(args.test_file)
        tokenizer = Tokenizer(vocab)
        test_tokens = [tokenizer.encode(poem) for poem in test_poems[:1000]]  # 使用前1000个
        perplexity = calculate_perplexity(model, test_tokens, config)
        print(f"测试集困惑度: {perplexity:.4f}")

    # 生成诗歌
    print("\n" + "=" * 60)
    print("诗歌生成示例")
    print("=" * 60)

    for i in range(args.num_samples):
        if args.prompt:
            prompt_text = args.prompt
        else:
            # 使用不同的起始词
            prompts = ["", "难忘旧情", "青楼一夜", "春风得意", "借酒浇愁", "月下独酌"]
            prompt_text = prompts[i % len(prompts)]

        generated = generate_poem(
            model, vocab, config,
            prompt=prompt_text,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty
        )

        print(f"\n示例 {i+1} (提示: '{prompt_text if prompt_text else '无'}'):")
        print(f"  生成: {generated}")

    # 交互模式
    print("\n" + "=" * 60)
    print("交互模式（输入提示生成诗歌，输入 'quit' 退出）")
    print("=" * 60)

    while True:
        user_input = input("\n输入提示 (或 'quit'): ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break

        # 格式化输入（替换标点）
        formatted_input = user_input.replace('，', ' ').replace('。', ' ')

        generated = generate_poem(
            model, vocab, config,
            prompt=formatted_input,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty
        )

        print(f"\n生成结果:")
        print(f"  {generated}")


if __name__ == '__main__':
    main()