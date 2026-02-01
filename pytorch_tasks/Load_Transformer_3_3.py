import torch
from torch import nn
import math
import sys
import os

# 设备设置
device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
print(f"Using device: {device}")

# 导入模型类和相关函数
from Transformer_3_3 import (
    NumTransformer, 
    PositionalEncoding, 
    NumTransformerEncoder, 
    NumTransformerDecoder, 
    create_tgt_mask,
    get_vocab,
    idx2token,
    vocab,
    User_pred
)

# 将模型类设置到 __main__ 模块中
import __main__
__main__.NumTransformer = NumTransformer
__main__.PositionalEncoding = PositionalEncoding
__main__.NumTransformerEncoder = NumTransformerEncoder
__main__.NumTransformerDecoder = NumTransformerDecoder
__main__.create_tgt_mask = create_tgt_mask

# 现在可以安全地加载模型了
model_path = "./pytorch_tasks/save_models/model_Transformer_3_3.pth"
if not os.path.exists(model_path):
    model_path = "save_models/model_Transformer_3_3.pth"

try:
    model = torch.load(model_path, map_location=device, weights_only=False)
    print("模型加载成功!")
    model.to(device)
    model.eval()
except Exception as e:
    print(f"加载模型失败: {e}")
    print("请确保模型文件存在且路径正确")
    sys.exit(1)

def prepare_input_sequence(n1, n2):
    """准备输入序列，格式与训练时相同"""
    # 构建源序列字符串
    source_str = f"{n1}+{n2}"
    
    # 按照训练时的格式构建序列：[<bos>] + 数字 + [<eos>] + <pad>填充
    src_len = 9  # 3+1+3+2 = 9 (3位数+加号+3位数+2个特殊标记)
    
    # 构建源序列
    source_seq = [vocab['<bos>']] + [vocab[c] for c in source_str] + [vocab['<eos>']]
    
    # 填充到固定长度
    if len(source_seq) < src_len:
        source_seq = source_seq + [vocab['<pad>']] * (src_len - len(source_seq))
    
    # 构建目标序列（占位符，实际不会用于生成）
    tgt_len = 6  # max(3,3)+1+2 = 3+1+2=6
    target_str = str(n1 + n2)
    target_seq = [vocab['<bos>']] + [vocab[c] for c in target_str] + [vocab['<eos>']]
    if len(target_seq) < tgt_len:
        target_seq = target_seq + [vocab['<pad>']] * (tgt_len - len(target_seq))
    
    # 转换为张量并添加批次维度
    src_tensor = torch.tensor([source_seq], dtype=torch.long)
    tgt_tensor = torch.tensor([target_seq], dtype=torch.long)
    
    return src_tensor, tgt_tensor

all_data = []
def check_all():
    for n1 in range(100,1000):
        for n2 in range(100, 1000):
            print(n1,"  ",n2)
            if n1 + n2 < 1000:
                continue
            s, t = prepare_input_sequence(n1, n2)
            s, t = s.to(device), t.to(device)
            all_data.append((s, t))
    _, targets, preds = User_pred(model, all_data)
    total = len(targets)
    correct = sum(a == b for a, b in zip(targets, preds))
    return total, correct

total, correct = check_all()
print(f'total : {total}')
print(f'correct : {correct}')


while True:
    try:
        user_input = input("请输入两个三位数相加（格式: 123+456），或输入'quit'退出: ").strip()
        
        if user_input.lower() == 'quit':
            break
        
        if '+' not in user_input:
            print("错误: 输入格式应为 '123+456'")
            continue
        n1_str, n2_str = user_input.split('+')
        n1 = int(n1_str.strip())
        n2 = int(n2_str.strip())
        
        # 准备输入数据
        src, tgt = prepare_input_sequence(n1, n2)
        src, tgt = src.to(device), tgt.to(device)

        data = [(src, tgt)]
        # 调用User_pred函数
        source, target, preds = User_pred(model, data)
        
        print(f"问题: {source}")
        print(f"正确答案: {target}")
        print(f"模型预测: {preds}")
        
            
        print("-" * 50)
        
    except ValueError:
        print("错误: 请输入有效的数字，如 '123+456'")
    except KeyboardInterrupt:
        print("\n程序已退出")
        break
