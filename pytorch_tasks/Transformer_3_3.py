import torch
from torch import nn
import math
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import os
"""
训练结果，展示最后5个epoch：
--------------------------------------------------
Epoch [46/50], Validation Loss: 0.8847
--------------------------------------------------
Source: 914+443
Target: 1357
Prediction: 1347
------------------------------
Source: 616+557
Target: 1173
Prediction: 1272
------------------------------
Source: 932+133
Target: 1065
Prediction: 1146
------------------------------
Source: 404+764
Target: 1168
Prediction: 1150
------------------------------
Source: 996+317
Target: 1313
Prediction: 1160
------------------------------
Source: 571+776
Target: 1347
Prediction: 1392
------------------------------
--------------------------------------------------
Epoch [47/50], Validation Loss: 0.8774
--------------------------------------------------
Source: 615+630
Target: 1245
Prediction: 1111
------------------------------
Source: 539+468
Target: 1007
Prediction: 1430
------------------------------
Source: 634+544
Target: 1178
Prediction: 1088
------------------------------
Source: 812+513
Target: 1325
Prediction: 1028
------------------------------
--------------------------------------------------
Epoch [48/50], Validation Loss: 0.8927
--------------------------------------------------
Source: 603+739
Target: 1342
Prediction: 1333
------------------------------
Source: 759+606
Target: 1365
Prediction: 1365
------------------------------
Source: 941+943
Target: 1884
Prediction: 1309
------------------------------
Source: 808+675
Target: 1483
Prediction: 1383
------------------------------
Source: 676+627
Target: 1303
Prediction: 1339
------------------------------
--------------------------------------------------
Epoch [49/50], Validation Loss: 0.8726
--------------------------------------------------
Source: 346+993
Target: 1339
Prediction: 1330
------------------------------
Source: 972+873
Target: 1845
Prediction: 1161
------------------------------
Source: 951+485
Target: 1436
Prediction: 140
------------------------------
Source: 554+979
Target: 1533
Prediction: 1443
------------------------------
Source: 146+982
Target: 1128
Prediction: 110
------------------------------
Source: 383+839
Target: 1222
Prediction: 1222
------------------------------
Source: 414+966
Target: 1380
Prediction: 110
------------------------------
--------------------------------------------------
Epoch [50/50], Validation Loss: 0.9052
--------------------------------------------------
Source: 987+329
Target: 1316
Prediction: 1270
------------------------------
Source: 125+957
Target: 1082
Prediction: 1172
------------------------------
Source: 831+763
Target: 1594
Prediction: 1099
------------------------------
Source: 849+801
Target: 1650
Prediction: 1398
------------------------------
Source: 434+568
Target: 1002
Prediction: 1029



"""



# 设备设置
device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
print(f"Using device: {device}")
#TODO : 创建3+3的加法数据
#TODO : 正确地创建数据
#TODO : 创建一个transformer模型
#TODO : 训练模型训练

def get_vocab():
    tokens = ['<pad>', '<bos>', '<eos>'] + [str(i) for i in range(10)] + ['+']
    vocab = {token: idx for idx, token in enumerate(tokens)}
    # 创建反向字典
    idx2token = {idx: token for token, idx in vocab.items()}
    return vocab, idx2token
vocab, idx2token = get_vocab()

def get_full_data():
    def get_num_dict(d1, d2, size):
        num_dict = {}
        for i in range(size):
            n1 = random.randint(10 ** (d1 - 1), 10 ** d1 - 1)
            n2 = random.randint(10 ** (d2 - 1), 10 ** d2 - 1)
            if n1 + n2 < 1000:
                continue
            num_dict[f'{n1}+{n2}'] = str(n1 + n2)
        return num_dict

    train_num_dict = get_num_dict(3, 3, 20000)
    val_num_dict = get_num_dict(3, 3, 1000)
    test_num_dict = get_num_dict(3, 3, 10)



    def create_data(num_dict, vocab, d1, d2):
        src_len = d1 + 1 + d2 + 2 
        tgt_len = max(d1, d2) + 1 + 2 
        features = torch.zeros(size=(len(num_dict), src_len), dtype=torch.long)
        labels = torch.zeros(size=(len(num_dict), tgt_len), dtype=torch.long)
        for idx, (source, target) in enumerate(num_dict.items()):
            source_seq = [vocab['<bos>']] + [vocab[c] for c in source] + [vocab['<eos>']]
            target_seq = [vocab['<bos>']] + [vocab[c] for c in target] + [vocab['<eos>']]
            features[idx, :len(source_seq)] = torch.tensor(source_seq, dtype=torch.long)
            labels[idx, :len(target_seq)] = torch.tensor(target_seq, dtype=torch.long)
        
        return features, labels
    def create_dataloader(features, labels, batch_size):
        dataset = TensorDataset(features, labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
    batch_size = 64
    train_src, train_tgt = create_data(train_num_dict, vocab, 3, 3)
    val_src, val_tgt = create_data(val_num_dict, vocab, 3, 3)
    test_src, test_tgt = create_data(test_num_dict, vocab, 3, 3)
    train_loader = create_dataloader(train_src, train_tgt, batch_size=batch_size)
    val_loader = create_dataloader(val_src, val_tgt, batch_size=batch_size)
    test_loader = create_dataloader(test_src, test_tgt, batch_size=1)
    return train_loader, val_loader, test_loader


class PositionalEncoding(nn.Module):
    """位置编码 - 为Transformer提供位置信息"""
    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        
        # 注册为buffer（不参与训练）
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]
    
class NumTransformerEncoder(nn.Module):
    def __init__(self, D, N_h, N_l, vocab_size): # D: embedding dimension, N_h: number of heads, N_l: number of layers
        super(NumTransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, D)
        self.pos_coder = PositionalEncoding(D)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=D, nhead=N_h),
            num_layers=N_l,
        )
    def forward(self, X):
        X = self.embedding(X) * math.sqrt(self.embedding.embedding_dim)
        X = self.pos_coder(X)
        X = X.permute(1, 0, 2) 
        X = self.encoder(X)
        X = X.permute(1, 0, 2)  
        return X
    

class NumTransformerDecoder(nn.Module):
    def __init__(self, D, N_h, N_l, vocab_size):
        super(NumTransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, D)
        self.pos_coder = PositionalEncoding(D)
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(d_model=D, nhead=N_h),
            num_layers=N_l,
        )
        self.output_layer = nn.Linear(D, vocab_size)
    def init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    def forward(self, memory, tgt, 
                    tgt_mask=None):
        if tgt_mask is None:
            tgt_mask = create_tgt_mask(tgt.size(1))
        tgt = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
        tgt = self.pos_coder(tgt)
        tgt = tgt.permute(1, 0, 2)
        output = self.decoder(tgt, memory.permute(1, 0, 2), tgt_mask=tgt_mask)
        output = output.permute(1, 0, 2)
        output = self.output_layer(output)
        return output
    def generate(self, memory, max_len=10): 
        self.eval()
        batch_size = memory.size(0)
        generate_seq = torch.zeros((batch_size, max_len), dtype=torch.long).to(memory.device)
        generate_seq[:, 0] = vocab['<bos>']
        for i in range(max_len-1):
            tgt_mask = create_tgt_mask(i+1)
            out = self.forward(memory, generate_seq[:, :i+1], tgt_mask=tgt_mask)
            next_token = out[:, -1, :].argmax(dim=-1)
            generate_seq[:, i+1] = next_token
        return generate_seq



def create_tgt_mask(size):
    mask = torch.triu(torch.ones(size, size) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask    

class NumTransformer(nn.Module):
    def __init__(self, D, N_h, N_l, vocab_size):
        super(NumTransformer, self).__init__()
        self.encoder = NumTransformerEncoder(D, N_h, N_l, vocab_size)
        self.decoder = NumTransformerDecoder(D, N_h, N_l, vocab_size)
        self.decoder.init_weight()

    def forward(self, src, tgt, tgt_mask=None):
        if tgt_mask is None:
            tgt_mask = create_tgt_mask(tgt.size(1))
        memory = self.encoder(src)
        output = self.decoder(memory, tgt, tgt_mask=tgt_mask)
        return output
    
    def generate(self, memory, max_len=10):
        """自回归生成序列"""
        self.eval()
        batch_size = memory.size(0)
        device = memory.device
        
        # 初始输入：只有<bos>
        generated = torch.full((batch_size, 1), vocab['<bos>'], dtype=torch.long, device=device)
        
        for i in range(max_len - 1):
            # 创建当前序列的因果掩码
            current_len = generated.size(1)
            tgt_mask = create_tgt_mask(current_len)
            if tgt_mask is not None:
                tgt_mask = tgt_mask.to(device)
            
            # 前向传播
            output = self.forward(memory, generated, tgt_mask=tgt_mask)
            
            # 取最后一个时间步的输出
            next_token_logits = output[:, -1, :]  # (batch_size, vocab_size)
            
            # 贪婪解码：选择概率最高的token
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # (batch_size, 1)
            
            # 添加到生成序列
            generated = torch.cat([generated, next_token], dim=1)
            
            # 如果生成了<eos>，可以提前停止（可选）
            # if (next_token == vocab['<eos>']).all():
            #     break
        
        return generated

    


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_tokens = 0  # 正确预测的token数
    total_tokens = 0    # 总token数（排除<pad>）
    
    for batch_idx, (src, tgt) in enumerate(train_loader):
        src, tgt = src.to(device), tgt.to(device)
        decoder_input = tgt[:, :-1]
        target = tgt[:, 1:]
        
        optimizer.zero_grad()
        output = model(src, decoder_input)  # shape: (batch, seq_len-1, vocab_size)
        loss = criterion(output.transpose(1, 2), target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item()
        predictions = output.argmax(dim=-1)  # shape: (batch, seq_len-1)
        mask = target != vocab['<pad>']  # shape: (batch, seq_len-1)
        batch_correct = ((predictions == target) & mask).sum().item()
        batch_total = mask.sum().item()
        
        correct_tokens += batch_correct
        total_tokens += batch_total
        
        if (batch_idx + 1) % 500 == 0:
            avg_loss = running_loss / (batch_idx + 1)
            accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
            print(f'Batch {batch_idx+1}/{len(train_loader)}, Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}')
    
    return running_loss / len(train_loader), correct_tokens / total_tokens
def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_idx, (src, tgt) in enumerate(val_loader):
            src, tgt = src.to(device), tgt.to(device)
            decoder_input = tgt[:, :-1]
            decodr_output = tgt[:, 1:]
            output = model(src, decoder_input)
            loss = criterion(output.transpose(1, 2), decodr_output)
            total_loss += loss.item()
    return total_loss / len(val_loader)
def train(model, criterion, optimizer, num_epochs, device):
    # 准备保存目录
    save_dir = os.path.join(os.path.dirname(__file__), './', 'save_models')
    save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    last_val_loss = None
    for epoch in range(num_epochs):
        train_loader, val_loader, test_loader = get_full_data()
        train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        print('-'*50)
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}')
        print('-'*50)
        test(model, test_loader, device)
        last_val_loss = val_loss

    # 保存整个模型
    model.eval()  
    final_path = os.path.join(save_dir, f'final_model_epoch_{num_epochs}_val_{last_val_loss:.4f}.pth')
    torch.save(model, final_path)
    print(f'Final model saved to: {final_path}')

def test(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        for batch_idx, (src, tgt) in enumerate(test_loader):
            src, tgt = src.to(device), tgt.to(device)
            generated_seq = model.generate(src, max_len=15)
            seq = generated_seq.squeeze()
            
            # 使用 idx2token 进行反向查找
            source = ''.join([idx2token[i.item()] for i in src.squeeze() 
                            if i.item() not in [vocab['<pad>'], vocab['<bos>'], vocab['<eos>']]])
            target = ''.join([idx2token[i.item()] for i in tgt.squeeze() 
                            if i.item() not in [vocab['<pad>'], vocab['<bos>'], vocab['<eos>']]])
            prediction = ''.join([idx2token[i.item()] for i in seq 
                                if i.item() not in [vocab['<pad>'], vocab['<bos>'], vocab['<eos>']]])
            
            print(f'Source: {source}')
            print(f'Target: {target}')
            print(f'Prediction: {prediction}')
            print('-' * 30)
def User_pred(model, test_data):
    model.eval()
    with torch.no_grad():
        targets = []
        sources = []
        predictions = [] 
        for batch_idx, (src, tgt) in enumerate(test_data):
            generated_seq = model.generate(src, max_len=15)
            seq = generated_seq.squeeze()
            source = ''.join([idx2token[i.item()] for i in src.squeeze() 
                            if i.item() not in [vocab['<pad>'], vocab['<bos>'], vocab['<eos>']]])
            target = ''.join([idx2token[i.item()] for i in tgt.squeeze() 
                            if i.item() not in [vocab['<pad>'], vocab['<bos>'], vocab['<eos>']]])
            prediction = ''.join([idx2token[i.item()] for i in seq 
                                if i.item() not in [vocab['<pad>'], vocab['<bos>'], vocab['<eos>']]])
            sources.append(source)
            targets.append(target)
            predictions.append(prediction)
        if len(targets) == 1:
            return source, target, prediction
        else:
            return sources, targets, predictions
            
if __name__ == '__main__':
    ##########################################
    D = 64
    N_h = 4
    N_l = 2
    model = NumTransformer(D=D, N_h=N_h, N_l=N_l, vocab_size=len(vocab))
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 40
    ###########################################
    train(model,  criterion, optimizer, num_epochs, device)

        

        




    


