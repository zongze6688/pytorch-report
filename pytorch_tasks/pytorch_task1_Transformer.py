import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import random
import math
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
"""
训练结果：
Epoch [1/20], Train Loss: 1.5030, Train Acc: 31.36%
Epoch [1/20], Test Acc: 34.90%
Epoch [2/20], Train Loss: 1.4506, Train Acc: 36.21%
Epoch [2/20], Test Acc: 38.38%
Epoch [3/20], Train Loss: 1.4061, Train Acc: 39.29%
Epoch [3/20], Test Acc: 39.11%
Epoch [4/20], Train Loss: 1.3579, Train Acc: 41.50%
Epoch [4/20], Test Acc: 42.37%
Epoch [5/20], Train Loss: 1.3039, Train Acc: 44.48%
Epoch [5/20], Test Acc: 42.76%
Epoch [6/20], Train Loss: 1.2706, Train Acc: 45.32%
Epoch [6/20], Test Acc: 44.82%
Epoch [7/20], Train Loss: 1.2291, Train Acc: 48.02%
Epoch [7/20], Test Acc: 43.97%
Epoch [8/20], Train Loss: 1.1937, Train Acc: 49.14%
Epoch [8/20], Test Acc: 44.85%
Epoch [9/20], Train Loss: 1.1629, Train Acc: 50.56%
Epoch [9/20], Test Acc: 45.06%
Epoch [10/20], Train Loss: 1.1310, Train Acc: 52.02%
Epoch [10/20], Test Acc: 46.27%
Epoch [11/20], Train Loss: 1.1120, Train Acc: 52.92%
Epoch [11/20], Test Acc: 45.84%
Epoch [12/20], Train Loss: 1.0729, Train Acc: 54.13%
Epoch [12/20], Test Acc: 44.55%
Epoch [13/20], Train Loss: 1.0527, Train Acc: 55.77%
Epoch [13/20], Test Acc: 46.30%
Epoch [14/20], Train Loss: 1.0305, Train Acc: 56.47%
Epoch [14/20], Test Acc: 46.78%
Epoch [15/20], Train Loss: 1.0072, Train Acc: 57.15%
Epoch [15/20], Test Acc: 46.42%
Epoch [16/20], Train Loss: 0.9894, Train Acc: 58.55%
Epoch [16/20], Test Acc: 46.39%
Epoch [17/20], Train Loss: 0.9717, Train Acc: 59.70%
Epoch [17/20], Test Acc: 45.69%
Epoch [18/20], Train Loss: 0.9528, Train Acc: 60.25%
Epoch [18/20], Test Acc: 45.87%
Epoch [19/20], Train Loss: 0.9335, Train Acc: 60.87%
Epoch [19/20], Test Acc: 46.78%
Epoch [20/20], Train Loss: 0.9186, Train Acc: 62.58%
Epoch [20/20], Test Acc: 46.81%

"""


# 设备设置
device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
print(f"Using device: {device}")

# 读取数据
ftest = pd.read_csv('new_test.tsv', sep='\t', header=None, names=['text', 'label'])
ftrain = pd.read_csv('new_train.tsv', sep='\t', header=None, names=['text', 'label'])
raw_test_text, raw_train_text = dict(ftest), dict(ftrain)
test_text, num_test_labels = raw_test_text['text'].tolist(), raw_test_text['label'].tolist()
train_text, num_train_labels = raw_train_text['text'].tolist(), raw_train_text['label'].tolist()
# 创建数据加载器
def create_data_loader(features, labels, batch_size=32, shuffle=True):
    dataset = TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def is_english(words):
    for word in words:
        if word.isalpha() == False:
            return False
    return True

# 构建词汇表
word_list = []
word_list_counter = []
for paras in train_text:
    words = paras.lower().split(' ')
    for word in words:
        if is_english(word) and word not in word_list:
            word_list.append(word)
            word_list_counter.append(1)
        elif is_english(word) and word in word_list:
            word_list_counter[word_list.index(word)] += 1

# 过滤低频词
for idx in range(len(word_list)-1, -1, -1):
    if word_list_counter[idx] < 5:
        word_list.pop(idx)
        word_list_counter.pop(idx)
print(f'vocab size : {len(word_list)}')


def get_vocab(text, word_list):
    vocab = {'<pad>':0, '<unk>':1}
    for word in word_list:
        vocab[word] = len(vocab)
    return vocab
vocab = get_vocab(train_text, word_list)

def create_data(text, labels, vocab, valid_len=30):
    features = torch.zeros(size=(len(text), valid_len), dtype=torch.long, device=device)
    labels_tensor = torch.zeros(size=(len(text), 5), device=device)
    for idx, (paras, label) in enumerate(zip(text, labels)):
        words = paras.lower().split(' ')
        labels_tensor[idx][label] = 1
        for widx in range(min(len(words), valid_len)):
            word = words[widx]
            if is_english(word) and word in vocab:
                features[idx][widx] = vocab[word]
            else:
                features[idx][widx] = vocab['<unk>']
    return features, labels_tensor

train_data, train_labels = create_data(train_text, num_train_labels, vocab)
test_data, test_labels = create_data(test_text, num_test_labels, vocab)
train_loader = create_data_loader(train_data, train_labels, batch_size=32, shuffle=True)
test_loader = create_data_loader(test_data, test_labels, batch_size=32, shuffle=False)

class PositionalEncoding(nn.Module):
    """位置编码 - 为Transformer提供位置信息"""
    def __init__(self, d_model, max_len=5000):
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

class TextTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, hidden_size, num_layers, num_classes, max_len=30, dropout=0.1):
        super(TextTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, max_len)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, 
                                                        dim_feedforward=hidden_size, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_size, num_classes)
    
    def forward(self, X):
        X = self.embedding(X) * math.sqrt(self.embedding.embedding_dim)
        X = X.permute(1, 0, 2)
        X = self.pos_encoder(X)
        X = X.permute(1, 0, 2)
        transformer_out = self.transformer_encoder(X)
        out = transformer_out.mean(dim=1)
        out = self.fc(out)
        return out


model = TextTransformer(vocab_size=len(vocab), embed_size=128, num_heads=8,
                        hidden_size=128, num_layers=3,
                        num_classes=5, max_len=30, dropout=0.1)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total = 0
    correct = 0
    running_loss = 0.0
    for idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, torch.argmax(labels, dim=1))
        loss.backward()
        optimizer.step()
        _, preds = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (preds == torch.argmax(labels, dim=1)).sum().item()
        running_loss += loss.item()
        
    accuracy = 100. * correct / total
    epoch_loss = running_loss / len(train_loader)
    return epoch_loss, accuracy
def evaluate(model, test_loader, criterion, device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for idx, (data, labels) in enumerate(test_loader):
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, preds = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (preds == torch.max(labels, 1)[1]).sum().item()
        accuracy = 100. * correct / total
        return accuracy

def train(model, train_loader, test_loader, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        test_acc = evaluate(model, test_loader, criterion, device)
        print(f'Epoch [{epoch+1}/{num_epochs}], Test Acc: {test_acc:.2f}%')
train(model, train_loader, test_loader, criterion, optimizer, num_epochs, device)