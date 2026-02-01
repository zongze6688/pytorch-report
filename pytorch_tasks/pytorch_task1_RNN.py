import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
"""
运行结果：（使用LSTM双向RNN模型）
Using device: mps
vocab size : 3301
Epoch [1/10], Train Loss: 1.4891, Train Acc: 32.47%
Epoch [1/10], Test Acc: 32.97%
Epoch [2/10], Train Loss: 1.3933, Train Acc: 39.14%
Epoch [2/10], Test Acc: 39.62%
Epoch [3/10], Train Loss: 1.2790, Train Acc: 45.42%
Epoch [3/10], Test Acc: 41.40%
Epoch [4/10], Train Loss: 1.1535, Train Acc: 51.21%
Epoch [4/10], Test Acc: 41.70%
Epoch [5/10], Train Loss: 1.0110, Train Acc: 57.79%
Epoch [5/10], Test Acc: 44.09%
Epoch [6/10], Train Loss: 0.8654, Train Acc: 64.82%
Epoch [6/10], Test Acc: 44.42%
Epoch [7/10], Train Loss: 0.6953, Train Acc: 72.40%
Epoch [7/10], Test Acc: 44.18%
Epoch [8/10], Train Loss: 0.5277, Train Acc: 79.56%
Epoch [8/10], Test Acc: 44.73%
Epoch [9/10], Train Loss: 0.3760, Train Acc: 86.35%
Epoch [9/10], Test Acc: 44.21%
Epoch [10/10], Train Loss: 0.2566, Train Acc: 91.04%
Epoch [10/10], Test Acc: 43.64%


"""

"""
运行结果（使用带注意力机制的LSTM双向RNN模型）：
Using device: mps
vocab size : 3301
Epoch [1/20], Train Loss: 1.5045, Train Acc: 30.82%
Epoch [1/20], Test Acc: 34.21%
Epoch [2/20], Train Loss: 1.4694, Train Acc: 33.88%
Epoch [2/20], Test Acc: 37.08%
Epoch [3/20], Train Loss: 1.4328, Train Acc: 36.56%
Epoch [3/20], Test Acc: 39.14%
Epoch [4/20], Train Loss: 1.3923, Train Acc: 38.23%
Epoch [4/20], Test Acc: 40.19%
Epoch [5/20], Train Loss: 1.3553, Train Acc: 41.64%
Epoch [5/20], Test Acc: 42.49%
Epoch [6/20], Train Loss: 1.3197, Train Acc: 42.86%
Epoch [6/20], Test Acc: 44.24%
Epoch [7/20], Train Loss: 1.2893, Train Acc: 43.55%
Epoch [7/20], Test Acc: 46.03%
Epoch [8/20], Train Loss: 1.2444, Train Acc: 47.05%
Epoch [8/20], Test Acc: 45.72%
Epoch [9/20], Train Loss: 1.2148, Train Acc: 47.56%
Epoch [9/20], Test Acc: 46.72%
Epoch [10/20], Train Loss: 1.1869, Train Acc: 49.04%
Epoch [10/20], Test Acc: 46.12%
Epoch [11/20], Train Loss: 1.1610, Train Acc: 50.49%
Epoch [11/20], Test Acc: 47.66%
Epoch [12/20], Train Loss: 1.1252, Train Acc: 52.45%
Epoch [12/20], Test Acc: 47.08%
Epoch [13/20], Train Loss: 1.1053, Train Acc: 53.04%
Epoch [13/20], Test Acc: 45.75%
Epoch [14/20], Train Loss: 1.0683, Train Acc: 55.00%
Epoch [14/20], Test Acc: 47.17%
Epoch [15/20], Train Loss: 1.0538, Train Acc: 55.87%
Epoch [15/20], Test Acc: 48.38%
Epoch [16/20], Train Loss: 1.0203, Train Acc: 57.04%
Epoch [16/20], Test Acc: 45.48%
Epoch [17/20], Train Loss: 1.0097, Train Acc: 57.28%
Epoch [17/20], Test Acc: 48.08%
Epoch [18/20], Train Loss: 0.9882, Train Acc: 58.43%
Epoch [18/20], Test Acc: 49.41%
Epoch [19/20], Train Loss: 0.9674, Train Acc: 59.44%
Epoch [19/20], Test Acc: 48.41%
Epoch [20/20], Train Loss: 0.9483, Train Acc: 60.38%
Epoch [20/20], Test Acc: 47.57%

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

# TODO : 构建一个RNN GRU LSTM模型进行文本分类
# TODO ：构建一个带有attention机制的RNN模型进行文本分类
# TODO ：使用Transformer模型进行文本分类

class TextRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes, rnn_type = 'rnn', num_layers=1, bidirectional=False):
        super(TextRNN, self).__init__()
        self.bidirectional = bidirectional
        if rnn_type == 'rnn':
            self.rnn = nn.RNN(embed_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(embed_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), num_classes)

    def init_weight(self):
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def forward(self, X):
        embedded_X = self.embedding(X)
        if isinstance(self.rnn, nn.LSTM):
            _, (hidden, _) = self.rnn(embedded_X)
        else:
            _, hidden = self.rnn(embedded_X)
        if self.bidirectional:
            forward_H = hidden[-2]
            backward_H = hidden[-1]
            hidden_state = torch.cat((forward_H, backward_H), dim=1)
        else:
            hidden_state = hidden[-1]

        out = self.fc(hidden_state)
        return out
class TextRNNWithAttention(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes, rnn_type = 'lstm', num_layers=1,
                 num_heads=4, dropout=0.5 ,bidirectional=True):
        super().__init__()
        self.biderectional = bidirectional
        if rnn_type.lower() == 'rnn':
            self.rnn = nn.RNN(embed_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        elif rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(embed_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        elif rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * (2 if bidirectional else 1),
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), num_classes)
    def forward(self, X):
        embedded_X = self.embedding(X)
        embedded_X = self.dropout(embedded_X)
        outputs, _ = self.rnn(embedded_X)
        att_output, _ = self.attention(outputs, outputs, outputs)
        a_output = att_output.mean(dim=1)
        out = self.fc(a_output)
        return out
        

    
        
model = TextRNNWithAttention(vocab_size=len(vocab), 
                embed_size=50, 
                hidden_size=128, 
                num_classes=5, 
                rnn_type='lstm', 
                num_layers=2, 
                bidirectional=True)
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