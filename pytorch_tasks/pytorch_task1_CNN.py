import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

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





#TODO : 构建一个vocab <pad>0 <unk>1
#TODO : 将文本转化为索引序列 [batch_size, max_len]
#TODO : 构建一个带有嵌入层的CNN模型
#TODO : 训练并测试模型

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

#带有嵌入层的CNN模型
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, embed_size), padding=1)
        #self.bn1 = nn.BatchNorm2d(32)
        self.mxp1 = nn.MaxPool2d(kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 1), padding=1)
        #self.bn2 = nn.BatchNorm2d(64)
        self.mxp2 = nn.MaxPool2d(kernel_size=(2,1))
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(64, num_classes)
    def forward(self, X):
        X = self.embedding(X).unsqueeze(1)  # [batch_size, 1, max_len, embed_size]
        X = F.relu(self.conv1(X))
        X = self.mxp1(X)
        X = F.relu(self.conv2(X))
        X = self.mxp2(X)
        X = self.avgpool(X) # [batch_size, 64, 1, 1]
        X = X.squeeze(-1).squeeze(-1) # [batch_size, 64]
        X = self.fc1(X)
        return X
    
'''尝试优化上面的结构'''
class TextCNN_OptimizedV1(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes):
        super(TextCNN_OptimizedV1, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, embed_size), padding=(1, 0))
        self.bn1 = nn.BatchNorm2d(32)
        self.mxp1 = nn.MaxPool2d(kernel_size=(2, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 1), padding=(1, 0))
        self.bn2 = nn.BatchNorm2d(64)
        self.mxp2 = nn.MaxPool2d(kernel_size=(2, 1))
        self.dropout = nn.Dropout(0.5)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64, num_classes)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, X):
        # X: [batch_size, seq_len]
        X = self.embedding(X)  # [batch_size, seq_len, embed_size]
        X = X.unsqueeze(1)     # [batch_size, 1, seq_len, embed_size]
        X = F.relu(self.bn1(self.conv1(X)))  # [batch, 32, seq_len, 1]
        X = self.mxp1(X)                     # [batch, 32, seq_len//2, 1]
        X = F.relu(self.bn2(self.conv2(X)))  # [batch, 64, seq_len//2, 1]
        X = self.mxp2(X)                     # [batch, 64, seq_len//4, 1]
        X = self.avgpool(X)                  # [batch, 64, 1, 1]
        X = X.squeeze(-1).squeeze(-1)       # [batch, 64]
        X = self.dropout(X)
        X = self.fc1(X)
        
        return X
class TextCNN_OptimizedV2(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes):
        super(TextCNN_OptimizedV2, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        
        # 在嵌入层后加Dropout
        self.embed_dropout = nn.Dropout(0.2)

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, embed_size), padding=(2, 0))
        self.bn1 = nn.BatchNorm2d(32)
        self.mxp1 = nn.MaxPool2d(kernel_size=(2, 1))
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 1), padding=(2, 0))
        self.bn2 = nn.BatchNorm2d(64)
        self.mxp2 = nn.MaxPool2d(kernel_size=(2, 1))
        # 在卷积层后加Dropout
        self.conv_dropout = nn.Dropout(0.3)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64, 32)  # 中间层
        self.fc2 = nn.Linear(32, num_classes)  # 输出层
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, X):
        X = self.embedding(X)
        X = self.embed_dropout(X)
        X = X.unsqueeze(1)
        X = F.relu(self.bn1(self.conv1(X)))
        X = self.mxp1(X)
        X = F.relu(self.bn2(self.conv2(X)))
        X = self.mxp2(X)
        X = self.conv_dropout(X)
        X = self.avgpool(X)
        X = X.squeeze(-1).squeeze(-1)
        X = F.relu(self.fc1(X))
        X = self.dropout(X)
        X = self.fc2(X)
        
        return X

#加深CNN
class DeepTextCNN_V1(nn.Module):
    def __init__(self, vocab_size, embed_size=100, num_classes=5):
        super(DeepTextCNN_V1, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, embed_size), padding=(1,0))
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 1), padding=(1,0))
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 1), padding=(1,0))
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=(3, 1), padding=(1,0))
        self.bn4 = nn.BatchNorm2d(512)
        
        self.pool1 = nn.MaxPool2d(kernel_size=(2,1))
        self.pool2 = nn.MaxPool2d(kernel_size=(2,1))
        self.pool3 = nn.MaxPool2d(kernel_size=(2,1))
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        self.fc1 = nn.Linear(512, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, X):
        X = self.embedding(X).unsqueeze(1)  # [batch, 1, seq_len, embed_size]
        X = F.relu(self.bn1(self.conv1(X)))  # [batch, 64, seq_len, 1]
        X = self.pool1(X)                    # [batch, 64, seq_len//2, 1]
        X = F.relu(self.bn2(self.conv2(X)))  # [batch, 128, seq_len//2, 1]
        X = self.pool2(X)                    # [batch, 128, seq_len//4, 1]
        X = F.relu(self.bn3(self.conv3(X)))  # [batch, 256, seq_len//4, 1]
        X = self.pool3(X)                    # [batch, 256, seq_len//8, 1]
        X = F.relu(self.bn4(self.conv4(X)))  # [batch, 512, seq_len//8, 1]
        X = self.avgpool(X)                  # [batch, 512, 1, 1]
        X = X.squeeze(-1).squeeze(-1)       # [batch, 512]
        X = F.relu(self.fc1(X))
        X = self.dropout1(X)
        X = F.relu(self.fc2(X))
        X = self.dropout2(X)
        X = self.fc3(X)
        
        return X
    
class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                              kernel_size=(kernel_size, 1), 
                              padding=(kernel_size//2, 0))
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                              kernel_size=(kernel_size, 1), 
                              padding=(kernel_size//2, 0))
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(identity)  # 残差连接
        out = F.relu(out)
        
        return out

class DeepResTextCNN(nn.Module):
    """带有残差连接的深度文本CNN"""
    def __init__(self, vocab_size, embed_size=100, num_classes=5):
        super(DeepResTextCNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        self.init_conv = nn.Conv2d(1, 64, kernel_size=(3, embed_size), padding=(1,0))
        self.init_bn = nn.BatchNorm2d(64)

        self.res_block1 = ResidualBlock(64, 128, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,1))
        
        self.res_block2 = ResidualBlock(128, 256, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,1))
        
        self.res_block3 = ResidualBlock(256, 512, kernel_size=3)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = F.relu(self.init_bn(self.init_conv(x)))
        x = self.res_block1(x)
        x = self.pool1(x)
        x = self.res_block2(x)
        x = self.pool2(x)
        x = self.res_block3(x)
        x = self.global_pool(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.fc(x)
        
        return x
class MultiScaleDeepTextCNN(nn.Module):
    """多尺度深度文本CNN"""
    def __init__(self, vocab_size, embed_size=100, num_classes=5):
        super(MultiScaleDeepTextCNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.conv1_1 = nn.Conv2d(1, 64, kernel_size=(3, embed_size))
        self.conv1_2 = nn.Conv2d(64, 128, kernel_size=(3, 1), padding=(1,0))
        self.conv1_3 = nn.Conv2d(128, 256, kernel_size=(3, 1), padding=(1,0))
        self.conv2_1 = nn.Conv2d(1, 64, kernel_size=(4, embed_size))
        self.conv2_2 = nn.Conv2d(64, 128, kernel_size=(4, 1), padding=(2,0))
        self.conv2_3 = nn.Conv2d(128, 256, kernel_size=(4, 1), padding=(2,0))
        self.conv3_1 = nn.Conv2d(1, 64, kernel_size=(5, embed_size))
        self.conv3_2 = nn.Conv2d(64, 128, kernel_size=(5, 1), padding=(2,0))
        self.conv3_3 = nn.Conv2d(128, 256, kernel_size=(5, 1), padding=(2,0))
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool1 = nn.AdaptiveMaxPool2d((1,1))
        self.pool2 = nn.AdaptiveMaxPool2d((1,1))
        self.pool3 = nn.AdaptiveMaxPool2d((1,1))
        self.fusion_conv = nn.Conv1d(3, 1, kernel_size=1)  # 融合三个分支
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x).unsqueeze(1)  # [batch, 1, seq_len, embed_size]
        x1 = F.relu(self.conv1_1(embedded))  # [batch, 64, seq_len-3+1, 1]
        x1 = F.relu(self.conv1_2(x1))        # [batch, 128, seq_len-3+1, 1]
        x1 = F.relu(self.conv1_3(x1))        # [batch, 256, seq_len-3+1, 1]
        x1 = self.bn1(x1)
        x1 = self.pool1(x1).squeeze(-1).squeeze(-1)  # [batch, 256]
        x2 = F.relu(self.conv2_1(embedded))  # [batch, 64, seq_len-4+1, 1]
        x2 = F.relu(self.conv2_2(x2))        # [batch, 128, seq_len-4+1, 1]
        x2 = F.relu(self.conv2_3(x2))        # [batch, 256, seq_len-4+1, 1]
        x2 = self.bn2(x2)
        x2 = self.pool2(x2).squeeze(-1).squeeze(-1)  # [batch, 256]
        x3 = F.relu(self.conv3_1(embedded))  # [batch, 64, seq_len-5+1, 1]
        x3 = F.relu(self.conv3_2(x3))        # [batch, 128, seq_len-5+1, 1]
        x3 = F.relu(self.conv3_3(x3))        # [batch, 256, seq_len-5+1, 1]
        x3 = self.bn3(x3)
        x3 = self.pool3(x3).squeeze(-1).squeeze(-1)  # [batch, 256]
        features = torch.stack([x1, x2, x3], dim=1)  # [batch, 3, 256]
        fused = self.fusion_conv(features).squeeze(1)  # [batch, 256]
        out = F.relu(self.fc1(fused))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

model = TextCNN_OptimizedV2(vocab_size=len(vocab), embed_size=100, num_classes=5)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
model.apply(init_weights)
# 训练模型
num_epochs = 40

def train_epoch(model, data_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (data, labels) in enumerate(data_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(data) 
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == torch.argmax(labels, dim=1)).sum().item()
        if batch_idx % 10 == 0:
            print(f'  Batch {batch_idx}/{len(data_loader)}, Loss: {loss.item():.4f}')
    avg_loss = running_loss / len(data_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for idx, (data, labels) in enumerate(test_loader):
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, preds = torch.max(outputs, dim=1)
            total += data.shape[0]
            correct += (preds == torch.argmax(labels, dim=1)).sum().item()
        accuracy = 100.0 * correct / total
    print('-' * 50)
    print(f'test accuracy : {accuracy}')
    return accuracy
def train(model, train_loader, criterion, optimizer):
    for epoch in range(num_epochs):
        loss, acc = train_epoch(model, train_loader, criterion, optimizer)
        print(f'loss at epoch {epoch+1} : {loss:.3f}, acc : {acc:.2f}')
        evaluate(model, test_loader)
def show_result(model, tests, labels):
    model.eval()
    with torch.no_grad():
        for idx, (data, label) in enumerate(zip(tests, labels)):
            data, label = data.to(device), label.to(device)
            output = model(data.unsqueeze(0))  # [1, num_classes]
            _, pred = torch.max(output, dim=1)  # 获取每个样本的预测类别
            print(f'{test_text[idx]}')
            print(f'output : {output.squeeze().tolist()} , pred : {pred.item()}')
            print(f'real label : {torch.argmax(label).item()}')
            print('-' * 50)

train(model, train_loader, criterion, optimizer)
evaluate(model, test_loader)
show_result(model, test_data[:5], test_labels[:5])
            

        


