import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

"""
训练结果：
vocabulary size : 3301
loss at batch 0 : 1.817878246307373
loss at batch 30 : 1.7494590282440186
loss at batch 60 : 1.5609805583953857
loss at batch 90 : 1.4296588897705078
loss at batch 120 : 1.5198894739151
loss at batch 150 : 1.6616333723068237
loss at batch 180 : 1.4898509979248047
loss at batch 210 : 1.3350763320922852
Epoch 1/8:
  训练损失: 1.5650, 训练准确率: 31.84%
  验证损失: 1.3757, 验证准确率: 44.49%
--------------------------------------------------
loss at batch 0 : 1.2377300262451172
loss at batch 30 : 1.2262369394302368
loss at batch 60 : 1.3057199716567993
loss at batch 90 : 1.3871718645095825
loss at batch 120 : 1.189626693725586
loss at batch 150 : 1.1863001585006714
loss at batch 180 : 1.2519142627716064
loss at batch 210 : 1.2633835077285767
Epoch 2/8:
  训练损失: 1.3308, 训练准确率: 45.71%
  验证损失: 1.2735, 验证准确率: 46.01%
--------------------------------------------------
loss at batch 0 : 1.1360244750976562
loss at batch 30 : 1.0299259424209595
loss at batch 60 : 1.1499513387680054
loss at batch 90 : 1.172273874282837
loss at batch 120 : 1.188576340675354
loss at batch 150 : 1.007507085800171
loss at batch 180 : 0.9198834300041199
loss at batch 210 : 1.3086979389190674
Epoch 3/8:
  训练损失: 1.1423, 训练准确率: 54.49%
  验证损失: 1.2431, 验证准确率: 46.83%
--------------------------------------------------
loss at batch 0 : 0.9672365784645081
loss at batch 30 : 0.9554058909416199
loss at batch 60 : 0.8480309247970581
loss at batch 90 : 1.1134099960327148
loss at batch 120 : 1.0281788110733032
loss at batch 150 : 0.9031245708465576
loss at batch 180 : 0.9781384468078613
loss at batch 210 : 0.8327819108963013
Epoch 4/8:
  训练损失: 1.0117, 训练准确率: 60.61%
  验证损失: 1.2612, 验证准确率: 46.54%
--------------------------------------------------
loss at batch 0 : 0.7415857911109924
loss at batch 30 : 0.9043591618537903
loss at batch 60 : 1.0622800588607788
loss at batch 90 : 1.0367116928100586
loss at batch 120 : 0.8947807550430298
loss at batch 150 : 0.6744070649147034
loss at batch 180 : 0.832672119140625
loss at batch 210 : 0.8537123799324036
Epoch 5/8:
  训练损失: 0.9093, 训练准确率: 65.54%
  验证损失: 1.3252, 验证准确率: 45.60%
--------------------------------------------------
loss at batch 0 : 0.5274830460548401
loss at batch 30 : 0.6660799384117126
loss at batch 60 : 0.6578310132026672
loss at batch 90 : 0.6400795578956604
loss at batch 120 : 0.7439295053482056
loss at batch 150 : 0.7828940749168396
loss at batch 180 : 0.8537034392356873
loss at batch 210 : 0.8492198586463928
Epoch 6/8:
  训练损失: 0.8160, 训练准确率: 68.72%
  验证损失: 1.3585, 验证准确率: 45.02%
--------------------------------------------------
loss at batch 0 : 0.7921894788742065
loss at batch 30 : 0.8633372783660889
loss at batch 60 : 0.525318443775177
loss at batch 90 : 0.4674370288848877
loss at batch 120 : 1.0963196754455566
loss at batch 150 : 0.8389272093772888
loss at batch 180 : 0.8018089532852173
loss at batch 210 : 0.6508316993713379
Epoch 7/8:
  训练损失: 0.7547, 训练准确率: 72.40%
  验证损失: 1.4477, 验证准确率: 45.49%
--------------------------------------------------
loss at batch 0 : 0.5746660232543945
loss at batch 30 : 0.5811178088188171
loss at batch 60 : 0.4704658091068268
loss at batch 90 : 0.6254042387008667
loss at batch 120 : 0.9420700073242188
loss at batch 150 : 0.8552841544151306
loss at batch 180 : 0.6583836674690247
loss at batch 210 : 0.7143605947494507
Epoch 8/8:
  训练损失: 0.6932, 训练准确率: 74.68%
  验证损失: 1.4978, 验证准确率: 45.60%
--------------------------------------------------
测试集结果: 损失=1.4969, 准确率=45.66%
训练集子集结果: 损失=0.2933, 准确率=93.70%

预测示例:
文本: Kidman is really the only thing that 's worth watc...
真实标签: 1, 预测标签: 3
预测概率: [[0.0056169  0.10870252 0.25148937 0.6191518  0.01503946]]
------------------------------
文本: Once you get into its rhythm ... the movie becomes...
真实标签: 3, 预测标签: 2
预测概率: [[0.11338624 0.17749816 0.2883006  0.2631794  0.15763566]]
------------------------------
文本: I kept wishing I was watching a documentary about ...
真实标签: 1, 预测标签: 1
预测概率: [[0.3121501  0.31845093 0.13314159 0.09906703 0.13719036]]
------------------------------
文本: Kinnear does n't aim for our sympathy , but rather...
真实标签: 3, 预测标签: 3
预测概率: [[3.3206010e-04 1.3079267e-02 1.4965660e-02 9.4706744e-01 2.4555573e-02]]
------------------------------
文本: All ends well , sort of , but the frenzied comic m...
真实标签: 2, 预测标签: 2
预测概率: [[6.0806423e-04 1.5522365e-01 7.8173226e-01 6.2084630e-02 3.5137494e-04]]

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

print(f"vocabulary size : {len(word_list)}")

# 词袋数据处理
def create_data(text_data, labels):
    features = torch.zeros((len(text_data), len(word_list)), device=device)
    labels_tensor = torch.zeros(size=(len(text_data), 5), device=device)
    for idx, (paras, label) in enumerate(zip(text_data, labels)):
        words = paras.lower().split(' ')
        labels_tensor[idx][label] = 1
        for word in words:
            if is_english(word) and word in word_list:
                features[idx][word_list.index(word)] += 1

    # 归一化
    features_sum = features.sum(axis=1, keepdim=True)
    features_sum = torch.where(features_sum == 0, torch.ones_like(features_sum), features_sum)
    features /= features_sum
    return features, labels_tensor

# 准备数据
train_data, train_labels = create_data(train_text, num_train_labels)
test_data, test_labels = create_data(test_text, num_test_labels)

# 划分训练集和验证集
def split_train_val(features, labels, val_ratio=0.2):
    num_samples = len(features)
    indices = list(range(num_samples))
    random.shuffle(indices)
    
    split_idx = int(num_samples * (1 - val_ratio))
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_features = features[train_indices]
    train_labels_split = labels[train_indices]
    val_features = features[val_indices]
    val_labels_split = labels[val_indices]
    
    return train_features, train_labels_split, val_features, val_labels_split

train_data, train_labels, val_data, val_labels = split_train_val(train_data, train_labels)

# 创建数据加载器
def create_data_loader(features, labels, batch_size=32, shuffle=True):
    dataset = TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# 定义MLP模型
class MLP(nn.Module):
    def __init__(self, *layers, weight_decay=1e-4):
        super(MLP, self).__init__()
        
        self.network = nn.Sequential(*layers)
        self.weight_decay = weight_decay
        
    def forward(self, x):
        return self.network(x)
    
    def regularization_loss(self):
        reg_loss = 0
        for param in self.parameters():
            reg_loss += torch.norm(param, p=2) 
        return self.weight_decay * reg_loss


input_size = len(word_list)
hidden_sizes = [512, 256, 128, 64]  
num_classes = 5
learning_rate = 0.001
batch_size = 32
num_epochs = 8
dropout_rate = [0.2, 0.5, 0.5, 0.5]
weight_decay = 3e-4

class Residual(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):      
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dr1 = nn.Dropout(0.2)
        self.lin2 = nn.Linear(input_size, output_size)
        self.bn2 = nn.BatchNorm1d(output_size)
        self.dr2 = nn.Dropout(0.3)
    def forward(self, x):
        Y = F.relu((self.bn1(self.lin1(x))))
        Y = self.dr1(Y)
        Y = self.bn2(self.lin2(x))
        if self.input_size != self.output_size:
            self.shortcut = nn.Linear(self.input_size, self.output_size)
            x = self.shortcut(x)
        Y += x
        return self.dr2(F.relu(Y))
layers = [
    Residual(input_size, hidden_sizes[0], hidden_sizes[1]),
    nn.Dropout(0.3),
    Residual(hidden_sizes[1], hidden_sizes[2], hidden_sizes[3]),
    nn.Dropout(0.5),
    nn.Linear(hidden_sizes[3], num_classes)
    
]
# 创建模型、损失函数和优化器
model = MLP(*layers, weight_decay=weight_decay).to(device)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

train_loader = create_data_loader(train_data, train_labels, batch_size)
val_loader = create_data_loader(val_data, val_labels, batch_size, shuffle=False)
test_loader = create_data_loader(test_data, test_labels, batch_size, shuffle=False)

# 训练和验证函数
def train_epoch(model, dataloader, loss, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        
        # 计算损失（ + L2正则化）
        ce_loss = loss(output, torch.argmax(target, dim=1))
        if batch_idx % 30 == 0:
            print(f'loss at batch {batch_idx} : {ce_loss}')
        reg_loss = model.regularization_loss()
        overall_loss = ce_loss + reg_loss
        
        overall_loss.backward()
        optimizer.step()
        
        running_loss += overall_loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == torch.argmax(target, dim=1)).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, loss):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            overall_loss = loss(output, torch.argmax(target, dim=1))
            running_loss += overall_loss.item()
            
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == torch.argmax(target, dim=1)).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

# 训练循环
train_losses = []
val_losses = []
train_accs = []
val_accs = []

best_val_acc = 0
best_model_state = None

for epoch in range(num_epochs):
    # 训练
    train_loss, train_acc = train_epoch(model, train_loader, loss, optimizer)
    
    # 验证
    val_loss, val_acc = validate_epoch(model, val_loader, loss)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    
    
    print(f'Epoch {epoch+1}/{num_epochs}:')
    print(f'  训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%')
    print(f'  验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%')
    print('-' * 50)


# 在测试集上评估
test_loss, test_acc = validate_epoch(model, test_loader, loss)
print(f'测试集结果: 损失={test_loss:.4f}, 准确率={test_acc:.2f}%')

# 在训练集上评估（前3000个样本）
train_subset = TensorDataset(train_data[:3000], train_labels[:3000])
train_subset_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False)
train_subset_loss, train_subset_acc = validate_epoch(model, train_subset_loader, loss)
print(f'训练集子集结果: 损失={train_subset_loss:.4f}, 准确率={train_subset_acc:.2f}%')

# 预测示例
print("\n预测示例:")
model.eval()
with torch.no_grad():
    for i in range(5):
        sample = test_data[i].unsqueeze(0).to(device)
        test_label_eval = test_labels[i].to(device)
        prediction = model(sample)
        pred_class = torch.argmax(prediction, dim=1).item()
        true_class = torch.argmax(test_label_eval, dim=0).item()
        
        print(f"文本: {test_text[i][:50]}...")
        print(f"真实标签: {true_class}, 预测标签: {pred_class}")
        print(f"预测概率: {torch.softmax(prediction, dim=1).cpu().numpy()}")
        print("-" * 30)



    




        

    








