import pandas as pd
import torch
import numpy as np
import random

# 设备设置
device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
print(f"Using device: {device}")
"""
训练结果：

loss at batch 51 : 1.594391
loss at batch 101 : 1.573541
loss at batch 151 : 1.556525
loss at batch 201 : 1.573594
loss at batch 251 : 1.530967
epoch 1 ends.....
loss at batch 51 : 1.605683
loss at batch 101 : 1.497238
loss at batch 151 : 1.576917
loss at batch 201 : 1.470782
loss at batch 251 : 1.561073
epoch 2 ends.....
loss at batch 51 : 1.478766
loss at batch 101 : 1.454579
loss at batch 151 : 1.464379
loss at batch 201 : 1.482676
loss at batch 251 : 1.581146
epoch 3 ends.....
loss at batch 51 : 1.497352
loss at batch 101 : 1.512538
loss at batch 151 : 1.582034
loss at batch 201 : 1.446118
loss at batch 251 : 1.473812
epoch 4 ends.....
loss at batch 51 : 1.519116
loss at batch 101 : 1.515544
loss at batch 151 : 1.450097
loss at batch 201 : 1.377778
loss at batch 251 : 1.506994
epoch 5 ends.....
loss at batch 51 : 1.553029
loss at batch 101 : 1.370107
loss at batch 151 : 1.477551
loss at batch 201 : 1.478038
loss at batch 251 : 1.534450
epoch 6 ends.....
loss at batch 51 : 1.468205
loss at batch 101 : 1.464519
loss at batch 151 : 1.520252
loss at batch 201 : 1.524724
loss at batch 251 : 1.452872
epoch 7 ends.....
loss at batch 51 : 1.569808
loss at batch 101 : 1.605006
loss at batch 151 : 1.519989
loss at batch 201 : 1.528706
loss at batch 251 : 1.542280
epoch 8 ends.....
loss at batch 51 : 1.583972
loss at batch 101 : 1.408936
loss at batch 151 : 1.536768
loss at batch 201 : 1.501734
loss at batch 251 : 1.336245
epoch 9 ends.....
loss at batch 51 : 1.436669
loss at batch 101 : 1.719072
loss at batch 151 : 1.496130
loss at batch 201 : 1.578673
loss at batch 251 : 1.556490
epoch 10 ends.....
Train Accuracy: 0.2640
Test Accuracy: 0.2623

Prediction examples:
Text: This quiet , introspective and entertaining indepe...
Real label: 3, Predicted label: 1
Prediction probabilities: [0.19884382 0.20109367 0.1989347  0.20086111 0.20026666]
---
Text: Even fans of Ismail Merchant 's work , I suspect ,...
Real label: 1, Predicted label: 0
Prediction probabilities: [0.2003262  0.2000123  0.2000845  0.20008995 0.1994871 ]
---
Text: A positively thrilling combination of ethnography ...
Real label: 4, Predicted label: 4
Prediction probabilities: [0.19954525 0.19952916 0.19996919 0.20046382 0.20049256]
---
Text: Aggressive self-glorification and a manipulative w...
Real label: 1, Predicted label: 2
Prediction probabilities: [0.19931886 0.20017552 0.20088677 0.20020816 0.1994107 ]
---
Text: A comedy-drama of nearly epic proportions rooted i...
Real label: 3, Predicted label: 4
Prediction probabilities: [0.19981523 0.19991468 0.20016545 0.19992241 0.20018224]
---
Training completed.


"""


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


for idx in range(len(word_list)-1, -1, -1):
    if word_list_counter[idx] < 5:
        word_list.pop(idx)
        word_list_counter.pop(idx)

print(f"vocabulary size : {len(word_list)}")
print(word_list)
print(word_list_counter)




#词袋数据处理 标签one-hot
def create_train_data(train_data, train_labels):
    features = torch.zeros((len(train_data), len(word_list)), device=device)
    labels = torch.zeros(size=(len(train_data), 5), device=device)
    for idx, (paras, label) in enumerate(zip(train_data, train_labels)):
        words = paras.lower().split(' ')
        labels[idx][label] = 1
        for word in words:
            if is_english(word) and word in word_list:
                features[idx][word_list.index(word)] += 1

    features_sum = features.sum(axis=1, keepdim=True)
    features_sum = torch.where(features_sum == 0, torch.ones_like(features_sum), features_sum)
    features /= features_sum
    return features, labels



train_data, train_labels = create_train_data(train_text, num_train_labels)
test_data, test_labels = create_train_data(test_text, num_test_labels)
input_size = train_data.shape[1]
num_classes = 5


#获取小批量数据
def data_iter(batch_size, features, labels, counter=0):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    features, labels = features[indices], labels[indices]
    for i in range(0, num_examples, batch_size):
        counter += 1
        yield features[i: min(i + batch_size, num_examples)], labels[i: min(i + batch_size, num_examples)], counter

'''batch_size = 32
for a , b in data_iter(batch_size, train_data, train_labels):
    print(a.shape)
    print(b.shape)
    break'''


#输入的是bag-of-words
W = torch.normal(0, 0.01, size=(input_size, num_classes), requires_grad=True, device=device)
b = torch.zeros(num_classes, dtype=torch.float32, requires_grad=True, device=device)  
def softmax_net(W, b, data):   #net
    return softmax(torch.matmul(data, W) + b)
def sigmoid_net(W, b, data):
    return sigmoid(torch.matmul(data, W) + b)
def sigmoid(X):
    if X.dim() == 1:
        X = X.unsqueeze(0)
    return 1 / (1 + torch.exp(-X))
def softmax(X):
    if X.dim() == 1:
        X = X.unsqueeze(0)
    X_max = torch.max(X, dim=1, keepdim=True)[0]
    X_exp = torch.exp(X - X_max)  
    partition = X_exp.sum(1, keepdim=True)
    result = X_exp / partition
    epsilon = 1e-8
    result = torch.clamp(result, min=epsilon, max=1.0-epsilon)
    return result
def crossentropy(prediction, labels):       #loss
    loss = -torch.sum(labels * torch.log(prediction), dim=1)
    return loss
def squares_loss(prediction, labels):     #metrics
    return (prediction - labels) ** 2
def sgd(params, lr, batch_size):        #optimizer
    with torch.no_grad():
        for param in params:
            if param.grad is not None:
                param -= (param.grad * lr / batch_size)
                param.grad.zero_()
def onebatch_train_step(net, loss, optimizer, batch_size,lr,
                   train_data, train_labels, counter):
    raw_prediction = net(W, b, train_data)
    train_loss = loss(raw_prediction, train_labels)
    train_loss.sum().backward()
    optimizer([W, b], lr, batch_size)
    with torch.no_grad():
        if counter % 50 == 0:
            print(f"loss at batch {counter+1} : {float(train_loss.mean()):.6f}")
    return train_loss.mean()


lr = 0.01
num_epochs = 10
batch_size = 32

train_loss_list = []
best_w = None
best_b = None
for epoch in range(num_epochs):
    for data_batch, labels_batch, counter in data_iter(batch_size, train_data, train_labels):
        train_loss = onebatch_train_step(net=softmax_net, loss=crossentropy, optimizer=sgd,
                            batch_size=batch_size, lr=lr, train_data=data_batch,
                              train_labels=labels_batch, counter=counter)
        train_loss_list.append(train_loss.item())
        if best_w is None or train_loss.item() < min(train_loss_list):
            best_w = W.clone()
            best_b = b.clone()
    print(f'epoch {epoch+1} ends.....')

with torch.no_grad():
    # 使用最佳参数
    W_eval = best_w if best_w is not None else W
    b_eval = best_b if best_b is not None else b
    
    # 训练集准确率
    train_data_eval = train_data[:3000].to(device)
    train_labels_eval = train_labels[:3000].to(device)
    train_prediction = softmax_net(W_eval, b_eval, train_data_eval)
    train_pred_classes = torch.argmax(train_prediction, dim=1)  # 预测类别
    train_true_classes = torch.argmax(train_labels_eval, dim=1)  # 真实类别
    train_correct = (train_pred_classes == train_true_classes).sum().item()
    train_accuracy = train_correct / len(train_pred_classes)
    print(f"Train Accuracy: {train_accuracy:.4f}")
    
    # 测试集准确率
    test_data_eval = test_data.to(device)
    test_labels_eval = test_labels.to(device)
    test_prediction = softmax_net(W_eval, b_eval, test_data_eval)
    test_pred_classes = torch.argmax(test_prediction, dim=1)
    test_true_classes = torch.argmax(test_labels_eval, dim=1)
    test_correct = (test_pred_classes == test_true_classes).sum().item()
    test_accuracy = test_correct / len(test_pred_classes)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # 打印一些预测示例
    print("\nPrediction examples:")
    for idx in range(5):
        train_feature = train_data[idx].to(device)
        train_label_eval = train_labels[idx].to(device)
        true_label = torch.argmax(train_label_eval).item()
        prediction = softmax_net(W_eval, b_eval, train_feature.unsqueeze(0))
        pred_label = torch.argmax(prediction, dim=1).item()
        pred_probs = prediction.squeeze().cpu().detach().numpy()
        
        print(f"Text: {train_text[idx][:50]}...")
        print(f"Real label: {true_label}, Predicted label: {pred_label}")
        print(f"Prediction probabilities: {pred_probs}")
        print("---")

print("Training completed.")


    




        

    








