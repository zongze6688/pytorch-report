import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 设备设置
device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
print(f"Using device: {device}")


class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.texts[idx].lower().split()
        ids = [self.vocab.get(tok, self.vocab['<UNK>']) for tok in tokens]
        # truncate or pad
        if len(ids) >= self.max_len:
            ids = ids[:self.max_len]
        else:
            ids = ids + [self.vocab['<PAD>']] * (self.max_len - len(ids))

        return torch.LongTensor(ids), torch.tensor(int(self.labels[idx]), dtype=torch.long)


def build_vocab(texts, min_freq=5, max_size=None):
    counter = {}
    for s in texts:
        for tok in s.lower().split():
            counter[tok] = counter.get(tok, 0) + 1

    # sort tokens by freq
    items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    for tok, freq in items:
        if freq < min_freq:
            continue
        if max_size and idx >= max_size:
            break
        vocab[tok] = idx
        idx += 1
    return vocab


class ResidualConvBlock2D(nn.Module):
    """Residual block using Conv2d for text classification."""
    def __init__(self, in_channels, out_channels, kernel_size_k, embed_dim):
        super().__init__()
        # kernel = (k, embed_dim) to cover the entire embedding dimension
        padding_k = (kernel_size_k - 1) // 2
        
        # first conv: from in_channels to out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                              kernel_size=(kernel_size_k, embed_dim), 
                              padding=(padding_k, 0))
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # second conv: keep channels the same
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                              kernel_size=(kernel_size_k, 1), 
                              padding=(padding_k, 0))
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 
                                     kernel_size=(1, 1), 
                                     padding=(0, 0))
        else:
            self.shortcut = None

    def forward(self, x):
        # x: (batch, in_channels, seq_len, embed_dim)
        out = F.relu(self.bn1(self.conv1(x)))  # (batch, out_channels, seq_len, 1)
        out = self.bn2(self.conv2(out))        # (batch, out_channels, seq_len, 1)
        
        # shortcut connection
        if self.shortcut is not None:
            residual = self.shortcut(x)  # (batch, out_channels, seq_len, embed_dim)
            # adjust dimensions if needed
            if residual.size(2) != out.size(2):
                min_len = min(residual.size(2), out.size(2))
                residual = residual[:, :, :min_len, :]
                out = out[:, :, :min_len, :]
            # adjust the last dimension: from embed_dim to 1
            if residual.size(3) != out.size(3):
                residual = residual.mean(dim=3, keepdim=True)
        else:
            residual = x
            if residual.size(3) != out.size(3):
                residual = residual.mean(dim=3, keepdim=True)
        
        out = F.relu(out + residual)
        return out


class ResidualTextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_classes=5, 
                 filter_sizes=(3,4,5), num_filters=100, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.blocks = nn.ModuleList()
        for k in filter_sizes:
            block = ResidualConvBlock2D(
                in_channels=1, 
                out_channels=num_filters, 
                kernel_size_k=k, 
                embed_dim=embed_dim
            )
            self.blocks.append(block)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x)  # (batch, seq_len, embed_dim)
        emb2d = emb.unsqueeze(1)  # (batch, 1, seq_len, embed_dim)
        
        outs = []
        for block in self.blocks:
            out = block(emb2d)  # (batch, num_filters, seq_len, 1)
            # global average pooling over spatial dimensions
            out = F.adaptive_avg_pool2d(out, (1, 1))  # (batch, num_filters, 1, 1)
            out = out.view(out.size(0), -1)  # (batch, num_filters)
            outs.append(out)

        cat = torch.cat(outs, dim=1)  # (batch, num_filters * len(filter_sizes))
        cat = self.dropout(cat)
        logits = self.fc(cat)  # (batch, num_classes)
        return logits


def load_tsv(path):
    df = pd.read_csv(path, sep='\t', header=None, names=['text', 'label'])
    texts = df['text'].fillna('').astype(str).tolist()
    labels = df['label'].tolist()
    return texts, labels


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total * 100.0


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

    return total_loss / total, correct / total * 100.0


def main(args):
    base_dir = os.path.dirname(__file__)
    train_path = os.path.join(base_dir, 'new_train.tsv')
    test_path = os.path.join(base_dir, 'new_test.tsv')

    # Check if files exist
    if not os.path.exists(train_path):
        print(f"Training file not found: {train_path}")
        return
    if not os.path.exists(test_path):
        print(f"Test file not found: {test_path}")
        return

    train_texts, train_labels = load_tsv(train_path)
    test_texts, test_labels = load_tsv(test_path)

    print(f"Training samples: {len(train_texts)}, Test samples: {len(test_texts)}")

    # build vocab from training texts
    vocab = build_vocab(train_texts, min_freq=args.min_freq, max_size=args.vocab_size)
    print(f'Vocab size: {len(vocab)} (including PAD/UNK)')

    # create datasets and loaders
    train_dataset = TextDataset(train_texts, train_labels, vocab, args.max_len)
    test_dataset = TextDataset(test_texts, test_labels, vocab, args.max_len)

    # simple train/val split
    val_split = int(len(train_dataset) * args.val_ratio)
    if val_split > 0:
        val_dataset = torch.utils.data.Subset(train_dataset, range(val_split))
        train_dataset = torch.utils.data.Subset(train_dataset, range(val_split, len(train_dataset)))
    else:
        val_dataset = None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False) if val_dataset is not None else None
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # create model
    model = ResidualTextCNN(
        vocab_size=len(vocab), 
        embed_dim=args.embed_dim, 
        num_classes=args.num_classes,
        filter_sizes=args.filter_sizes, 
        num_filters=args.num_filters, 
        dropout=args.dropout
    )
    model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    best_state = None
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        print(f'Epoch {epoch}/{args.epochs}  Train loss: {train_loss:.4f}  Train acc: {train_acc:.2f}%')

        if val_loader is not None:
            val_loss, val_acc = eval_epoch(model, val_loader, criterion)
            print(f'  Val loss: {val_loss:.4f}  Val acc: {val_acc:.2f}%')
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = model.state_dict().copy()
                print(f'  New best validation accuracy: {best_val_acc:.2f}%')

    # 测试评估（若有保存的最好模型则加载）
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f'Loaded best model with val_acc={best_val_acc:.2f}%')

    test_loss, test_acc = eval_epoch(model, test_loader, criterion)
    print(f'Test loss: {test_loss:.4f}  Test acc: {test_acc:.2f}%')

    # 打印若干预测示例
    model.eval()
    with torch.no_grad():
        print("\nPrediction examples:")
        for i in range(min(5, len(test_dataset))):
            x, y = test_dataset[i]
            logits = model(x.unsqueeze(0).to(device))
            pred = torch.argmax(logits, dim=1).item()
            true_label = int(test_labels[i])
            print(f'Text: {test_texts[i][:120]}...')
            print(f'True: {true_label}, Pred: {pred} {"✓" if true_label == pred else "✗"}')
            print('---')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Residual TextCNN 文本分类')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--embed-dim', type=int, default=128, help='词嵌入维度')
    parser.add_argument('--num-classes', type=int, default=5)
    parser.add_argument('--filter-sizes', type=int, nargs='+', default=[3,4,5])
    parser.add_argument('--num-filters', type=int, default=100)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--max-len', type=int, default=100)
    parser.add_argument('--min-freq', type=int, default=5)
    parser.add_argument('--vocab-size', type=int, default=None)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    args = parser.parse_args()

    main(args)