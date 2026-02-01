import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import random
import os
import matplotlib.pyplot as plt  
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm 
def check_mps_compatibility():
    if torch.backends.mps.is_available():
        print("MPS is available. Enforcing float32 compatibility.")
        # 设置默认的浮点类型为float32
        torch.set_default_dtype(torch.float32)
check_mps_compatibility()
"""
使用BERT-large训练5个epoch的结果(batch_size = 16):
Starting training...

Epoch 1/5
--------------------------------------------------
Train Loss: 1.0781, Train Acc: 0.5504
Val Loss: 0.8213, Val Acc: 0.6506, Val F1: 0.6341
-> New best model saved!

Epoch 2/5
--------------------------------------------------
Train Loss: 0.6565, Train Acc: 0.7401
Val Loss: 0.8163, Val Acc: 0.6635, Val F1: 0.6620
-> New best model saved!

Epoch 3/5
--------------------------------------------------
Train Loss: 0.3411, Train Acc: 0.8782
Val Loss: 0.9886, Val Acc: 0.6788, Val F1: 0.6780
-> New best model saved!

Epoch 4/5
--------------------------------------------------
Train Loss: 0.1711, Train Acc: 0.9459
Val Loss: 1.3560, Val Acc: 0.6729, Val F1: 0.6739

Epoch 5/5
--------------------------------------------------
Train Loss: 0.0779, Train Acc: 0.9770
Val Loss: 1.6156, Val Acc: 0.6788, Val F1: 0.6794

==================================================
Final Evaluation on Test Set (Using Best Model)
==================================================
Test Loss: 1.0365
Test Accuracy: 0.6691
Test Weighted F1 Score: 0.6676
Text: What a fantastic movie! => Predicted Class: 4
Text: I did not like the movie. => Predicted Class: 1
Text: It was an average film. => Predicted Class: 2

"""
"""

使用BERT-base训练3个epoch的结果:
Starting training...

Epoch 1/3
--------------------------------------------------
Train Loss: 1.1285, Train Acc: 0.5365
Val Loss: 0.9281, Val Acc: 0.6032, Val F1: 0.5920
-> New best model saved!

Epoch 2/3
--------------------------------------------------
Train Loss: 0.7827, Train Acc: 0.6907
Val Loss: 0.8810, Val Acc: 0.6190, Val F1: 0.6127
-> New best model saved!

Epoch 3/3
--------------------------------------------------
Train Loss: 0.6052, Train Acc: 0.7751
Val Loss: 0.8831, Val Acc: 0.6383, Val F1: 0.6371
-> New best model saved!

==================================================
Final Evaluation on Test Set (Using Best Model)
==================================================
Test Loss: 0.9135
Test Accuracy: 0.6392
Test Weighted F1 Score: 0.6360
Text: What a fantastic movie! => Predicted Class: 4
Text: I did not like the movie. => Predicted Class: 1
Text: It was an average film. => Predicted Class: 1

"""

# 1. Set Random Seed for Reproducibility
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

set_seed(42)

# Set device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# 2. Load Data Robustly
try:
    # Try newer pandas syntax
    ftrain = pd.read_csv('./NLP/new_train.tsv', sep='\t', header=None, names=['text', 'label'], on_bad_lines='skip')
    ftest = pd.read_csv('./NLP/new_test.tsv', sep='\t', header=None, names=['text', 'label'], on_bad_lines='skip')
except TypeError:
    # Fallback for older pandas versions
    ftrain = pd.read_csv('./NLP/new_train.tsv', sep='\t', header=None, names=['text', 'label'], error_bad_lines=False)
    ftest = pd.read_csv('./NLP/new_test.tsv', sep='\t', header=None, names=['text', 'label'], error_bad_lines=False)

# Drop any rows that might have become NaN during loading
ftrain.dropna(inplace=True)
ftest.dropna(inplace=True)

# 3. Create Validation Split
train_df, val_df = train_test_split(ftrain, test_size=0.2, random_state=42)

print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples: {len(ftest)}")

# Dataset Class
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Encoding
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Initialize Tokenizer
model_name = 'bert-large-uncased' # Keeping your choice of large
tokenizer = BertTokenizer.from_pretrained(model_name)

# Create Datasets
train_dataset = TextClassificationDataset(train_df['text'].tolist(), train_df['label'].tolist(), tokenizer)
val_dataset = TextClassificationDataset(val_df['text'].tolist(), val_df['label'].tolist(), tokenizer)
test_dataset = TextClassificationDataset(ftest['text'].tolist(), ftest['label'].tolist(), tokenizer)

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize Model
num_classes = 5
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
model = model.to(device)

# Optimizer and Scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

epochs = 4 
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# --- Training Function ---
def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    progress_bar = tqdm(data_loader, desc="Training", leave=False, position=0)
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        total_predictions += labels.size(0)
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions.float() / total_predictions
    
    return avg_loss, accuracy.item() # Ensure we return a float

# --- Evaluation Function ---
def eval_model(model, data_loader, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(data_loader, desc="Evaluating", leave=False, position=0)
    
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_predictions += labels.size(0)
            total_loss += loss.item()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions.float() / total_predictions
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy.item(), f1

# Lists to store metrics for plotting
train_losses, train_accs = [], []
val_losses, val_accs = [], []

# Training Loop
print("\nStarting training...")
best_val_accuracy = 0
save_path = './NLP/bert_finetuned_best_model'

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    print("-" * 50)
    
    # Train
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    
    # Validate
    val_loss, val_acc, val_f1 = eval_model(model, val_loader, device)
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
    
    # Store metrics
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    # Save Best Model
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print("-> New best model saved!")

# --- Plotting the Results ---
def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    epochs_range = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accs, label='Train Acc')
    plt.plot(epochs_range, val_accs, label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("\nTraining history plot saved as 'training_history.png'")
    plt.show()

plot_training_history(train_losses, val_losses, train_accs, val_accs)

print("\n" + "=" * 50)
print("Final Evaluation on Test Set (Using Best Model)")
print("=" * 50)

# Load the Best Saved Model for Final Testing
best_model = BertForSequenceClassification.from_pretrained(save_path)
best_model.to(device)

test_loss, test_acc, test_f1 = eval_model(best_model, test_loader, device)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Weighted F1 Score: {test_f1:.4f}")


# Example Predictions
def predict_sentiment(text, model, tokenizer, device, max_length=128):
    # Prepare the text
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Move to GPU/CPU
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        _, prediction = torch.max(logits, dim=1)
    
    return prediction.item()
samples = [
    "This movie is utterly unwatchable and a complete waste of time.",
    "She delivers a performance that is both skillful and deeply moving.",
    "The plot is so predictable it becomes tedious halfway through.",
    "It’s a hilarious and brilliantly executed comedy from start to finish.",
    "The film feels rushed and poorly edited, with no emotional payoff.",
    "A visually stunning but emotionally empty exercise in style.",
    "Every performance in this film is perfectly pitched and utterly convincing.",
    "An absolute masterpiece that will be remembered for years to come.",
    "The script is filled with clichés and the acting is painfully wooden.",
    "A charming and gently humorous tale that warms the heart.",
    "The pacing is sluggish and the story goes nowhere interesting.",
    "It’s a well-crafted thriller that keeps you guessing until the end.",
    "Despite a strong start, the film collapses into melodrama.",
    "One of the most boring and pointless films I’ve ever seen.",
    "The cinematography is breathtaking and the score is haunting.",
    "An uneven mix of comedy and drama that never quite gels.",
    "The dialogue is sharp, the characters are vivid, and the direction is assured.",
    "A mess of incoherent scenes with no narrative thread.",
    "It has its moments, but overall it’s forgettable and slight.",
    "A powerful and unforgettable cinematic experience.",
]
for sample in samples:
    prediction = predict_sentiment(sample, best_model, tokenizer, device)
    print(f"Text: {sample} => Predicted Class: {prediction}")



"""
results:

Test Loss: 1.2981
Test Accuracy: 0.6851
Test Weighted F1 Score: 0.6852

Text: This movie is utterly unwatchable and a complete waste of time. => Predicted Class: 0
Text: She delivers a performance that is both skillful and deeply moving. => Predicted Class: 4
Text: The plot is so predictable it becomes tedious halfway through. => Predicted Class: 1
Text: It’s a hilarious and brilliantly executed comedy from start to finish. => Predicted Class: 4
Text: The film feels rushed and poorly edited, with no emotional payoff. => Predicted Class: 1
Text: A visually stunning but emotionally empty exercise in style. => Predicted Class: 2
Text: Every performance in this film is perfectly pitched and utterly convincing. => Predicted Class: 4
Text: An absolute masterpiece that will be remembered for years to come. => Predicted Class: 4
Text: The script is filled with clichés and the acting is painfully wooden. => Predicted Class: 1
Text: A charming and gently humorous tale that warms the heart. => Predicted Class: 4
Text: The pacing is sluggish and the story goes nowhere interesting. => Predicted Class: 1
Text: It’s a well-crafted thriller that keeps you guessing until the end. => Predicted Class: 3
Text: Despite a strong start, the film collapses into melodrama. => Predicted Class: 2
Text: One of the most boring and pointless films I’ve ever seen. => Predicted Class: 0
Text: The cinematography is breathtaking and the score is haunting. => Predicted Class: 4
Text: An uneven mix of comedy and drama that never quite gels. => Predicted Class: 2
Text: The dialogue is sharp, the characters are vivid, and the direction is assured. => Predicted Class: 3
Text: A mess of incoherent scenes with no narrative thread. => Predicted Class: 0
Text: It has its moments, but overall it’s forgettable and slight. => Predicted Class: 2
Text: A powerful and unforgettable cinematic experience. => Predicted Class: 4
"""