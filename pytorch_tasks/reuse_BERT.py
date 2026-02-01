import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 1. Setup Device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# 2. Load the Saved Model and Tokenizer
# Ensure the folder name matches where you saved it in the training script
model_path = './bert_finetuned_best_model'

print(f"Loading model from {model_path}...")
try:
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
except OSError:
    print(f"Error: Could not find model in directory '{model_path}'.")
    print("Please run the training script first to generate the model files.")
    exit()

model.to(device)
model.eval()
print("Model loaded successfully!")

# 3. Prediction Function
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

# 4. Run on Samples
while True:
    text = input("Enter your sentence (q to quit): ")
    if text == "q":
        break
    pred_class = predict_sentiment(text, model, tokenizer, device)
    # Truncate text for cleaner printing
    display_text = (text[:47] + '...') if len(text) > 47 else text
    print(f"{display_text:<50} | {pred_class}")