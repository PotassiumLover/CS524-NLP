# Binary Authorship Attribution G.K. Chesterton using BERT

# --------------------------------------------
# Import Libraries
# --------------------------------------------

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import time
import datetime

# --------------------------------------------
# Load the Dataset
# --------------------------------------------

# Load the dataset
df = pd.read_csv('text_to_authorship.csv')

# Display first few rows
print("First 5 entries:")
print(df.head())

# Check for null values
print("\nNull values in each column:")
print(df.isnull().sum())

# Remove missing values
df = df.dropna()

# Encode labels
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'],
    df['label'],
    test_size=0.2,
    random_state=42
)

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# --------------------------------------------
# Section 4: Prepare Dataset and DataLoaders
# --------------------------------------------

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
            
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
            
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def eval_model(model, data_loader, device):
    model.eval()
    predictions, true_labels = [], []
        
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
                
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
        
    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, target_names=['Other Authors', 'G.K. Chesterton'])
    cm = confusion_matrix(true_labels, predictions)
    return accuracy, report, cm

def train_epoch(model, data_loader, optimizer, device, scheduler):
    model.train()
    total_loss = 0
        
    for batch in data_loader:
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
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
    avg_loss = total_loss / len(data_loader)
    return avg_loss

import optuna

def objective(trial):
    # Hyperparameters to optimize
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 5e-5)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    epochs = trial.suggest_int('epochs', 2, 4)
    max_len = trial.suggest_categorical('max_len', [128, 256, 512])
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_len=max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    total_steps = len(train_loader) * epochs
    warmup_steps = trial.suggest_int('warmup_steps', 0, int(total_steps * 0.1))
    weight_decay = trial.suggest_uniform('weight_decay', 0.01, 0.1)
    # Update the dataset and DataLoader with new batch_size and max_len
    #train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_len=max_len)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, max_len=max_len)
    
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model and optimizer
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2
    )
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=len(train_loader) * epochs
    )
    
    # Train and evaluate the model
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, scheduler)
    
    accuracy, _, _ = eval_model(model, val_loader, device)
    
    return accuracy

# Run Optuna optimization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Best hyperparameters
print("Best hyperparameters:", study.best_params)

"""
# Create datasets
train_dataset = TextDataset(train_texts, train_labels, tokenizer)
val_dataset = TextDataset(val_texts, val_labels, tokenizer)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# --------------------------------------------
# Model Setup
# --------------------------------------------

# Load pre-trained BERT model for binary classification
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2,  # Binary classification
    output_attentions=False,
    output_hidden_states=False
)

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# --------------------------------------------
# Training the Model
# --------------------------------------------

# Set up optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

epochs = 3
total_steps = len(train_loader) * epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# --- 6.3 Training Loop ---
for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    train_loss = train_epoch(model, train_loader, optimizer, device, scheduler)
    print(f'Training loss: {train_loss:.4f}')

# --------------------------------------------
# Evaluating the Model
# --------------------------------------------

# Evaluate the model
accuracy, report, cm = eval_model(model, val_loader, device)
print(f'Validation Accuracy: {accuracy:.4f}')
print('Classification Report:')
print(report)
print('Confusion Matrix:')
print(cm)

# --------------------------------------------
# Saving the Model
# --------------------------------------------

# Save the fine-tuned model
model.save_pretrained('bert-authorship-attribution-binary')
tokenizer.save_pretrained('bert-authorship-attribution-binary')
"""