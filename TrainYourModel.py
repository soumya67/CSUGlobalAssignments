import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

from sklearn.model_selection import train_test_split

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your dataset from CSV file
twcs = pd.read_csv("/Users/soumyabhattacharyya/Downloads/archive/sample.csv")

# Initialize NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Define preprocessing functions
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in stemmed_tokens]
    cleaned_text = " ".join(lemmatized_tokens)
    return cleaned_text

# Example usage
sample_text = "Natural language processing involves analyzing unstructured text data."
cleaned_sample_text = preprocess_text(sample_text)
print("Cleaned text:", cleaned_sample_text)

# Define tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Tokenize and encode sequences in the dataset
def tokenize_and_encode(text):
    return tokenizer.encode(text, add_special_tokens=True, max_length=512, truncation=True, padding='max_length')

twcs['encoded'] = twcs['text'].apply(tokenize_and_encode)

# Split data into train, val, and test sets
X_train, X_test, y_train, y_test = train_test_split(
    twcs['encoded'], twcs['author_id'], test_size=0.2, random_state=42
)

# Custom Dataset class
class TextDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
        self.label_map = {label: idx for idx, label in enumerate(set(labels))}

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.inputs[idx])
        label = torch.tensor(self.label_map[self.labels[idx]])
        return input_ids, label

# Create datasets and dataloaders
train_dataset = TextDataset(X_train.tolist(), y_train.tolist())
test_dataset = TextDataset(X_test.tolist(), y_test.tolist())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define model parameters
hidden_size = 768  # BERT hidden size
num_layers = 2
bidirectional = True
rnn_cell = 'lstm'
use_attention = True
attn_method = 'dot'

# Define encoder and decoder using BERT
class BertEncoder(nn.Module):
    def __init__(self, bert_model):
        super(BertEncoder, self).__init__()
        self.bert = bert_model

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

class DecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers, bidirectional, rnn_cell, use_attention, attn_method):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers, bidirectional=bidirectional, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)
        self.use_attention = use_attention
        self.attn_method = attn_method

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden = self.rnn(embedded, hidden)
        output = self.fc(output)
        return output, hidden

encoder = BertEncoder(model).to(device)
decoder = DecoderRNN(output_size=len(tokenizer.vocab), hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional, rnn_cell=rnn_cell, use_attention=use_attention, attn_method=attn_method).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    encoder.train()
    decoder.train()
    total_loss = 0

    for batch in train_loader:
        input_ids, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = (input_ids != tokenizer.pad_token_id).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        encoder_outputs = encoder(input_ids, attention_mask)
        decoder_outputs, _ = decoder(labels, encoder_outputs)
        loss = criterion(decoder_outputs.view(-1, decoder_outputs.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

# Optional: Evaluation on the test set can be added here
