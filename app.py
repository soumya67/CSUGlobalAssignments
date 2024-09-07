import torch
from transformers import BertTokenizerFast, BertForQuestionAnswering, BertForSequenceClassification, pipeline
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import time
import spacy
from flask import Flask, request, jsonify, render_template
from transformers import AdamW

# Load pre-trained models and tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
qa_model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
classification_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
sentiment_analyzer = pipeline('sentiment-analysis')
nlp = spacy.load("en_core_web_sm")

# Load and preprocess the SQuAD dataset
squad_data = pd.read_csv('E:/SQUAD/squad_formatted.csv').head(1000)

class SquadDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = str(self.data.iloc[idx]['question'])
        context = str(self.data.iloc[idx]['context'])
        answer = str(self.data.iloc[idx]['answer'])
        
        start_idx = context.find(answer)
        end_idx = start_idx + len(answer)
        
        if start_idx == -1:
            return None
        
        inputs = self.tokenizer.encode_plus(
            question, context, add_special_tokens=True, return_tensors="pt", 
            return_offsets_mapping=True, truncation=True, max_length=512
        )
        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()
        offsets = inputs["offset_mapping"].squeeze()
        
        token_start_idx = None
        token_end_idx = None
        
        for i, (start, end) in enumerate(offsets):
            if start <= start_idx < end:
                token_start_idx = i
            if start < end_idx <= end:
                token_end_idx = i
                break
        
        if token_start_idx is None or token_end_idx is None:
            return None
        
        start_positions = torch.tensor(token_start_idx)
        end_positions = torch.tensor(token_end_idx)
        
        return input_ids, attention_mask, start_positions, end_positions

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None, None, None, None
    
    input_ids = pad_sequence([item[0] for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence([item[1] for item in batch], batch_first=True, padding_value=0)
    start_positions = torch.stack([item[2] for item in batch])
    end_positions = torch.stack([item[3] for item in batch])
    return input_ids, attention_mask, start_positions, end_positions

# Create DataLoader
dataset = SquadDataset(squad_data, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

# Fine-tune the QA model
optimizer = AdamW(qa_model.parameters(), lr=5e-5)

qa_model.train()
start_time = time.time()
for epoch in range(3):
    print(f"Starting epoch {epoch + 1}")
    for batch_idx, batch in enumerate(dataloader):
        input_ids, attention_mask, start_positions, end_positions = batch
        if input_ids is None:
            continue
        outputs = qa_model(input_ids=input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx} processed")
    print(f"Epoch {epoch + 1} completed")
end_time = time.time()
print(f"Total training time: {end_time - start_time:.2f} seconds")

# Save the fine-tuned QA model
qa_model.save_pretrained('fine-tuned-bert-squad')

# Function to answer questions using the fine-tuned QA model
def answer_question(question, context):
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].tolist()[0]
    outputs = qa_model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    return answer.title()

# Function to classify text using the classification model
def classify_text(text):
    inputs = tokenizer.encode_plus(text, add_special_tokens=True, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    outputs = classification_model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class

# Function to analyze sentiment
def analyze_sentiment(text):
    result = sentiment_analyzer(text)
    return result[0]['label'], result[0]['score']

# Function for POS tagging
def pos_tagging(text):
    doc = nlp(text)
    return [(token.text, token.pos_) for token in doc]

# Flask app for chatbot interface
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    context = data.get('context')
    question = data.get('question')
    answer = answer_question(question, context)
    sentiment, score = analyze_sentiment(answer)
    pos_tags = pos_tagging(answer)
    classification = classify_text(answer)
    response = {
        'answer': answer,
        'sentiment': {'label': sentiment, 'score': score},
        'pos_tags': pos_tags,
        'classification': classification
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
