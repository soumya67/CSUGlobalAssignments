import torch
import pandas as pd
from transformers import BertTokenizer, BertForQuestionAnswering, AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify
import csv


class QuestionAnsweringDataset(Dataset):
    def __init__(self, questions, contexts, answers, tokenizer, max_len=512):
        self.questions = questions
        self.contexts = contexts
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = str(self.questions[idx])
        context = str(self.contexts[idx])
        answer = str(self.answers[idx])

        encoding = self.tokenizer.encode_plus(
            question,
            context,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        answer_tokens = self.tokenizer.encode(answer, add_special_tokens=False)
        for i in range(len(input_ids) - len(answer_tokens) + 1):
            if input_ids[i:i + len(answer_tokens)].tolist() == answer_tokens:
                start_position = i
                end_position = i + len(answer_tokens) - 1
                break
        else:
            start_position = 0
            end_position = 0

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'start_positions': torch.tensor([start_position]),
            'end_positions': torch.tensor([end_position])
        }


df = pd.read_csv('squad_formatted.csv')
sample_size = min(500, len(df))
reduced_df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
train_df, val_df = train_test_split(reduced_df, test_size=0.1, random_state=42)
train_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True, inplace=True)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

train_dataset = QuestionAnsweringDataset(train_df['question'], train_df['context'], train_df['answer'], tokenizer)
val_dataset = QuestionAnsweringDataset(val_df['question'], val_df['context'], val_df['answer'], tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


def train(model, train_loader, val_loader, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions,
                            end_positions=end_positions)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {total_loss / len(train_loader)}")
        evaluate(model, val_loader)


def evaluate(model, val_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions,
                            end_positions=end_positions)
            loss = outputs.loss
            total_loss += loss.item()
        print(f"Validation Loss: {total_loss / len(val_loader)}")


app = Flask(__name__)


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data['question']
    context = data['context']
    answer = answer_question(question, context)
    return jsonify({"answer": answer})


def answer_question(question, context):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer.encode_plus(question, context, return_tensors="pt", max_length=512, truncation=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)

        # Debugging outputs
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        start_probs = torch.nn.functional.softmax(start_logits, dim=-1)
        end_probs = torch.nn.functional.softmax(end_logits, dim=-1)
        start_index = torch.argmax(start_probs)
        end_index = torch.argmax(end_probs) + 1

        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        answer_tokens = tokens[start_index:end_index]
        answer = tokenizer.convert_tokens_to_string(answer_tokens)

        print("Input Tokens:", tokens)
        print("Start Logit Index:", start_index, "Token:", tokens[start_index])
        print("End Logit Index:", end_index, "Token:", tokens[end_index - 1])
        print("Answer:", answer)

        return answer


@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    with open('feedback_data.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([data['question'], data['answer'], data['feedback']])
    return jsonify({"message": "Feedback received. Thank you!"})


if __name__ == '__main__':
    train(model, train_loader, val_loader, optimizer, epochs=2)
    app.run(debug=True)
