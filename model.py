import torch
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertModel
from torch.nn import CrossEntropyLoss
import sys
import os
from  dataset import KidsMentDataset
from tqdm import tqdm
import time
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import torch.nn as nn
from transformers import BertModel
from transformers import get_cosine_schedule_with_warmup



class BertClassification(nn.Module):
    def __init__(self, model_name, num_labels, dropout_prob=0.1):
        super(BertClassification, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}
def collate_fn(batch):
    input_ids, attention_masks, labels = zip(*batch)
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    labels = torch.tensor(labels)
    return input_ids, attention_masks, labels


def train_model(train_dataloader, val_dataloader, model, optimizer, scheduler, criterion, device, epochs):
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        start_time = time.time()

        total_steps = len(train_dataloader)
        print(f"Epoch {epoch+1} 시작")

        for step, (input_ids, attention_mask, labels) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            if step % 10 == 0:
                elapsed_time = time.time() - start_time
                steps_left = len(train_dataloader) - step - 1 + (epochs - epoch - 1) * len(train_dataloader)
                estimated_time_left = elapsed_time / (step + 1) * steps_left
                estimated_time_left_minutes = estimated_time_left / 60
                print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item()}, 남은 예상 시간: {int(estimated_time_left_minutes // 60)}시간 {int(estimated_time_left_minutes % 60)}분")

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} 완료. Average Loss: {avg_loss}")

        val_accuracy = evaluate_model(val_dataloader, model, device)
        print(f"Epoch {epoch+1} Validation Accuracy: {val_accuracy}")
def evaluate_model(eval_dataloader, model, device):
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for input_ids, attention_mask, labels in eval_dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits']
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_predictions += labels.size(0)

    accuracy = correct_predictions.double() / total_predictions
    return accuracy.item()



if __name__ == "__main__":
    # 데이터 준비
    train_data_dir = './Training_data'
    val_data_dir = './Validation_data'

    tokenizer = BertTokenizer.from_pretrained('monologg/kobert')
    train_dataset = KidsMentDataset(train_data_dir, tokenizer)
    val_dataset = KidsMentDataset(val_data_dir, tokenizer)

    # 검증 데이터셋을 검증용과 테스트용으로 분리
    val_size = int(0.5 * len(val_dataset))
    test_size = len(val_dataset) - val_size
    val_dataset, test_dataset = random_split(val_dataset, [val_size, test_size])

    # 데이터로더 생성
    train_dataloader = DataLoader(train_dataset, batch_size=16, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=16, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=16, collate_fn=collate_fn)

    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class_weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).to(device)
    criterion = CrossEntropyLoss(weight=class_weights)

    # KoBERT 모델 로드 및 설정
    model = BertClassification("monologg/kobert", num_labels=5, dropout_prob=0.3)

    # 학습 파라미터
    epochs = 10
    learning_rate = 1e-5
    warmup_steps = 0

    # 옵티마이저 설정
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

    # 스케줄러 설정
    total_steps = len(train_dataloader) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # 모델 학습
    train_model(train_dataloader, val_dataloader, model, optimizer, scheduler, criterion, device, epochs)

    # 모델 최종 평가 (테스트 데이터 사용)
    final_accuracy = evaluate_model(test_dataloader, model, device)
    print(f'Final Test Accuracy: {final_accuracy}')
