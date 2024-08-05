import torch
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.nn import CrossEntropyLoss
import sys
import os
from  dataset import KidsMentDataset
from tqdm import tqdm
import time


def collate_fn(batch):
    texts, labels = zip(*batch)
    tokenizer = BertTokenizer.from_pretrained('monologg/kobert')
    encodings = tokenizer(list(texts), return_tensors='pt', padding=True, truncation=True, max_length=128)
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    labels = torch.tensor(labels)
    return input_ids, attention_mask, labels


def train_model(train_dataloader, val_dataloader, model, optimizer, criterion, epochs=3):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        # 예상 시간을 처음에 한 번 출력
        total_steps = len(train_dataloader)
        estimated_total_time = (total_steps * epochs) * 0.5  # 초기 대략적인 예상 시간 (0.5초를 임의로 설정)
        print(f"Epoch {epoch+1} 시작. 예상 학습 시간: {int(estimated_total_time // 3600)}시간 {int((estimated_total_time % 3600) // 60)}분")

        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
            input_ids, attention_mask, labels = batch
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # 첫 스텝 후 남은 시간 계산 및 출력
            if step == 0:
                elapsed_time = time.time() - start_time
                estimated_time_per_epoch = elapsed_time * total_steps
                estimated_total_time = estimated_time_per_epoch * epochs
                print(f"예상 학습 시간 업데이트: {int(estimated_total_time // 3600)}시간 {int((estimated_total_time % 3600) // 60)}분")
            
            # 매 10 스텝마다 남은 시간 출력
            if step % 10 == 0:
                elapsed_time = time.time() - start_time
                steps_left = len(train_dataloader) - step - 1
                estimated_time_left = elapsed_time / (step + 1) * steps_left
                print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item()}, 남은 예상 시간: {int(estimated_time_left // 3600)}시간 {int((estimated_time_left % 3600) // 60)}분")

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} 완료. Average Loss: {avg_loss}")

        # Validation 데이터로 평가
        val_accuracy = evaluate_model(val_dataloader, model)
        print(f"Epoch {epoch+1} Validation Accuracy: {val_accuracy}")



def evaluate_model(eval_dataloader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
    accuracy = correct / total
    return accuracy


# 데이터셋 디렉토리 설정
train_data_dir = '/content/drive/MyDrive/Colab Notebooks/train_data'

# 데이터셋 로드
train_dataset = KidsMentDataset(train_data_dir)

# 데이터셋 크기 구하기
dataset_size = len(train_dataset)
print(f"Train dataset size: {dataset_size}")


batch_size = 16

# 스텝 수 계산
steps_per_epoch = dataset_size // batch_size + (1 if dataset_size % batch_size != 0 else 0)
print(f"Steps per epoch: {steps_per_epoch}")


if __name__ == "__main__":
    # 데이터 준비
    train_data_dir = '/content/drive/MyDrive/Colab Notebooks/train_data'
    val_data_dir = '/content/drive/MyDrive/Colab Notebooks/val_data'

    train_dataset = KidsMentDataset(train_data_dir)
    val_dataset = KidsMentDataset(val_data_dir)

    # 검증용과 테스트용으로 분리
    val_size = int(0.5 * len(val_dataset))
    test_size = len(val_dataset) - val_size
    val_dataset, test_dataset = random_split(val_dataset, [val_size, test_size])

    # 데이터로더 생성
    train_dataloader = DataLoader(train_dataset, batch_size=16, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=16, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=16, collate_fn=collate_fn)

    # KoBERT 모델 로드 및 설정
    model = BertForSequenceClassification.from_pretrained("monologg/kobert", num_labels=5)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    criterion = CrossEntropyLoss()

    # 모델 학습
    train_model(train_dataloader, val_dataloader, model, optimizer, criterion, epochs=5)

    # 테스트 데이터
    final_accuracy = evaluate_model(test_dataloader, model)
    print(f'Final Test Accuracy: {final_accuracy}')