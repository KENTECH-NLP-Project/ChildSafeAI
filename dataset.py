import os
import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

class KidsMentDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length=300):
        self.data = []
        self.labels = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.load_data(data_dir)

    def load_data(self, data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(data_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as file:
                    json_data = json.load(file)
                    info = json_data.get("info", {})
                    text = json_data.get("text", "")
                    if "위기단계" in info and text:
                        label = self.label_mapping(info["위기단계"])
                        self.data.append(text)
                        self.labels.append(label)

    def label_mapping(self, 위기단계):
        label_map = {
            "정상군": 0,
            "관찰필요": 1,
            "상담필요": 2,
            "학대의심": 3,
            "응급": 4
        }
        return label_map.get(위기단계, -1)  # 위기단계가 없으면 -1 반환

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        print(input_ids, attention_mask, label)
        return 

