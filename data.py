import os
import json
from torch.utils.data import Dataset

class KidsMentDataset(Dataset):
    def __init__(self, data_dir):
        self.data = []
        self.labels = []
        self.load_data(data_dir)

    def load_data(self, data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(data_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as file:
                    json_data = json.load(file)
                    info = json_data.get("info", {})
                    kids_ment = json_data.get("kids_ment", [])
                    if "위기단계" in info and kids_ment:
                        label = self.label_mapping(info["위기단계"])
                        text = " ".join([f"{q['Q']} {q['A']}" for q in kids_ment])
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
        return self.data[idx], self.labels[idx]
