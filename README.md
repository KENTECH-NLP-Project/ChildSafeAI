
# ChildSafeAI
 _Summer Internship Project at KENTECH Energy AI under the supervision of Prof. Chongkwon Kim_ <p>
ChildSafeAI is an AI-based project designed to analyze children's conversation data and detect signs of abuse. By leveraging natural language processing and machine learning, this project aims to provide a tool that can help in safeguarding children's welfare.  <p>
ChildSafeAI는 아동의 대화 데이터를 분석하고 학대의 징후를 감지하는 프로젝트입니다. NLP와 ML을 활용하여 아동의 복지를 보호하는 데 도움을 줄 수 있는 모델을 구축하고자 합니다.






## Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [License](#license)
- [Contact](#contact)

## Introduction

ChildSafeAI uses advanced machine learning techniques to analyze conversations between children and counselors. The primary goal is to identify potential abuse cases based on the dialogue. The model is trained on labeled datasets where each conversation is categorized by its risk level.

## Features

- **NLP(Natural Language Processing)**: Uses KoBERT (Korean BERT) for understanding and processing the Korean language.
- **Abuse Detection**: Identifies various levels of risk and potential abuse from children's conversations.
- **Scalable**: Can be trained on new datasets to improve accuracy and expand functionality.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/childsafeai.git
cd childsafeai
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. **Prepare your dataset**: Ensure your dataset is in the correct format as described in the [Dataset](#dataset) section.

2. **Train the model**: Use the provided scripts to train the model on your dataset.

```bash
python train.py
```

3. **Evaluate the model**: Run evaluations to check the model's performance.

```bash
python evaluate.py
```

## Dataset

The dataset should be in JSON format with the following structure:

```json
{
    "info": {
        "위기단계": "정상군"
    },
    "kids_ment": [
        {"Q": "최근에 아픈 곳이 있었어?", "A": "아픈 곳 없어요."},
        {"Q": "최근에 다친 적은 있었어?", "A": "아니요. 안 다쳤어요."},
        {"Q": "친구는 뭐 할 때가 즐거워?", "A": "엄마랑 아빠 기다리는 시간이 즐거워요."},
        {"Q": "그럴 때 어떤 점이 즐거운 거야?", "A": "그냥 제가 좋아하는 사람들이라서 기다리는 시간이 좋아요."}
        // More Q&A pairs
    ]
}
```

## Model Training

To train the model, run the `train.py` script. Ensure that your training and validation data directories are correctly specified.

```bash
python train.py
```

The training script will display the progress, including loss and estimated time remaining.

## Evaluation

To evaluate the trained model, use the `evaluate.py` script. This will provide accuracy, precision, recall, and F1 score.

```bash
python evaluate.py
```



## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For any questions or inquiries, please contact us at [hayun4475@gmail.com](mailto:hayun4475@gmail.com).

