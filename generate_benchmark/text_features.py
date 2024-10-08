#!pip install convokit
import torch
from transformers import BertTokenizer, BertModel
import json
import numpy as np
import random
import os
from convokit import Corpus, download

from tqdm import tqdm


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_all_texts(file_path):
    texts = []
    with open(file_path, 'r', encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            texts.append(data['text'])
    return texts


def extract_text_features(texts, tokenizer, model, batch_size, write_path):
    features = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        tokens = tokenizer(batch_texts, return_tensors='pt', max_length=512,
                        truncation=True, padding='max_length')

        tokens = {key: value.to(device) for key, value in tokens.items()}

        with torch.no_grad():
            outputs = model(**tokens)

        pooled_output = outputs.pooler_output.cpu().numpy()
        features.append(pooled_output)
    features = np.concatenate(features, axis=0)
    print(features.shape)
    np.save(write_path, features)


if __name__ == "__main__":
    seed_everything(42)

    file_path = './utterances.jsonl'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    model.to(device)

    texts = get_all_texts(file_path)
    print(texts[:5])
    extract_text_features(texts, tokenizer, model, 32, "./bert_movie-corpus_text_features.npy")
    print("Done!")
