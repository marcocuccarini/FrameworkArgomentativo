from model.BERTModel import BERT_Model
from sentence_transformers import SentenceTransformer
import json
import numpy as np

def encode_json(model, json_file):
    """Encodes the textual arguments using the provided model."""
    for item in json_file:
        for argument in item.get('arguments', []):
            # Convert ndarray to list for JSON serialization
            argument['sentences'] = model.text_embedding(argument['sentences']).tolist()

    return json_file

def extract_json(path):
    """Extracts JSON data from a given file path."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, path):
    """Saves JSON data to a specified file path."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    # Load the dataset
    json_file = extract_json("dataset/KB.json")

    # Initialize the BERT model
    bert_model = BERT_Model("multi-qa-mpnet-base-dot-v1", SentenceTransformer)

    # Encode the textual arguments
    json_file_encoded = encode_json(bert_model, json_file)

    # Save the encoded data
    save_json(json_file_encoded, "dataset/KB_enc.json")

    print("Encoding completed and saved to data.json")