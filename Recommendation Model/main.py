
import torch
from transformers import AutoModel, AutoProcessor
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from helper_functions import preprocess_data, create_text_description, find_similar_fashion_items, generate_embedding


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess data
data_path = r"C:\Users\Green_Mile\Desktop\Recommendation Model\sampled_data.csv"
df = preprocess_data(data_path)

# Generate text descriptions
df["text_description"] = df.apply(create_text_description, axis=1)
df['text_description'] = df['text_description'].astype(str)

# Load model and processor for embeddings
model_save_path = r"C:\Users\Green_Mile\Desktop\Recommendation Model\local_models\marqo-fashionSigLIP"
print("Loading model and processor from local storage...")
model = AutoModel.from_pretrained(model_save_path, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained(model_save_path, trust_remote_code=True)

# Generate embeddings
batch_size = 64
embeddings = []
for i in tqdm(range(0, len(df), batch_size), desc="Generating Embeddings"):
    batch_texts = df['text_description'].iloc[i:i+batch_size].tolist()
    processed = processor(
        text=batch_texts,
        padding=True,
        return_tensors="pt",
        truncation=True
    )
    input_ids = processed['input_ids'].to(device)
    with torch.no_grad():
        text_features = model.get_text_features(input_ids, normalize=True)
    batch_embeddings = text_features.cpu().numpy()
    embeddings.extend(batch_embeddings)

# Store embeddings and save data
output_dir = './data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Store embeddings and save data
df['embedding'] = embeddings
embedding_matrix = np.vstack(df['embedding'].to_numpy())
df['embedding'] = df['embedding'].apply(lambda x: ','.join(map(str, x)))
output_path = os.path.join(output_dir, 'fashion_recommendations_dataset.csv')
df.to_csv(output_path, index=False)
print(f"Data saved at {output_path}")
