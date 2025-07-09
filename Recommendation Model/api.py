from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModel, AutoProcessor
import pandas as pd
import numpy as np
import cohere
from helper_functions import (
    preprocess_data, create_text_description, find_similar_fashion_items, generate_embedding
)


# Initialize FastAPI app
app = FastAPI(
    title="Fashion Recommendation API",
    description="API that provides fashion recommendations based on body type",
    version="1.0.1"
)

# Load the model and dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_save_path = r"C:\Users\Green_Mile\Desktop\Recommendation Model\local_models\marqo-fashionSigLIP"
model = AutoModel.from_pretrained(model_save_path, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained(model_save_path, trust_remote_code=True)

# Load precomputed embeddings
dataset_path = r"C:\Users\Green_Mile\Desktop\Recommendation Model\data\fashion_recommendations_dataset.csv"
df = pd.read_csv(dataset_path)
df['embedding'] = df['embedding'].apply(lambda x: np.fromstring(x, sep=','))
embedding_matrix = np.vstack(df['embedding'].to_numpy())


# Initialize Cohere
API_KEY = '3GuDMsWlQRGnwC8crhZUqmoh39hqrLSlpgPQVQTS'
co = cohere.Client(API_KEY)

# Request schema
class RecommendationRequest(BaseModel):
    body_type: str  

# Function to dynamically generate fashion advice using Cohere
def analyze_body_type(body_type):
    prompt = f"""
    You are a professional female fashion consultant specializing in personalized styling based on body types.

    Please provide a concise list of characteristics for a {body_type} body type based on the following features:

    - **Tops Fit:** [Short answer]
    - **Sleeve Type:** [Short answer]
    - **Neckline Type:** [Short answer]
    - **Sleeve Length:** [Short answer]
    - **Waist Type:** [Short answer]
    - **Bottoms Length:** [Short answer]
    - **Skirt Type:** [Short answer]
    - **Bottoms Fit:** [Short answer]
    - **Bottoms Type:** [Short answer]

    Additionally, provide recommendations for:
    - **Style:** [Short answer]
    - **Colors:** [Short answer]
    - **Patterns:** [Short answer]

    If you don't have any ideas for a specific feature, skip it without explanation.

    **Format:** Follow the bullet-point format exactly as shown above without additional explanations or paragraphs.

    Write in a concise, insightful, and professional tone.
    """


    response = co.generate(
        model='command-xlarge-nightly',
        prompt=prompt,
        max_tokens=500,
        temperature=0.7
    )

    # Extract and return the generated text
    result_text = response.generations[0].text.strip()
    return result_text

# API root route
@app.get("/")
def root():
    return {"message": "Fashion Recommendation API is running successfully!"}

# Recommendation endpoint
@app.post("/recommend")
def recommend_clothes(request: RecommendationRequest):
    body_type = request.body_type.lower()

    # Generate personalized advice dynamically
    try:
        fashion_advice = analyze_body_type(body_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating fashion advice: {str(e)}")

    # Find similar fashion items based on advice
    try:
        recommendations = find_similar_fashion_items(
            fashion_advice, model, processor, embedding_matrix, df, device
        )
        recommended_barcodes = recommendations["barcode"].tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

    return {
        "body_type": body_type,
        "fashion_advice": fashion_advice,
        "recommended_items": recommended_barcodes
    }

# Run the server using: uvicorn api:app --reload
# http://127.0.0.1:8000/docs