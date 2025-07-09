# Fashion Recommendation System Using LLMs and Embeddings

This project is an end‑to‑end intelligent **fashion recommendation system** that combines computer vision, natural language processing, and large language models (LLMs).  
Users receive **personalized clothing recommendations** based on their body type by analyzing fashion attributes, generating style advice, and performing similarity‑based matching with vector embeddings.

The system is implemented in **Python**—Jupyter notebooks for training, **FastAPI** for serving recommendations, and models such as **EfficientNet** and **Marqo‑FashionSigLIP** for classification and embedding generation. The recommendation process is powered by the **Cohere LLM** and similarity search via **cosine similarity**.

The full pipeline consists of:

- Clothing image classification using transfer learning  
- Metadata‑driven product description generation  
- Embedding generation with a multimodal transformer  
- Personalized style advice using a large language model  
- Recommendation delivery through a RESTful API  

---

## Project Components

### 1. Clothing Image Classification

A CNN‑based model was developed using transfer learning to classify images of clothing items into predefined categories.

Techniques used:

- Dataset cleaning and preprocessing (removing NaNs, duplicates, irrelevant columns)  
- One‑hot encoding of categorical features and multi‑label column handling  
- Data augmentation to expand the dataset with flipped and rotated versions  
- Transfer learning with **EfficientNet‑B0**, replacing the final layer to suit fashion‑specific categories  

The model achieved high accuracy by leveraging pre‑trained ImageNet features, enabling efficient training and good generalization on the fashion dataset.

### 2. Fashion Recommendation Using LLM & Embeddings

Personalized recommendations workflow:

1. Aggregate descriptive metadata (material, fit, color, pattern) for each product.  
2. Embed the generated text with **Marqo‑FashionSigLIP** into vector space.  
3. Analyze user body‑type input (e.g., “hourglass”) with **Cohere LLM** to generate stylistic advice.  
4. Embed the advice and compare it with product embeddings via cosine similarity.  
5. Return the **top 10** most similar items to the user.  

This architecture adapts to many body types and can be extended with improved LLMs or fine‑tuned embeddings.

### 3. FastAPI‑Based Recommendation API

An interactive REST API serves personalized clothing recommendations:

- **POST** endpoint `/recommend` accepts JSON containing the user’s body type.  
- The API generates advice via Cohere, embeds it, and performs similarity matching.  
- Results include a list of the most relevant clothing items, returned as barcodes.  

FastAPI automatically provides **Swagger UI** and **ReDoc** for easy testing.

---

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download and Save the Embedding Model

```bash
python download_emb.py
```

This will download the Marqo‑FashionSigLIP model from Hugging Face and save it in `local_models/`.

### 3. Generate Embeddings for Clothing Data

```bash
python main.py
```

This script:

- Loads and cleans `sampled_data.csv`.  
- Generates descriptive text for each item.  
- Creates and saves vector embeddings into `fashion_recommendations_dataset.csv`.

### 4. Start the API Server

```bash
python api.py
```

The server starts at <http://127.0.0.1:8000> and loads the pre‑computed embeddings.

### 5. Test the API with Swagger UI

Open your browser at <http://127.0.0.1:8000/docs> and use the `POST /recommend` endpoint with JSON such as:

```json
{
  "body_type": "hourglass"
}
```

Replace `"hourglass"` with `"apple"`, `"rectangle"`, or any custom body type.

---

## How It Works

### Body‑Type Analysis (Cohere LLM)

The LLM receives the user’s body type and returns tailored fashion advice based on well‑known stylistic guidelines.

### Embedding & Similarity Search

The generated advice is embedded via **Marqo‑FashionSigLIP** and compared with item embeddings using **cosine similarity**.

### Recommendation Output

The API returns a JSON object that contains:

- The personalized advice text  
- A list of the **10** most relevant clothing items (by barcode)  

---

> **Enjoy smarter styling!** Feel free to fork, open issues, or submit PRs to enhance the recommendation pipeline.
