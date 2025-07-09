import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import requests
from PIL import Image
from io import BytesIO
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import random


def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df = df.drop_duplicates()

    df = df.drop_duplicates()

    # List of accessory-related columns to remove
    accessory_columns = [
        "image_url_1", "gender",
        "accessory_type", "bag_type", "belt_type", "heels_width", "heels_height",
        "boots-shoe_type", "shoe_material", "bracelet_type", "glasses_type",
        "headwear_type", "necklet_type", "neckwear_type", "sandals-shoe_type",
        "shorts_type", "slippers-shoe_type", "sneakers_type", "sportswear_type",
        "sweatshirt_type", "swimwear_type", "tank_type"
    ]

    # Drop the columns from the DataFrame
    return df.drop(columns=accessory_columns, errors='ignore')




# Define the function for text description
# Generates a formatted descriptive text while omitting empty values.
def create_text_description(row):

    def get_value(col_name):
        """Return the value if it's not null, otherwise return an empty string."""
        return str(row[col_name]).strip() if pd.notnull(row[col_name]) else ""

    # Structured description format
    description_parts = []

    # Main clothing attributes
    category = get_value("category")
    occasion = get_value("occasion")
    style = get_value("style")
    material = get_value("material")
    colors = get_value("colors")
    pattern = get_value("pattern")
    more_attributes = get_value("more_attributes")

    tops_fit = get_value("tops_fit")
    tops_length = get_value("tops_length")
    sleeve_type = get_value("sleeve_type")
    sleeve_length = get_value("sleeve_length")
    neckline_type = get_value("neckline_type")
    blazer_neckline_type = get_value("blazer_neckline_type")

    overclothes_type = get_value("overclothes_type")
    overclothes_neckline_type = get_value("overclothes_neckline_type")
    overclothes_closure = get_value("overclothes_closure")
    overclothes_sleeveless_type = get_value("overclothes-sleeveless_type")

    bottoms_fit = get_value("bottoms_fit")
    bottoms_length = get_value("bottoms_length")
    skirt_type = get_value("skirt_type")
    skirt_length = get_value("skirt_length")
    jumpsuit_length = get_value("jumpsuit_length")

    waist_type = get_value("waist_type")
    poncho_type = get_value("poncho_type")
    brand = get_value("brand")

    # Build description dynamically
    if category:
        description_parts.append(f"This is a {category} designed for {occasion}." if occasion else f"This is a {category}.")
    if style:
        description_parts.append(f"It belongs to the {style} style category.")
    if material:
        description_parts.append(f"It is made from {material}.")
    if colors:
        description_parts.append(f"The item is available in {colors}.")
    if pattern:
        description_parts.append(f"It features a {pattern} pattern.")
    if more_attributes:
        description_parts.append(f"Additional design elements include {more_attributes}.")

    # Tops description
    tops_description = []
    if tops_fit:
        tops_description.append(f"a {tops_fit} fit")
    if tops_length:
        tops_description.append(f"a {tops_length} length")
    if sleeve_type:
        tops_description.append(f"a {sleeve_type} sleeve type")
    if sleeve_length:
        tops_description.append(f"a {sleeve_length} sleeve length")
    if neckline_type:
        tops_description.append(f"a {neckline_type} neckline")

    if tops_description:
        description_parts.append("For tops, it has " + ", ".join(tops_description) + ".")

    if blazer_neckline_type:
        description_parts.append(f"If it's a blazer, the neckline follows the {blazer_neckline_type} style.")

    # Overclothes description
    overclothes_description = []
    if overclothes_type:
        overclothes_description.append(f"a {overclothes_type} type")
    if overclothes_neckline_type:
        overclothes_description.append(f"a {overclothes_neckline_type} neckline")
    if overclothes_closure:
        overclothes_description.append(f"a {overclothes_closure} closure")

    if overclothes_description:
        description_parts.append("If it's an overclothes item, it has " + ", ".join(overclothes_description) + ".")

    if overclothes_sleeveless_type:
        description_parts.append(f"If sleeveless, it belongs to the {overclothes_sleeveless_type} category.")

    # Bottoms description
    bottoms_description = []
    if bottoms_fit:
        bottoms_description.append(f"a {bottoms_fit} fit")
    if bottoms_length:
        bottoms_description.append(f"a {bottoms_length} length")

    if bottoms_description:
        description_parts.append("For bottoms, it features " + ", ".join(bottoms_description) + ".")

    if skirt_type and skirt_length:
        description_parts.append(f"If it's a skirt, it falls under the {skirt_type} category with a {skirt_length} length.")
    elif skirt_type:
        description_parts.append(f"If it's a skirt, it falls under the {skirt_type} category.")
    elif skirt_length:
        description_parts.append(f"If it's a skirt, it has a {skirt_length} length.")

    if jumpsuit_length:
        description_parts.append(f"If it's a jumpsuit, it has a {jumpsuit_length} length.")

    if waist_type:
        description_parts.append(f"It has a {waist_type} waist design.")

    if poncho_type:
        description_parts.append(f"If it’s a poncho, it follows the {poncho_type} type.")

    # Brand information
    if brand:
        description_parts.append(f"The item is produced by the brand {brand}.")

    return " ".join(description_parts)

#  Generates an embedding for the given text using a pretrained model.
def generate_embedding(model, processor, text, device):

    processed = processor(text=[text], padding=True, return_tensors="pt", truncation=True)
    input_ids = processed['input_ids'].to(device)
    with torch.no_grad():
        embedding = model.get_text_features(input_ids, normalize=True)
    return embedding[0].cpu().numpy()


# Finds the top_k most similar fashion items using cosine similarity.
def find_similar_fashion_items(query_text, model, processor, embedding_matrix, df, device, top_k=10):

    query_embedding = generate_embedding(model, processor, query_text, device).reshape(1, -1)
    similarities = cosine_similarity(query_embedding, embedding_matrix)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    recommended_items = df.iloc[top_indices][['barcode']]
    return recommended_items


