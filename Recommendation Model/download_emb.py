from transformers import AutoModel, AutoProcessor

# Define the local save path
model_save_path = "./local_models/marqo-fashionSigLIP"

# Download and save the model
model = AutoModel.from_pretrained('Marqo/marqo-fashionSigLIP', trust_remote_code=True)
model.save_pretrained(model_save_path)

# Download and save the processor
processor = AutoProcessor.from_pretrained('Marqo/marqo-fashionSigLIP', trust_remote_code=True)
processor.save_pretrained(model_save_path)

print("Model and processor saved locally at:", model_save_path)
