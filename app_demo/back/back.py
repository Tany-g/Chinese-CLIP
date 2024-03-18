from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
import numpy as np
import torch 
from PIL import Image

import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
print("Available models:", available_models())  
# Available models: ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']
from PIL import Image
import base64
import io
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read())
    return encoded_string.decode('utf-8')

def base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image



class clip_model:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = load_from_name("ViT-B-16", device=self.device, download_root='./')
        self.model.eval()

    def processimg(self,image:"str"):
        image = self.preprocess(Image.open("examples/pokemon.jpeg")).unsqueeze(0).to(self.device)
        image_features = self.model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True) 
        return image_features
    
    
    def processtxt(self,txt):
        text = clip.tokenize([txt]).to(self.device)
        text_features = self.model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)  
        return text_features


app = FastAPI()
model = clip_model()
# Connect to the SQLite database

conn = sqlite3.connect('database.db')
cursor = conn.cursor()


# Define data model for request body
class ImageUploadData(BaseModel):
    image_base64: str

class ImageSearchData(BaseModel):
    image_base64: str
    num_results: int

class TextSearchData(BaseModel):
    text: str
    num_results: int

# API 1: Image Upload and Clip Feature Extraction
@app.post("/upload")
def upload_image(image_data: ImageUploadData):
    # Your code for image upload and Clip feature extraction goes here
    # Placeholder for demonstration purposes
    return {"message": "Image uploaded and Clip features extracted successfully."}

# API 2: Image Retrieval
@app.post("/retrieve_images")
def retrieve_images(image_search_data: ImageSearchData):
    # Your code for image retrieval using KNN goes here
    # Placeholder for demonstration purposes
    return {"message": f"{image_search_data.num_results} similar images retrieved successfully."}

# API 3: Text Retrieval
@app.post("/retrieve_text")
def retrieve_text(text_search_data: TextSearchData):
    # Tokenize and vectorize the input text

    return {"similar_texts": ""}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
