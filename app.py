# app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
import tensorflow as tf
import numpy as np
from typing import List
import re

# Import wordcloud function
from wordcloud_generator import generate_wordcloud

# FastAPI setup
app = FastAPI(
    title="Comment Sentiment Classifier API",
    description="Classify comments as Positive, Neutral, or Negative",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class CommentRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float

class WordCloudRequest(BaseModel):
    comments: List[str]

class WordCloudResponse(BaseModel):
    wordcloud_base64: str

# Load model and tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert_sentiment_model")
model = TFDistilBertForSequenceClassification.from_pretrained("distilbert_sentiment_model")

# Label mapping
LABEL_MAPPING = {0: "negative", 1: "neutral", 2: "positive"}

# Preprocess text
def preprocess_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r'^\d+\.\s*', '', text.strip())
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Prediction
def predict_sentiment(text: str):
    processed_text = preprocess_text(text)
    if not processed_text:
        return "neutral", 0.0

    inputs = tokenizer(
        processed_text,
        return_tensors="tf",
        truncation=True,
        padding=True,
        max_length=128
    )

    logits = model(inputs).logits
    probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
    predicted_label_id = int(np.argmax(probs))
    predicted_sentiment = LABEL_MAPPING[predicted_label_id]
    confidence = float(np.max(probs))

    return predicted_sentiment, confidence

# Routes
@app.get("/")
async def root():
    return {"status": "active", "endpoint": "/predict"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_comment(request: CommentRequest):
    try:
        sentiment, confidence = predict_sentiment(request.text)
        return PredictionResponse(
            text=request.text,
            sentiment=sentiment,
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/wordcloud", response_model=WordCloudResponse)
async def wordcloud_endpoint(request: WordCloudRequest):
    """
    Accepts a list of comments and returns a base64 encoded PNG word cloud
    """
    try:
        if not request.comments:
            raise HTTPException(status_code=400, detail="No comments provided")
        
        img_base64 = generate_wordcloud(request.comments)
        return WordCloudResponse(wordcloud_base64=img_base64)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
