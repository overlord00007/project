import os
import re
import json
from statistics import mode
from typing import List, Dict

# Core AI/ML libraries
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Preprocessing utilities
import nltk
import emoji
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from pydantic import BaseModel

#Summarisation imports:

from summarizer_app import router as summary_router


# FastAPI setup
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse



# --- 1. SETUP AND CACHING ---

# Define directories and constants
MODEL_DIR = "./models"
ENSEMBLE_CONFIG_PATH = os.path.join(MODEL_DIR, "top3_models.json")
# Define the label mapping from the training notebook
ID2LABEL = {0: 'negative', 1: 'neutral', 2: 'positive'}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

# Global cache for models and tokenizers (to prevent re-loading on each request)
MODEL_CACHE = {}

# NLP Components (instantiated globally for performance)
# NLTK components must be downloaded first (Step 6)
try:
    LEMMA = WordNetLemmatizer()
    STEMMER = PorterStemmer()
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    # This will be caught if NLTK is not downloaded
    print("NLTK data not found. Please run NLTK download step.")
    LEMMA, STEMMER, STOP_WORDS = None, None, None

# Custom domain noise words used in the notebook's preprocessing cell (Cell 9)
DOMAIN_NOISE_WORDS = set([
    "ref", "hai", "okay", "ok", "hello", "welcome", "slide",
    "course", "video", "module", "session", "training",
    "instructor", "teacher", "points", "unit", "mins", "thank",
    "thanks", "good", "morning", "afternoon", "evening", "lecture",
    "the", "and", "for", "this", "that", "with", "you", "are", "was",
    "were", "will", "shall", "from", "have", "had", "has", "been",
    "but", "not", "can", "may", "would", "should", "could", "a", "an",
    "to", "of", "in", "on", "at", "by", "it", "they", "them", "we", "i",
    "is", "as", "or", "be", "our", "your", "their", "my", "please",
    "kindly", "thanks", "yes", "no", "need", "view", "angle", "tbh",
    "compliance", "change", "saying", "good", "great", "super", "TBH",
    "forward", "looking", "toh", "thoda"
])
# Emoji mapping used in the notebook
EMOJI_MAP = {"😊": "happy", "😍": "love", "😂": "funny", "😢": "sad", "😡": "angry", "👍": "good", "👎": "bad"}
# Slang mapping used in the notebook
SLANG_MAP = {"lol": "funny", "imo": "in my opinion", "btw": "by the way", "idk": "don't know", "omg": "surprised", "asap": "soon"}

# Check for CUDA device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. PREPROCESSING FUNCTIONS (Recreated from Jupyter Notebook) ---

def interpret_emojis(text: str) -> str:
    """Interprets emojis using the same map as the training notebook."""
    return emoji.replace_emoji(text, replace=lambda ch, data=None: " " + EMOJI_MAP.get(ch, ""))

def clean_text_basic(text: str) -> str:
    """Performs basic cleaning steps."""
    text = interpret_emojis(text)
    text = re.sub(r"http\S+|www\.\S+", "", text) 
    text = re.sub(r"@\w+", "", text)           
    text = re.sub(r"<.*?>", "", text)          
    text = text.replace("#", "")               
    text = re.sub(r"[^a-zA-Z\s']", " ", text)  
    text = re.sub(r"\s+", " ", text).strip()   
    return text

def clean_and_lemmatize(text: str) -> str:
    """Performs tokenization, slang expansion, stopword removal, domain noise removal, and stemming/lemmatization."""
    if LEMMA is None or STEMMER is None or STOP_WORDS is None:
        # If this happens, the NLTK download step (Step 6) was skipped or failed.
        # We raise a runtime error here to prevent incorrect results.
        raise RuntimeError("NLTK preprocessing components are not loaded. Run NLTK download step (Step 6).")

    # Only process if text is not empty after basic cleaning
    if not text:
        return ""
        
    # Use nltk.word_tokenize and convert to lowercase
    tokens = nltk.word_tokenize(text.lower())
    processed = []

    for token in tokens:
        # Remove domain noise
        if token in DOMAIN_NOISE_WORDS:
            continue

        # Slang expansion
        if token in SLANG_MAP:
            processed.extend(SLANG_MAP[token].split())
            continue

        # Remove stopwords
        if token in STOP_WORDS:
            continue

        # Lemma → Stem (same as training notebook)
        lemma = LEMMA.lemmatize(token)
        stem = STEMMER.stem(lemma)

        processed.append(stem)

    return " ".join(processed)

def preprocess_comment(comment: str) -> str:
    """Full preprocessing pipeline as in the training notebook."""
    if not isinstance(comment, str):
        return ""
    
    # 1. Basic Cleaning
    cleaned = clean_text_basic(comment)

    # 2. Linguistic Processing (includes cleaning, slang, stopword, domain noise removal)
    final_cleaned = clean_and_lemmatize(cleaned)

    return final_cleaned

# --- 3. MODEL LOADING AND PREDICTION ---

def load_models_to_cache():
    """Loads the required models and tokenizers into the global cache."""
    print(f"Device set to: {DEVICE}")
    print("Attempting to load models into cache...")
    
    if not os.path.exists(ENSEMBLE_CONFIG_PATH):
        print(f"ERROR: Ensemble config not found at {ENSEMBLE_CONFIG_PATH}. Check your 'models' folder.")
        return

    try:
        with open(ENSEMBLE_CONFIG_PATH, "r") as f:
            config = json.load(f)
        top3_models = config['top3']
        print(f"Top 3 models to load: {top3_models}")

    except Exception as e:
        print(f"Error loading ensemble config: {e}")
        return

    for model_name_raw in top3_models:
        # Map HF model name to local folder structure (as saved in the notebook)
        if "bert-base-uncased" in model_name_raw:
            local_dir_name = "bert_sentiment"
        elif "distilbert-base-uncased" in model_name_raw:
            local_dir_name = "distilbert_sentiment"
        elif "roberta-base" in model_name_raw:
            local_dir_name = "roberta_sentiment"
        else:
            print(f"Skipping unknown model: {model_name_raw}")
            continue

        local_dir = os.path.join(MODEL_DIR, local_dir_name)

        if not os.path.exists(local_dir):
            print(f"Skipping {model_name_raw}: directory not found at {local_dir}")
            continue

        try:
            tokenizer = AutoTokenizer.from_pretrained(local_dir)
            model = AutoModelForSequenceClassification.from_pretrained(local_dir)
            model.to(DEVICE)
            model.eval()

            MODEL_CACHE[model_name_raw] = {
                "tokenizer": tokenizer,
                "model": model
            }
            print(f"Successfully loaded {model_name_raw} from {local_dir} to {DEVICE}")

        except Exception as e:
            print(f"Failed to load {model_name_raw} from {local_dir}: {e}")

def get_ensemble_prediction(text: str) -> str:
    """Gets predictions from the top 3 models and performs a majority vote."""
    if not MODEL_CACHE:
        raise RuntimeError("Model cache is empty. Models failed to load at startup.")

    votes = []
    
    for name, obj in MODEL_CACHE.items():
        tokenizer = obj["tokenizer"]
        model = obj["model"]

        enc = tokenizer(
            [text],
            truncation=True,
            padding=True,
            return_tensors='pt'
        ).to(DEVICE) 

        with torch.no_grad():
            outputs = model(**enc)
            pred_id = torch.argmax(outputs.logits, dim=1).item()
            votes.append(pred_id)

    if not votes:
        return "ERROR: No models contributed a prediction."
        
    final_pred_id = mode(votes)
    return ID2LABEL[final_pred_id]

# --- 4. FASTAPI ENDPOINTS ---

app = FastAPI(
    title="eConsult Sentiment Analysis Ensemble API",
    description="Deploys a majority-vote ensemble of fine-tuned Hugging Face Transformers (RoBERTa, DistilBERT, BERT) for specialized legal/corporate sentiment analysis.",
    version="1.0.0"
)

app.include_router(summary_router, prefix="/api")

from pydantic import BaseModel
from fastapi import APIRouter

router = APIRouter()

class GroupSummaryRequest(BaseModel):
    comments: list[str]



# Request body model
class SentimentRequest(BaseModel):
    comment: str

# Response body model
class SentimentResponse(BaseModel):
    comment: str
    preprocessed_text: str
    predicted_sentiment: str

@app.on_event("startup")
async def startup_event():
    """Load models and NLTK data on application startup."""
    global LEMMA, STEMMER, STOP_WORDS

    import nltk

    # Add explicit NLTK path
    nltk.data.path.append("C:/Users/Darkg/AppData/Roaming/nltk_data")
    print("NLTK paths:", nltk.data.path)

    # Try loading NLTK data but DO NOT crash
    try:
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("corpora/stopwords")
        nltk.data.find("corpora/wordnet")
        print("NLTK verification successful.")
    except LookupError as e:
        print("WARNING: NLTK resources not fully found. Continuing startup anyway.")
        print(e)

    # Initialize processors (will fail gracefully if data missing)
    try:
        LEMMA = WordNetLemmatizer()
        STEMMER = PorterStemmer()
        STOP_WORDS = set(stopwords.words("english"))
    except Exception as e:
        print("WARNING: Could not initialize NLTK lemmatizer or stopwords:", e)
        LEMMA = None
        STEMMER = None
        STOP_WORDS = None

    # Load models
    load_models_to_cache()
    if not MODEL_CACHE:
        print("WARNING: Model cache is empty — predictions will not work.")



@app.get("/", summary="Health Check")
def read_root():
    """Simple health check endpoint."""
    return {"status": "ok", "message": "Sentiment Analysis Ensemble API is running.", "device": str(DEVICE)}

@app.post("/predict_sentiment", response_model=SentimentResponse, summary="Analyze Comment Sentiment")
async def predict_sentiment(request: SentimentRequest):
    """
    Analyzes the sentiment of a single comment using the trained ensemble model.
    """
    if not MODEL_CACHE:
         raise HTTPException(
            status_code=503, 
            detail="Models are not loaded. Service unavailable. Check server startup logs."
        )

    # 1. Apply the preprocessing pipeline
    try:
        preprocessed_text = preprocess_comment(request.comment)
    except RuntimeError as e:
         raise HTTPException(
            status_code=500, 
            detail=str(e)
        )
    
    if not preprocessed_text:
        return SentimentResponse(
            comment=request.comment,
            preprocessed_text="Comment resulted in empty string after preprocessing (likely filtered out as noise).",
            predicted_sentiment="CANNOT_CLASSIFY"
        )

    # 2. Get ensemble prediction (majority vote)
    try:
        sentiment = get_ensemble_prediction(preprocessed_text)
    except Exception as e:
         raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed during ensemble voting: {e}"
        )

    # 3. Return the result
    return SentimentResponse(
        comment=request.comment,
        preprocessed_text=preprocessed_text,
        predicted_sentiment=sentiment.upper()
    )