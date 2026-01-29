# preprocessing/universal_preproc.py

import re
import emoji
import nltk
import torch
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Ensure correct NLTK data path
nltk.data.path.append("C:/Users/Darkg/AppData/Roaming/nltk_data")

# Initialize tools
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

# Domain noise words EXACTLY as notebook
domain_noise_words = set([
    "ref","hai","okay","ok","hello","welcome","slide",
    "course","video","module","session","training",
    "instructor","teacher","points","unit","mins","thank",
    "thanks","good","morning","afternoon","evening","lecture",
    "the","and","for","this","that","with","you","are","was",
    "were","will","shall","from","have","had","has","been",
    "but","not","can","may","would","should","could","a","an",
    "to","of","in","on","at","by","it","they","them","we","i",
    "is","as","or","be","our","your","their","my","please",
    "kindly","thanks","yes","no","need","view","angle","tbh",
    "compliance","change","saying","good","great","super","TBH",
    "forward","looking","toh","thoda"
])

# Emoji → word mapping (from notebook)
emoji_map = {
    "😊": "happy", "😍": "love", "😂": "funny", "😢": "sad",
    "😡": "angry", "👍": "good", "👎": "bad"
}

def interpret_emojis(text):
    return emoji.replace_emoji(text, replace=lambda e: " " + emoji_map.get(e, ""))


# Hinglish replacements EXACT from notebook
hinglish_replacements = {
    "hai": "is", "hain": "are", "tha": "was", "thi": "was",
    "hogaya": "is done", "hoga": "will be",
    "accha": "good", "acha": "good",
    "bahut": "very", "bohot": "very",
    "thoda": "little", "zyada": "more",
    "nahi": "not", "nhi": "not",
    "kya": "what", "kyu": "why", "kab": "when",
    "mera": "my", "meri": "my", "tum": "you"
}

def normalize_hinglish(text):
    text = text.lower()
    text = interpret_emojis(text)
    for k, v in hinglish_replacements.items():
        text = re.sub(rf"\b{k}\b", v, text)
    return text.strip()

# Hinglish detection (Notebook logic)
def is_hinglish(text):
    hinglish_keywords = [
        "hai","hain","tha","thi","nahi","accha","acha","bahut",
        "bohot","zyada","kya","kyu","mera","meri","tum"
    ]
    count = sum(word in text.lower().split() for word in hinglish_keywords)
    return count >= 2

# Load Translators ONLY ONCE
def load_translator():
    model_name = "facebook/nllb-200-distilled-600M"
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "translation",
        model=model_name,
        tokenizer=model_name,
        src_lang="eng_Latn",
        tgt_lang="eng_Latn",
        device=device
    )

TRANSLATOR = load_translator()

def translate_if_needed(text):
    if not isinstance(text, str):
        return text

    if is_hinglish(text):
        cleaned = normalize_hinglish(text)
        try:
            out = TRANSLATOR(cleaned, max_new_tokens=150)
            return out[0]["translation_text"].strip()
        except:
            return cleaned
    else:
        return text

# Basic cleaners
def remove_urls(t): return re.sub(r"http\S+|www\.\S+", "", t)
def remove_mentions(t): return re.sub(r"@\w+", "", t)
def remove_html(t): return re.sub(r"<.*?>", "", t)
def remove_hashtags(t): return t.replace("#", "")
def remove_special_chars(t): return re.sub(r"[^a-zA-Z\s']", " ", t)
def normalize_spaces(t): return re.sub(r"\s+", " ", t).strip()

slang_map = {
    "lol": "funny", "imo": "in my opinion", "btw": "by the way",
    "idk": "don't know", "omg": "surprised", "asap": "soon"
}

def clean_and_lemmatize(text):
    tokens = word_tokenize(text.lower())
    processed = []

    for token in tokens:
        if token in domain_noise_words:
            continue
        if token in slang_map:
            processed.extend(slang_map[token].split())
            continue
        if token in stop_words:
            continue

        lemma = lemmatizer.lemmatize(token)
        stem = stemmer.stem(lemma)
        processed.append(stem)

    return " ".join(processed)

# FULL PIPELINE (Final)
def preprocess_comment(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""

    text = translate_if_needed(text)
    text = remove_urls(text)
    text = remove_mentions(text)
    text = remove_html(text)
    text = remove_hashtags(text)
    text = remove_special_chars(text)
    text = normalize_spaces(text)
    text = clean_and_lemmatize(text)

    return text

def preprocess_batch(comments):
    return [preprocess_comment(c) for c in comments]
