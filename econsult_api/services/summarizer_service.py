# services/summarizer_service.py

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from preprocessing.universal_preproc import (
    translate_if_needed,
    remove_urls,
    remove_mentions,
    remove_html,
    remove_hashtags,
    remove_special_chars,
    normalize_spaces
)

SUMMARY_MODEL_DIR = "./models/summarizer"


class SummarizerEngine:
    def __init__(self):
        self.device_id = 0 if torch.cuda.is_available() else -1
        self.load()

    def load(self):
        print("Loading BART-LARGE model from:", SUMMARY_MODEL_DIR)
        tokenizer = AutoTokenizer.from_pretrained(SUMMARY_MODEL_DIR)
        model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARY_MODEL_DIR)

        self.summarizer = pipeline(
            "summarization",
            model=model,
            tokenizer=tokenizer,
            device=self.device_id
        )
        print("Summarizer loaded successfully.")

    # ---------------------------------------------------------
    # CLEANING BEFORE SUMMARIZATION  (NO STEM, NO LEMMA)
    # ---------------------------------------------------------
    def summarization_clean(self, text: str):
        if not isinstance(text, str):
            return ""

        text = translate_if_needed(text)
        text = remove_urls(text)
        text = remove_mentions(text)
        text = remove_html(text)
        text = remove_hashtags(text)
        text = remove_special_chars(text)
        text = normalize_spaces(text)
        return text.strip()

    # ---------------------------------------------------------
    # SINGLE COMMENT SUMMARY
    # ---------------------------------------------------------
    def summarize_one(self, original_text: str):
        cleaned = self.summarization_clean(original_text)

        words = cleaned.split()
        if len(words) < 10:
            return cleaned

        dynamic_max = min(80, max(12, len(words) // 2))
        dynamic_min = min(15, dynamic_max - 5)
    
        try:
            out = self.summarizer(
                cleaned,
                max_length=dynamic_max,
                min_length=dynamic_min,
                do_sample=False
            )
            summary = out[0]["summary_text"].strip()
            return summary if summary else cleaned[:150]
        except Exception:
            return cleaned[:150]

    # ---------------------------------------------------------
    # MULTI COMMENT SUMMARY (PER COMMENT)
    # ---------------------------------------------------------
    def summarize_batch(self, comments):
        cleaned = []
        summaries = []

        for c in comments:
            clean = self.summarization_clean(c)
            cleaned.append(clean)
            summaries.append(self.summarize_one(c))

        return cleaned, summaries

    # ---------------------------------------------------------
    # GROUP SUMMARY (CELL 20 LOGIC)
    # ---------------------------------------------------------
    def summarize_group(self, comments, chunk_size=50):

        cleaned = [
            self.summarization_clean(c)
            for c in comments
            if isinstance(c, str) and c.strip()
        ]

        if not cleaned:
            return {
                "cleaned_comments": [],
                "chunk_summaries": [],
                "final_summary": ""
            }

        # -------- CHUNK SUMMARIES --------
        chunk_summaries = []

        for i in range(0, len(cleaned), chunk_size):
            group = cleaned[i:i + chunk_size]
            merged = " ".join(group)

            try:
                out = self.summarizer(
                    merged,
                    max_length=200,
                    min_length=30,
                    do_sample=False
                )
                chunk_summary = out[0]["summary_text"]
            except Exception:
                chunk_summary = merged[:300]

            chunk_summaries.append(chunk_summary)

        # -------- FINAL SUMMARY --------
        merged_all = " ".join(chunk_summaries)

        try:
            out = self.summarizer(
                merged_all,
                max_length=150,
                min_length=40,
                do_sample=False
            )
            final_summary = out[0]["summary_text"]
        except Exception:
            final_summary = merged_all[:500]

        return {
            "cleaned_comments": cleaned,
            "chunk_summaries": chunk_summaries,
            "final_summary": final_summary
        }


# Singleton
_SUM_INSTANCE = None

def get_summarizer():
    global _SUM_INSTANCE
    if _SUM_INSTANCE is None:
        _SUM_INSTANCE = SummarizerEngine()
    return _SUM_INSTANCE
