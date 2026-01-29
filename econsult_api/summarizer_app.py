from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator
from typing import List
from services.summarizer_service import get_summarizer

router = APIRouter()

class SummarizeRequest(BaseModel):
    comments: List[str]

    @field_validator("comments", mode="before")
    @classmethod
    def clean_list(cls, v):
        if isinstance(v, str):
            # User pasted text with newlines → convert to list
            v = [x.strip() for x in v.split("\n") if x.strip()]
        if not isinstance(v, list):
            raise ValueError("Comments must be a list of strings")

        cleaned = []
        for item in v:
            if not isinstance(item, str):
                continue
            # Remove weird spacing
            item = item.replace("\\n", " ").replace("\t", " ").strip()
            item = " ".join(item.split())
            if item:
                cleaned.append(item)

        if not cleaned:
            raise ValueError("No valid comments found")

        return cleaned


class SummarizeResponse(BaseModel):
    cleaned: List[str]
    summaries: List[str]


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize(req: SummarizeRequest):

    summarizer = get_summarizer()

    try:
        cleaned, summaries = summarizer.summarize_batch(req.comments)
    except Exception as e:
        raise HTTPException(500, f"Summarization failed: {e}")

    return SummarizeResponse(cleaned=cleaned, summaries=summaries)
class GroupSummaryRequest(BaseModel):
    comments: List[str]

@router.post("/summarize_group")
async def summarize_group(req: GroupSummaryRequest):
    summarizer = get_summarizer()
    try:
        result = summarizer.summarize_group(req.comments)
        return result
    except Exception as e:
        raise HTTPException(500, f"Group summarization failed: {e}")
