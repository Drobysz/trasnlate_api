import os
from typing import List, Optional

import deepl
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Req(BaseModel):
    targetLang: str = Field(..., description="Target language code, e.g. EN, FR, RU")
    texts: List[str] = Field(..., description="List of input strings to translate")


_translator: Optional[deepl.Translator] = None

def get_translator() -> deepl.Translator:
    global _translator
    if _translator is not None:
        return _translator

    key = os.getenv("DEEPL_API_KEY")
    if key:
        # Normalize: remove accidental whitespace and surrounding quotes
        key = key.strip().strip('"').strip("'")
    if not key or not key.strip():
        raise HTTPException(status_code=500, detail="Missing DEEPL_API_KEY")

    _translator = deepl.Translator(key)
    return _translator


@app.get("/")
def root():
    return {"status": "ok", "message": "Translation API is running. Use POST /api/translate"}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/debug/env")
def debug_env():
    key = os.getenv("DEEPL_API_KEY")
    if key:
        key = key.strip().strip('"').strip("'")
    return {
        "has_key": bool(key),
        "key_len": len(key) if key else 0,
        "key_tail": key[-3:] if key and len(key) >= 3 else None,
    }


@app.post("/api/translate")
def translate(req: Req):
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts must be a non-empty list")

    if not all(isinstance(t, str) and t.strip() for t in req.texts):
        raise HTTPException(status_code=400, detail="texts must contain non-empty strings")

    translator = get_translator()

    try:
        result = translator.translate_text(req.texts, target_lang=req.targetLang.upper())
    except Exception as e:
        msg = str(e) or type(e).__name__
        raise HTTPException(status_code=502, detail=f"DeepL error: {msg}")

    if isinstance(result, list):
        return {"translations": [r.text for r in result]}

    return {"translations": [result.text]}