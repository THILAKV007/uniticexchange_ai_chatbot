from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json
from pathlib import Path

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


JSONL_PATH = Path("combined_cleaned.jsonl")

if not JSONL_PATH.exists():
    raise FileNotFoundError("combined_cleaned.jsonl not found")

qa_map = {}

with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for line_no, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
            messages = obj.get("messages", [])

            user_q = next(
                (m["content"].strip() for m in messages if m.get("role") == "user"),
                None
            )
            assistant_a = next(
                (m["content"].strip() for m in messages if m.get("role") == "assistant"),
                None
            )

            if user_q and assistant_a:
                qa_map[user_q.lower()] = assistant_a

        except Exception as e:
            print(f"Skipping line {line_no}: {e}")

print(f"Loaded {len(qa_map)} Q&A pairs")


class Question(BaseModel):
    question: str


@app.post("/ask")
def ask_question(payload: Question):
    question = payload.question.lower().strip()
    answer = qa_map.get(question, "No response received")
    return {"answer": answer}

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
