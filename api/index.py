from fastapi import FastAPI
from pydantic import BaseModel
import json
from pathlib import Path

app = FastAPI()

JSONL_PATH = "combined_cleaned.jsonl"


qa_map = {}

if not Path(JSONL_PATH).exists():
    raise FileNotFoundError(f"{JSONL_PATH} not found")

with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for line_no, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
            messages = obj.get("messages", [])

            user_q = next(
                (m["content"].strip() for m in messages if m["role"] == "user"),
                None
            )
            assistant_a = next(
                (m["content"].strip() for m in messages if m["role"] == "assistant"),
                None
            )

            if user_q and assistant_a:
                # normalize key
                qa_map[user_q.lower()] = assistant_a

        except Exception as e:
            print(f"Skipping line {line_no}: {e}")

print(f"Loaded {len(qa_map)} Q&A pairs")


class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
def ask_question(data: QuestionRequest):
    question = data.question.strip().lower()

    answer = qa_map.get(question)

    if not answer:
        return {
            "question": data.question,
            "answer": "No exact answer found for this question."
        }

    return {
        "question": data.question,
        "answer": answer
    }


@app.get("/")
def read_root():
    return {"message": "Question API is running"}
