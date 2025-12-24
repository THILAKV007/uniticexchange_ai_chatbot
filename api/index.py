from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import json

app = FastAPI(title="Question Answer API")

# Path to your JSONL file
JSONL_PATH = Path(__file__).parent.parent / "combined_cleaned.jsonl"


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def read_root():
    return {"message": "Question API is running"}


@app.post("/ask")
def ask_question(data: QuestionRequest):
    question = data.question.strip().lower()

    if not JSONL_PATH.exists():
        return {
            "question": data.question,
            "answer": "Knowledge base file not found."
        }

    # Stream the JSONL file line-by-line (LOW MEMORY)
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                messages = obj.get("messages", [])

                # Extract user question
                user_q = next(
                    (
                        m.get("content", "").strip().lower()
                        for m in messages
                        if m.get("role") == "user"
                    ),
                    None
                )

                # Skip non-matching questions early
                if user_q != question:
                    continue

                # Extract assistant answer
                assistant_a = next(
                    (
                        m.get("content", "").strip()
                        for m in messages
                        if m.get("role") == "assistant"
                    ),
                    None
                )

                if assistant_a:
                    return {
                        "question": data.question,
                        "answer": assistant_a
                    }

            except Exception:
                # Skip malformed lines safely
                continue

    return {
        "question": data.question,
        "answer": "No exact answer found for this question."
    }
