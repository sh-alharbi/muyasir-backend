import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

app = FastAPI(
    title="Moyasir Backend",
    description="API for text simplification for accessibility.",
    version="1.0.0"
)




class SimplifyRequest(BaseModel):
    text: str


class SimplifyResponse(BaseModel):
    simplified: str




@app.get("/")
def root():
    return {"status": "ok", "message": "Moyasir backend is running"}





@app.post("/simplify", response_model=SimplifyResponse)
async def simplify(req: SimplifyRequest):
    system_prompt = (
        "Rewrite the user's text in the SAME language, but in a very clear and simple way. "
        "Your job is DEEP simplification: keep the full meaning but remove all formal or complex wording. "
        "Use very short sentences and very common, easy words. "
        "Do NOT summarize or delete important points. Preserve the meaning. "
        "If the text has multiple ideas, separate them into short sentences. "
        "Avoid government/formal expressions. "
        "Avoid difficult vocabulary or long phrases. "
        "The tone should be friendly, natural, and easy for people with weak reading skills. "
        "Do not add new ideas. Only rewrite what the user wrote. "
        "Output ONLY the simplified version. Keep it short but complete."
    )

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": req.text},
        ],
        temperature=0.4,
    )

    simplified_text = completion.choices[0].message.content.strip()

    return SimplifyResponse(simplified=simplified_text)
