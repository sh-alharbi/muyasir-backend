import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

app = FastAPI()


class SimplifyRequest(BaseModel):
    text: str


class SimplifyResponse(BaseModel):
    simplified: str


@app.post("/simplify", response_model=SimplifyResponse)
async def simplify(req: SimplifyRequest):
    system_prompt = (
    "Rewrite the user's text in the same language but in a very clear and simple way. "
    "Your goal is deep simplification: keep the full meaning but remove all formal or complex wording. "
    "Use short sentences and very common, easy words. "
    "Do not summarize or remove important points, but make the message direct and easy to understand and short. "
    "If the text is long or has multiple ideas, break it into multiple short sentences. "
    "Avoid any official or governmental expressions. "
    "Avoid difficult vocabulary, rare words, or long phrases. "
    "Make the text feel natural, friendly, and clear for people who rely mainly on reading. "
    "Do not add new information. Output only the simplified version."
    "Make it short but have all the important things"
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
