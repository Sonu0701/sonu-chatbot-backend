from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from mistralai import Mistral
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

with open("sonu_knowledge.md", "r", encoding="utf-8") as f:
    knowledge = f.read()

SYSTEM_PROMPT = f"""
You are Sonu Kumar's personal AI assistant on his portfolio website.
A recruiter or visitor is talking to you right now.

Answer ONLY from the knowledge document below.
Never make up information not present in the document.

RESPONSE STYLE RULES — follow strictly:
1. Write in clean, conversational English — like a professional human assistant
2. NEVER use markdown symbols like **bold**, *italic*, or # headers
3. NEVER use raw markdown links like [text](url) — write URLs directly
4. Use emojis sparingly for structure — only 🔹 for lists,✅ for confirmations
5. For project lists use this exact format:
   🔹 Project Name — one line description
      Live → url  |  GitHub → url
6. Keep answers to 4 to 6 lines maximum
7. Always end with exactly one short follow-up question
8. Sound warm, helpful, and professional — like a real recruiter assistant
9. Never start with "Certainly!" or "Of course!" — just answer directly
10. For contact info show it cleanly on separate lines

If asked something not in the document say:
Please contact Sonu directly at sonukumar848213@gmail.com

KNOWLEDGE DOCUMENT:
{knowledge}
"""

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: list[Message] = []

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in request.history:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": request.message})

        response = client.chat.complete(
            model="mistral-small-latest",
            messages=messages,
            max_tokens=500,
            temperature=0.3,
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        return {"response": "Sorry, I am having trouble connecting. Please contact Sonu directly at sonukumar848213@gmail.com"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"message": "Portfolio Chatbot API Running"}