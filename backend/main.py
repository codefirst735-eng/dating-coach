from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class ChatRequest(BaseModel):
    message: str

@app.get("/")
async def read_root():
    return {"message": "Welcome to FastAPI + Bootstrap API"}

@app.post("/chat")
async def chat(request: ChatRequest):
    # Simple "Reality" style responses for now
    responses = [
        "Stop seeking validation. Focus on your mission.",
        "They are testing your boundaries. Do not react emotionally.",
        "You are the prize. Act like it.",
        "Understand the dynamics at play. Don't ignore the signs.",
        "Invest in yourself. Your value is your leverage.",
        "Is this behavior serving your long-term goals? If not, cut it out.",
        "Maintain strong boundaries. Disrespect is not tolerated.",
    ]
    return {"response": random.choice(responses)}

@app.post("/analyze-screenshot")
async def analyze_screenshot(file: UploadFile = File(...)):
    # Mock analysis logic
    return {
        "assessment": "They are testing your compliance. This is a classic test designed to see if you will jump through their hoops.",
        "reply": "Haha, nice try. I'm busy tonight, but I might be free on Thursday.",
        "reasoning": "This reply maintains your frame, shows you have a life (high value), and sets the terms for the interaction on your schedule."
    }
