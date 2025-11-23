from fastapi import FastAPI, Request, UploadFile, File, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import os

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "API is working!", "status": "ok"}

@app.get("/health")
async def health():
    return {"status": "ok", "gemini_configured": bool(os.getenv("GEMINI_API_KEY"))}

# Import the main app routes
try:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from backend.main import app as main_app
    
    # Mount all routes from main app
    app.mount("/", main_app)
except Exception as e:
    print(f"Error importing main app: {e}")

from mangum import Mangum
handler = Mangum(app, lifespan="off")
