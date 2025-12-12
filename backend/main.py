from fastapi import FastAPI, Request, UploadFile, File, Form, Depends, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import random
import os
from dotenv import load_dotenv
import google.generativeai as genai
import io
import pathlib
import tempfile

# Load environment variables
backend_dir = pathlib.Path(__file__).parent.absolute()
project_root = backend_dir.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

# --- Configuration ---
SECRET_KEY = "your-secret-key-change-this-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./users.db")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Initialize Gemini
print(f"DEBUG: GEMINI_API_KEY loaded: {bool(GEMINI_API_KEY)}")
print(f"DEBUG: API Key length: {len(GEMINI_API_KEY) if GEMINI_API_KEY else 0}")
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        print("SUCCESS: Gemini API configured successfully")
    except Exception as e:
        print(f"ERROR: Failed to configure Gemini API: {e}")
else:
    print("WARNING: GEMINI_API_KEY not set. AI features will be disabled.")

app = FastAPI()

# --- Static Files ---
from fastapi.staticfiles import StaticFiles
import os

# Create uploads directory if it doesn't exist
os.makedirs("uploads/blogs", exist_ok=True)

# Mount static files
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# --- Database Setup ---
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- Security & Auth Setup ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# --- Models (SQLAlchemy) ---
class UserDB(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    full_name = Column(String)
    hashed_password = Column(String)
    disabled = Column(Boolean, default=False)
    queries_used = Column(Integer, default=0)
    screenshots_analyzed = Column(Integer, default=0)
    subscription_plan = Column(String, default="Sleeper")
    plan_expiry = Column(String, nullable=True)
    gender_preference = Column(String, default="male")  # 'male' or 'female'

class GeminiFileDB(Base):
    __tablename__ = "openai_files" # Keep table name to avoid migration issues
    id = Column(Integer, primary_key=True, index=True)
    file_id = Column(String, unique=True, index=True) # Stores Gemini file.name (e.g. files/xxxx)
    filename = Column(String)
    uploaded_at = Column(String)
    purpose = Column(String, default="assistants")
    gender_category = Column(String, default="male")  # 'male' or 'female' - which coach uses this PDF
    
class MessageDB(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    role = Column(String) # user or assistant
    content = Column(String)
    timestamp = Column(String) # ISO date string
    coach_type = Column(String, default="male") # 'male' or 'female'

class ScreenshotAnalysisDB(Base):
    __tablename__ = "screenshot_analyses"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True, nullable=True)  # NULL for guest users
    assessment = Column(Text)
    recommended_reply = Column(Text)
    reasoning = Column(Text)
    timestamp = Column(String)  # ISO date string
    image_data = Column(Text, nullable=True)  # Base64 encoded image (optional, for context)

class BlogDB(Base):
    __tablename__ = "blogs"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    slug = Column(String, unique=True, index=True)
    content = Column(Text)
    excerpt = Column(String)
    featured_image = Column(String, nullable=True)  # Path to uploaded image
    categories = Column(String, default="")  # Comma-separated categories
    meta_title = Column(String)
    meta_description = Column(String)
    keywords = Column(String)  # Comma-separated keywords
    author = Column(String, default="RFH Team")
    published = Column(Boolean, default=False)
    published_at = Column(String, nullable=True)
    created_at = Column(String)
    updated_at = Column(String)

class BlogCommentDB(Base):
    __tablename__ = "blog_comments"
    id = Column(Integer, primary_key=True, index=True)
    blog_id = Column(Integer, index=True)  # Foreign key to blogs table
    user_id = Column(Integer, index=True)  # Foreign key to users table
    username = Column(String)  # Denormalized for easy display
    content = Column(Text)
    parent_id = Column(Integer, nullable=True)  # NULL for top-level comments, ID for replies
    is_admin = Column(Boolean, default=False)  # True if comment is from RFH admin
    created_at = Column(String)
    updated_at = Column(String)

Base.metadata.create_all(bind=engine)

# --- Database Migration (Auto-fix for missing columns) ---
from sqlalchemy import inspect, text

def run_migrations():
    """
    Checks for missing columns in the database and adds them if necessary.
    This fixes the 'no such column' error on deployment without deleting data.
    """
    inspector = inspect(engine)
    
    with engine.connect() as conn:
        # 1. Fix 'openai_files' table (Legacy name kept for data preservation)
        if "openai_files" in inspector.get_table_names():
            columns = [col["name"] for col in inspector.get_columns("openai_files")]
            if "gender_category" not in columns:
                print("MIGRATION: Adding 'gender_category' column to 'openai_files' table...")
                try:
                    conn.execute(text('ALTER TABLE openai_files ADD COLUMN gender_category VARCHAR DEFAULT "male"'))
                    conn.commit()
                    print("MIGRATION: Success!")
                except Exception as e:
                    print(f"MIGRATION ERROR: {e}")

        # 2. Fix 'users' table
        if "users" in inspector.get_table_names():
            columns = [col["name"] for col in inspector.get_columns("users")]
            if "gender_preference" not in columns:
                print("MIGRATION: Adding 'gender_preference' column to 'users' table...")
                try:
                    conn.execute(text('ALTER TABLE users ADD COLUMN gender_preference VARCHAR DEFAULT "male"'))
                    conn.commit()
                    print("MIGRATION: Success!")
                except Exception as e:
                    print(f"MIGRATION ERROR: {e}")

        # 3. Fix 'messages' table
        if "messages" in inspector.get_table_names():
            columns = [col["name"] for col in inspector.get_columns("messages")]
            if "coach_type" not in columns:
                print("MIGRATION: Adding 'coach_type' column to 'messages' table...")
                try:
                    conn.execute(text('ALTER TABLE messages ADD COLUMN coach_type VARCHAR DEFAULT "male"'))
                    conn.commit()
                    print("MIGRATION: Success!")
                except Exception as e:
                    print(f"MIGRATION ERROR: {e}")

        # 4. Create 'blogs' table if it doesn't exist
        if "blogs" not in inspector.get_table_names():
            print("MIGRATION: Creating 'blogs' table...")
            try:
                conn.execute(text('''
                    CREATE TABLE blogs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        title VARCHAR NOT NULL,
                        slug VARCHAR UNIQUE NOT NULL,
                        content TEXT NOT NULL,
                        excerpt VARCHAR NOT NULL,
                        featured_image VARCHAR,
                        categories VARCHAR DEFAULT "",
                        meta_title VARCHAR NOT NULL,
                        meta_description VARCHAR NOT NULL,
                        keywords VARCHAR NOT NULL,
                        author VARCHAR DEFAULT "RFH Team",
                        published BOOLEAN DEFAULT 0,
                        published_at VARCHAR,
                        created_at VARCHAR NOT NULL,
                        updated_at VARCHAR NOT NULL
                    )
                '''))
                conn.commit()
                print("MIGRATION: Blogs table created successfully!")
            except Exception as e:
                print(f"MIGRATION ERROR creating blogs table: {e}")

        # 5. Add categories column to blogs table if it doesn't exist
        if "blogs" in inspector.get_table_names():
            columns = [col["name"] for col in inspector.get_columns("blogs")]
            if "categories" not in columns:
                print("MIGRATION: Adding 'categories' column to 'blogs' table...")
                try:
                    conn.execute(text('ALTER TABLE blogs ADD COLUMN categories VARCHAR DEFAULT ""'))
                    conn.commit()
                    print("MIGRATION: Categories column added successfully!")
                except Exception as e:
                    print(f"MIGRATION ERROR adding categories column: {e}")
# Run migrations on startup
run_migrations()

# Initialize database with default data if empty
def initialize_database():
    db = SessionLocal()
    try:
        # Check if we have any users
        user_count = db.query(UserDB).count()
        if user_count == 0:
            print("INITIALIZATION: Creating default test user...")
            # Create a test user
            hashed_password = get_password_hash("test123")
            test_user = UserDB(
                username="testuser",
                email="test@example.com",
                hashed_password=hashed_password,
                full_name="Test User",
                disabled=False
            )
            db.add(test_user)
            db.commit()
            print("INITIALIZATION: Test user created (username: testuser, password: test123)")

        # Check if we have any blogs
        blog_count = db.query(BlogDB).count()
        if blog_count == 0:
            print("INITIALIZATION: Creating sample blog...")
            sample_blog = BlogDB(
                title="Welcome to RFH Dating Coach",
                slug="welcome-to-rfh-dating-coach",
                content="<p>Welcome to Relationship for Humans! This AI-powered dating coach is designed to help you navigate the complex world of modern relationships.</p><p>Our platform uses advanced AI technology to provide personalized advice, analyze your screenshots, and offer insights into relationship dynamics.</p><p>Get started by exploring our features and taking the first step towards better relationships!</p>",
                excerpt="Welcome to Relationship for Humans! Discover how our AI-powered dating coach can help you navigate modern relationships.",
                featured_image="",
                categories="Getting Started,Welcome",
                meta_title="Welcome to RFH Dating Coach",
                meta_description="AI-powered dating coach to help you navigate modern relationships with personalized advice and insights.",
                keywords="dating coach, relationships, AI, advice",
                author="RFH Team",
                published=True,
                published_at=datetime.utcnow().isoformat(),
                created_at=datetime.utcnow().isoformat(),
                updated_at=datetime.utcnow().isoformat()
            )
            db.add(sample_blog)
            db.commit()
            print("INITIALIZATION: Sample blog created")

    except Exception as e:
        print(f"INITIALIZATION ERROR: {e}")
        db.rollback()
    finally:
        db.close()

initialize_database()

# --- Models (Pydantic) ---
class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None
    queries_used: int = 0
    screenshots_analyzed: int = 0
    subscription_plan: str = "Sleeper"
    plan_expiry: Optional[str] = None

class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    full_name: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []
    gender_preference: Optional[str] = "male"  # 'male' (default) or 'female'
    persona: Optional[str] = "standard"
    language: Optional[str] = "English"

class FollowUpRequest(BaseModel):
    question: str
    history: Optional[List[ChatMessage]] = []  # Conversation history for context


class PlanUpdate(BaseModel):
    plan: str

class ProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None
    gender_preference: Optional[str] = None  # 'male' or 'female'

class CommentCreate(BaseModel):
    content: str
    parent_id: Optional[int] = None  # None for top-level comments, ID for replies

class CommentResponse(BaseModel):
    id: int
    blog_id: int
    user_id: int
    username: str
    content: str
    parent_id: Optional[int] = None
    is_admin: bool
    created_at: str
    updated_at: str
    replies: List['CommentResponse'] = []  # Nested replies

    class Config:
        from_attributes = True

class FileUploadResponse(BaseModel):
    file_id: str
    filename: str
    message: str

class FileListResponse(BaseModel):
    files: List[dict]

class BlogBase(BaseModel):
    title: str
    slug: str
    content: str
    excerpt: str
    categories: str = ""
    meta_title: str
    meta_description: str
    keywords: str
    author: Optional[str] = "RFH Team"
    published: bool = False

class BlogCreate(BlogBase):
    pass

class BlogUpdate(BaseModel):
    title: Optional[str] = None
    slug: Optional[str] = None
    content: Optional[str] = None
    excerpt: Optional[str] = None
    featured_image: Optional[str] = None
    categories: Optional[str] = None
    meta_title: Optional[str] = None
    meta_description: Optional[str] = None
    keywords: Optional[str] = None
    author: Optional[str] = None
    published: Optional[bool] = None

class Blog(BaseModel):
    id: int
    title: str
    slug: str
    content: str
    excerpt: str
    featured_image: Optional[str] = None
    categories: str = ""
    meta_title: str
    meta_description: str
    keywords: str
    author: str
    published: bool
    published_at: Optional[str] = None
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True

class BlogList(BaseModel):
    blogs: List[Blog]

# --- Helper Functions ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_query_limit(plan: str) -> int:
    limits = {
        "Sleeper": 10,
        "Basic": 10,
        "Initiate": 100,
        "Pro": 100,
        "Master": 999999,
        "Premium": 999999
    }
    return limits.get(plan, 10)

def get_user(db: Session, username: str):
    return db.query(UserDB).filter(UserDB.username == username).first()

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: UserDB = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://localhost:8081",
        "http://localhost:8082",
        "http://127.0.0.1:8080",
        "http://127.0.0.1:8081",
        "http://127.0.0.1:8082",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_origin_regex=r"https://.*\.vercel\.app|https://.*\.onrender\.com",
)

# --- Routes ---

@app.get("/")
async def read_root():
    return {"message": "Welcome to FastAPI + Gemini API"}

@app.get("/debug")
async def debug_info():
    """Debug endpoint to check API connectivity"""
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "api_base_url": os.getenv("API_BASE_URL", "not set"),
        "database_url": os.getenv("DATABASE_URL", "sqlite default"),
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY"))
    }

@app.get("/admin/health")
async def admin_health():
    return {
        "status": "ok",
        "gemini_configured": bool(GEMINI_API_KEY)
    }

@app.post("/register", response_model=Token)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = get_user(db, username=user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = get_password_hash(user.password)
    new_user = UserDB(
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        hashed_password=hashed_password
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = get_user(db, form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=User)
async def read_users_me(current_user: UserDB = Depends(get_current_active_user)):
    return current_user

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    current_user: UserDB = Depends(get_current_active_user)
):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="AI service not configured")
        
    try:
        # Read file content
        content = await file.read()
        print(f"DEBUG: Received audio file: {file.filename}, size: {len(content)} bytes, type: {file.content_type}")
        
        # Validate file size (max 20MB for safety)
        if len(content) > 20 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Audio file too large (max 20MB)")
        
        if len(content) < 100:
            raise HTTPException(status_code=400, detail="Audio file too small or empty")
        
        # Determine mime type
        mime_type = file.content_type or "audio/webm"
        if "mp4" in mime_type or "m4a" in mime_type:
            mime_type = "audio/mp4"
        elif "mpeg" in mime_type or "mp3" in mime_type:
            mime_type = "audio/mpeg"
        elif "wav" in mime_type:
            mime_type = "audio/wav"
        elif "ogg" in mime_type:
            mime_type = "audio/ogg"
        else:
            mime_type = "audio/webm"  # Default
            
        print(f"DEBUG: Using mime type: {mime_type}")
        
        print(f"DEBUG: Using mime type: {mime_type}")
        
        # Use Inline Data for robustness (avoid File API connection resets)
        # We use the official google.ai.generativelanguage types
        import asyncio
        from google.ai import generativelanguage as glm
        
        loop = asyncio.get_running_loop()
        
        try:
            # Use gemini-flash-latest as confirmed by ListModels
            model = genai.GenerativeModel('gemini-flash-latest')
            
            print(f"DEBUG: Preparing inline audio data ({len(content)} bytes)...")
            
            # Create proper inline data part using Gemini SDK format
            # This avoids uploading to a separate File API endpoint
            audio_blob = glm.Blob(
                mime_type=mime_type,
                data=content
            )
            
            audio_part = glm.Part(
                inline_data=audio_blob
            )
            
            print(f"DEBUG: Generating transcription...")
            
            # Generate transcription
            def generate_transcription():
                return model.generate_content([
                    "Please transcribe this audio recording word-for-word. Return only the transcribed text, nothing else.",
                    audio_part
                ])
            
            response = await loop.run_in_executor(None, generate_transcription)
            
            print(f"DEBUG: Transcription response received")
            
            if not response or not response.text:
                print(f"ERROR: Empty response from AI")
                raise Exception("AI returned empty transcription")
            
            transcribed_text = response.text.strip()
            print(f"DEBUG: Transcription successful: {transcribed_text[:100]}...")
            
            return {"text": transcribed_text}
            
        except Exception as ai_error:
            print(f"ERROR in Gemini transcription: {type(ai_error).__name__}: {str(ai_error)}")
            # Try to provide more specific error message
            error_detail = str(ai_error)
            if "quota" in error_detail.lower():
                raise HTTPException(status_code=429, detail="API quota exceeded. Please try again later.")
            elif "invalid" in error_detail.lower() and "audio" in error_detail.lower():
                raise HTTPException(status_code=400, detail="Audio format not supported. Please try recording again.")
            else:
                raise HTTPException(status_code=500, detail=f"Transcription failed: {error_detail}")
        finally:
             if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
                 os.remove(temp_audio_path)
                 print(f"DEBUG: Cleaned up local temp file")
                
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"TRANSCRIPTION ERROR: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Failed to process audio: {error_msg}")

@app.post("/chat")
async def chat(request: ChatRequest, current_user: UserDB = Depends(get_current_active_user), db: Session = Depends(get_db)):
    # Check query limit
    query_limit = get_query_limit(current_user.subscription_plan)
    if current_user.queries_used >= query_limit:
        raise HTTPException(
            status_code=403,
            detail=f"Query limit reached for {current_user.subscription_plan} plan. Please upgrade your subscription."
        )
    
    # Increment usage
    current_user.queries_used += 1
    db.commit()
    
    # Gender-specific system prompts
    male_coach_prompt = """You are a direct, confident relationship coach specializing in female psychology and dating dynamics.

⚠️ OVERRIDE INSTRUCTION - READ THIS FIRST:
Questions about human biology and physiology (menstruation, periods, ovulation, hormones, fertility, arousal, erections, sexual health, etc.) are NORMAL and ALLOWED when discussed in the context of dating, attraction, and relationships. These are natural biological processes that affect dating dynamics. Answer these questions professionally and helpfully.

CRITICAL SAFETY & SCOPE RULES:
- REFUSE to answer questions about self-harm, suicide, violence, or murder.
- You CAN and SHOULD answer questions about human biology, physiology, and natural processes (periods, menstruation, ovulation, erections, arousal, hormones, etc.) when they relate to dating, attraction, or relationships.
- You CAN answer questions about influence, persuasion, manipulation psychology, and dating tactics - these are legitimate parts of attraction dynamics.
- Only refuse questions COMPLETELY unrelated to human interaction, relationships, or social dynamics (e.g., "how to fix a car", "what's the weather").
- If a question is truly out of scope or harmful, respond with: "I'm specifically designed to help with dating and relationship challenges. I can't assist with that topic, but I'm here if you have questions about attraction, communication, or relationship dynamics."

IMPORTANT INSTRUCTIONS:
- Keep responses concise and readable (150-200 words maximum).
- Use natural paragraph breaks for readability.
- You MAY use bullet points (with hyphens) when listing action steps or key points.
- Do NOT use bold text or headers.
- Write in plain text, like you are texting a knowledgeable friend.
- Be direct. Break down the situation and explain the dynamics at play.
- Take your knowledge from the PDFs I have uploaded for you.
- NEVER use exact terminology or concept names from books/PDFs. Always paraphrase and adapt concepts into your own words.
- If a concept comes from a source, rephrase it naturally without attribution (e.g., instead of "the concept of XYZ," say "I've found that...").
- Give actionable advice with clear reasoning behind it.
- Explain WHY things work the way they do, and WHAT to do.
- Focus on: attraction psychology, evolutionary psychology, and behavioural psychology.
- ALWAYS end with just one follow-up question to continue the conversation.
"""


    female_coach_prompt = """You are a direct, confident relationship coach specializing in male psychology and dating dynamics.

⚠️ OVERRIDE INSTRUCTION - READ THIS FIRST:
Questions about human biology and physiology (menstruation, periods, ovulation, hormones, fertility, arousal, erections, sexual health, etc.) are NORMAL and ALLOWED when discussed in the context of dating, attraction, and relationships. These are natural biological processes that affect dating dynamics. Answer these questions professionally and helpfully.

CRITICAL SAFETY & SCOPE RULES:
- REFUSE to answer questions about self-harm, suicide, violence, or murder.
- You CAN and SHOULD answer questions about human biology, physiology, and natural processes (periods, menstruation, ovulation, erections, arousal, hormones, etc.) when they relate to dating, attraction, or relationships.
- You CAN answer questions about influence, persuasion, manipulation psychology, and dating tactics - these are legitimate parts of attraction dynamics.
- Only refuse questions COMPLETELY unrelated to human interaction, relationships, or social dynamics (e.g., "how to fix a car", "what's the weather").
- If a question is truly out of scope or harmful, respond with: "I'm specifically designed to help with dating and relationship challenges. I can't assist with that topic, but I'm here if you have questions about attraction, communication, or relationship dynamics."

IMPORTANT INSTRUCTIONS:
- Keep responses concise and readable (150-200 words maximum).
- Use natural paragraph breaks for readability.
- You MAY use bullet points when listing action steps or key points.
- Do NOT use bold text or headers.
- Write in plain text, like you are texting a supportive friend.
- Be direct. Break down the situation and explain the dynamics at play.
- NEVER mention any book names, PDF names, authors, or specific sources. Present all knowledge as your own expertise.
- NEVER use exact terminology or concept names from books/PDFs. Always paraphrase and adapt concepts into your own words.
- If a concept comes from a source, rephrase it naturally without attribution (e.g., instead of "the concept of XYZ," say "I've found that...").
- Give actionable advice with clear reasoning behind it.
- Explain WHY things work the way they do, and WHAT to do.
- Focus on: attraction psychology, behavioural psychology, and evolutionary psychology.
- ALWAYS end with just one follow-up question to continue the conversation.

Your goal is to sound like a real person who deeply understands male dating dynamics, not a robot giving one-liners."""


    # Select prompt based on user's gender preference
    base_prompt = male_coach_prompt if current_user.gender_preference == "male" else female_coach_prompt
    
    # Persona overrides
    persona_prompts = {
        "drill_sergeant": "\n\nPERSONA OVERRIDE: You are a TOUGH LOVE 'Drill Sergeant' type coach. Be brutally honest. Stop the user from making excuses. Use short, punchy sentences. Don't coddle them. Focus on discipline and action. Your tone is commanding.",
        "wingman": "\n\nPERSONA OVERRIDE: You are an enthusiastic 'Wingman'. Use casual, bro-like language (like 'dude', 'bro', 'man'). Be hype and supportive. Focus on boosting their confidence and giving practical, 'street-smart' advice. Your tone is high-energy.",
        "therapist": "\n\nPERSONA OVERRIDE: You are an Empathetic Relationship Therapist. Focus on the emotional undercurrents, anxiety, and connection. Be gentle, validating, and ask deep questions about how they feel. Use softer language. Your tone is calming."
    }
    
    # Apply persona if selected (and not standard)
    persona_instruction = persona_prompts.get(request.persona, "")

    # Add user's name to the prompt to prevent hallucinated names
    user_name = current_user.full_name if current_user.full_name else "User"
    user_name_instruction = f"\n\nIMPORTANT: The user's name is '{user_name}'. Address them by this name occasionally. Do NOT make up a name for the user."
    
    # Language instruction - MUST be very strong to override conversation history
    language_instruction = ""
    if request.language and request.language != "English":
        language_instruction = f"\n\n🚨 CRITICAL LANGUAGE OVERRIDE 🚨\nYou MUST respond ONLY in {request.language} language, regardless of what language was used in previous messages. The user has explicitly selected {request.language} as their preferred language. ALL your responses from this point forward MUST be in {request.language}. Do NOT use English unless the selected language is English. If 'Hinglish' is selected, use a natural mix of Hindi and English as spoken in informal Indian conversation (use Roman/Latin script, not Devanagari)."

    system_prompt = base_prompt + user_name_instruction + persona_instruction + language_instruction

    if not GEMINI_API_KEY:
        return {"response": "Gemini API key not configured. Please check .env file."}

    try:
        # Get stored files filtered by user's gender preference
        stored_files = db.query(GeminiFileDB).filter(
            GeminiFileDB.gender_category == current_user.gender_preference
        ).all()
        
        # Retrieve conversation history (last 20 messages for context)
        history_messages = db.query(MessageDB).filter(
            MessageDB.user_id == current_user.id,
            MessageDB.coach_type == current_user.gender_preference
        ).order_by(MessageDB.timestamp.desc()).limit(20).all()
        
        # Reverse to get chronological order
        history_messages = list(reversed(history_messages))
        
        # Initialize model with system instruction
        model = genai.GenerativeModel(
            model_name="gemini-flash-latest",
            system_instruction=system_prompt
        )
        
        # Start a chat session with history
        chat_history = []
        for msg in history_messages:
            chat_history.append({
                "role": "user" if msg.role == "user" else "model",
                "parts": [msg.content]
            })
        
        chat = model.start_chat(history=chat_history)
        
        # Prepare content parts for the current message
        content_parts = []
        
        # Always include PDFs for context on every query
        for file_record in stored_files:
            try:
                file_obj = genai.get_file(file_record.file_id)
                content_parts.append(file_obj)
            except Exception as e:
                print(f"Error fetching file {file_record.file_id}: {e}")
                continue
        
        # Add user message
        content_parts.append(request.message)
        
        # Generate response with context (run in executor to avoid blocking)
        import asyncio
        from functools import partial
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            partial(chat.send_message, content_parts)
        )
        ai_response_text = response.text
        
        # Save user message
        user_msg = MessageDB(
            user_id=current_user.id,
            role="user",
            content=request.message,
            timestamp=datetime.utcnow().isoformat(),
            coach_type=current_user.gender_preference
        )
        db.add(user_msg)
        
        # Save AI response
        ai_msg = MessageDB(
            user_id=current_user.id,
            role="assistant",
            content=ai_response_text,
            timestamp=datetime.utcnow().isoformat(),
            coach_type=current_user.gender_preference
        )
        db.add(ai_msg)
        db.commit()
        
        return {"response": ai_response_text}

        
    except Exception as e:
        import traceback
        print(f"Gemini API error: {str(e)}")
        print(f"Full traceback: {traceback.format_exc()}")
        # Fallback responses
        responses = [
            "Stop seeking validation. Focus on your mission.",
            "They are testing your boundaries. Do not react emotionally.",
            "You are the prize. Act like it.",
            "Understand the dynamics at play. Don't ignore the signs.",
            "Invest in yourself. Your value is your leverage.",
        ]
        fallback_response = random.choice(responses)
        
        # Save fallback to history
        user_msg = MessageDB(
            user_id=current_user.id,
            role="user",
            content=request.message,
            timestamp=datetime.utcnow().isoformat(),
            coach_type=current_user.gender_preference
        )
        ai_msg = MessageDB(
            user_id=current_user.id,
            role="assistant",
            content=fallback_response,
            timestamp=datetime.utcnow().isoformat(),
            coach_type=current_user.gender_preference
        )
        db.add(user_msg)
        db.add(ai_msg)
        db.commit()
        
        return {"response": fallback_response}

@app.get("/chat/history")
async def get_chat_history(
    coach_type: Optional[str] = None,
    current_user: UserDB = Depends(get_current_active_user), 
    db: Session = Depends(get_db)
):
    query = db.query(MessageDB).filter(MessageDB.user_id == current_user.id)
    
    # If coach_type is provided, filter by it. 
    # If not, default to current preference OR return all (depending on desired behavior).
    # Given the requirement "histories should not overlap", we should filter.
    if coach_type:
        query = query.filter(MessageDB.coach_type == coach_type)
    else:
        # Fallback to current preference if not specified
        query = query.filter(MessageDB.coach_type == current_user.gender_preference)
        
    messages = query.order_by(MessageDB.timestamp).all()
    return {
        "messages": [
            {"role": m.role, "content": m.content, "timestamp": m.timestamp}
            for m in messages
        ]
    }

@app.delete("/chat/history")
async def clear_chat_history(
    coach_type: Optional[str] = None,
    current_user: UserDB = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    query = db.query(MessageDB).filter(MessageDB.user_id == current_user.id)
    
    if coach_type:
        query = query.filter(MessageDB.coach_type == coach_type)
    else:
        # Default to current preference if not specified
        query = query.filter(MessageDB.coach_type == current_user.gender_preference)
        
    # Delete the messages
    query.delete(synchronize_session=False)
    db.commit()
    
    return {"message": "Chat history cleared successfully"}

@app.post("/analyze-screenshot")
async def analyze_screenshot(
    files: List[UploadFile] = File(...), 
    user_color: str = Query("blue"),
    other_color: str = Query("gray"),
    user_gender: str = Query("male"),
    other_gender: str = Query("female"),
    goal: str = Query("build_attraction"),
    language: str = Query("english"),
    tone: str = Query("balanced"),
    current_user: UserDB = Depends(get_current_active_user), 
    db: Session = Depends(get_db)
):
    current_user.screenshots_analyzed += len(files)
    db.commit()
    
    if not GEMINI_API_KEY:
        return {"error": "Gemini API key not configured"}
    
    try:
        # Read all uploaded images
        image_parts = []
        for file in files:
            image_data = await file.read()
            mime_type = file.content_type or "image/jpeg"
            image_parts.append({
                "mime_type": mime_type,
                "data": image_data
            })
        
        # Retrieve previous analyses for context (last 3)
        previous_analyses = db.query(ScreenshotAnalysisDB).filter(
            ScreenshotAnalysisDB.user_id == current_user.id
        ).order_by(ScreenshotAnalysisDB.timestamp.desc()).limit(3).all()
        
        # Build context from previous analyses
        context_text = ""
        if previous_analyses:
            context_text = "\n\nPREVIOUS ANALYSES CONTEXT (for continuity):\n"
            for i, analysis in enumerate(reversed(previous_analyses), 1):
                context_text += f"\nAnalysis {i}:\n"
                context_text += f"- Assessment: {analysis.assessment[:150]}...\n"
                context_text += f"- Recommended Reply: {analysis.recommended_reply}\n"
        
        # User color and gender clarification
        color_text = f"\n\n!!!CRITICAL INSTRUCTION - READ CAREFULLY!!!\n"
        color_text += f"The user asking for help has {user_color.upper()} colored message bubbles.\n"
        color_text += f"The other person has {other_color.upper()} colored message bubbles.\n"
        color_text += f"\n"
        color_text += f"YOU MUST:\n"
        color_text += f"1. Identify ALL messages with {user_color.upper()} color - these are the USER's messages\n"
        color_text += f"2. Identify ALL messages with {other_color.upper()} color - these are the OTHER PERSON's messages\n"
        color_text += f"3. Analyze ONLY the {user_color.upper()} messages (the user's messages)\n"
        color_text += f"4. DO NOT analyze the {other_color.upper()} messages\n"
        color_text += f"5. DO NOT assume based on left/right position - ONLY use the COLOR to identify messages\n"
        color_text += f"\n"
        color_text += f"Additional context:\n"
        color_text += f"- The user is {user_gender.upper()} talking to a {other_gender.upper()}\n"
        color_text += f"- Tailor advice for {user_gender}-{other_gender} dating dynamics\n"
        color_text += f"- Focus your analysis on what the {user_color.upper()} bubble person is saying/doing"
        
        # Map goal codes to descriptions
        goal_descriptions = {
            "build_attraction": "build attraction and romantic interest",
            "get_date": "secure a date or in-person meetup",
            "maintain_interest": "keep the conversation engaging and maintain their interest",
            "re_engage": "re-engage after a period of silence or lack of response",
            "escalate": "escalate intimacy and move the relationship forward",
            "end_politely": "end the conversation or relationship politely",
            "general": "get general dating advice"
        }
        goal_description = goal_descriptions.get(goal, "navigate this conversation effectively")
        
        # Map language codes to language names
        language_names = {
            "english": "English",
            "hinglish": "Hinglish (a natural mix of Hindi and English)",
            "hindi": "Hindi (Devanagari script)",
            "spanish": "Spanish",
            "french": "French",
            "german": "German",
            "italian": "Italian",
            "portuguese": "Portuguese",
            "russian": "Russian",
            "japanese": "Japanese",
            "korean": "Korean",
            "chinese": "Chinese (Simplified)"
        }
        language_name = language_names.get(language, "English")
        
        # Map tone codes to descriptions
        tone_descriptions = {
            "balanced": "a balanced mix of different styles",
            "playful": "playful and teasing",
            "direct": "direct and confident",
            "mysterious": "mysterious and intriguing",
            "casual": "casual and friendly",
            "witty": "witty and clever",
            "romantic": "romantic and sweet",
            "cocky_funny": "cocky-funny (confident with humor)",
            "challenge": "challenge/push-pull dynamics"
        }
        tone_description = tone_descriptions.get(tone, "a balanced mix of different styles")
        
        # Create system prompt for screenshot analysis
        analysis_prompt = f"""You are an expert dating coach analyzing text message screenshots.

ANALYSIS TASK:
1. Read the conversation in the screenshot
2. Identify who sent the LAST message in the conversation
3. Assess the power dynamics and "frame" (who is chasing whom)
4. Identify signs of attraction, interest level, or red flags
5. Determine the appropriate next action based on conversation state
6. Provide context-aware reply options

USER'S GOAL: {goal}
Tailor your advice and reply suggestions specifically to help achieve this exact goal.

PREFERRED TONE/STYLE: The user wants replies that are {tone_description}.
Focus on this tone while still providing variety in the 3 reply options.

LANGUAGE REQUIREMENT: Generate ALL recommended reply text messages in {language_name}.
- The assessment and reasoning should be in English
- ONLY the actual text message replies (the "text" field) should be in {language_name}
- Make the replies sound natural and authentic in {language_name}

CRITICAL RULES:
- Keep your analysis concise (150-200 words total)
- Be direct and strategic
- Focus on attraction psychology and power dynamics
- Do NOT mention books, PDFs, or specific methodologies
- ANALYZE THE CONVERSATION STATE CAREFULLY:
  * If the USER sent the last message and it's a QUESTION → Recommend WAITING for their response
  * If the USER is double/triple texting → Flag this as desperate behavior
  * If the OTHER PERSON hasn't responded in a while → Provide "follow-up" or "move on" advice
  * If the OTHER PERSON responded → Provide immediate reply suggestions
{color_text}
{context_text}

OUTPUT FORMAT:
Return a JSON object with the following keys:
- "assessment": Your analysis of the situation and power dynamics. IMPORTANT:
    * Explain WHY waiting is strategically important if applicable
    * Then provide your full analysis of power dynamics, attraction signals, and what's happening
    * Consider the user's specific goal: "{goal}"
- "replies": A list of 3 distinct reply options. Each should be an actual text message they could send in {language_name}:
    * Focus on the {tone_description} style the user prefers
    * Provide subtle variations within that preferred style
    * Each option must have:
      - "type": The specific sub-style of the reply (e.g., "Playful/Teasing", "Direct/Confident", etc.)
      - "text": The ACTUAL message text to copy and send (IN {language_name})
- "reasoning": Brief explanation of the overall strategy behind these reply styles (helps with user's goal: {goal})

Do not include markdown formatting (like ```json) in the response, just the raw JSON string.
"""


        # Initialize Gemini model with vision capability
        model = genai.GenerativeModel('gemini-flash-latest')
        
        import base64
        
        # Build prompt parts: start with the analysis prompt
        prompt_parts = [analysis_prompt]
        
        # Add note about multiple screenshots if applicable
        if len(image_parts) > 1:
            prompt_parts.append(f"\n\nNOTE: The user has uploaded {len(image_parts)} screenshots. Please analyze all of them together to provide comprehensive advice. Look for patterns, progression, and context across all the images.")
        
        # Add all image parts
        prompt_parts.extend(image_parts)
        
        # Run synchronous generation in a thread pool to avoid blocking the event loop
        import asyncio
        from functools import partial
        
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None, 
                partial(model.generate_content, prompt_parts)
            )
        except Exception as e:
            print(f"Gemini generation error type: {type(e).__name__}")
            print(f"Gemini generation error: {str(e)}")
            import traceback
            print(f"Full traceback:\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"AI generation failed: {str(e)}")

        # Check for safety blocks or empty response
        if not response.parts:
            print(f"Safety ratings: {response.prompt_feedback}")
            raise HTTPException(status_code=400, detail="AI refused to analyze this image due to safety filters. Please try a different image.")
        
        # Parse the response to extract assessment, reply, and reasoning
        full_response = response.text
        
        # Clean up potential markdown code blocks
        import json
        from datetime import datetime
        cleaned_response = full_response.replace("```json", "").replace("```", "").strip()
        
        result_data = {}
        try:
            parsed_response = json.loads(cleaned_response)
            
            # Handle both old (single reply) and new (multiple replies) formats for robustness
            replies = parsed_response.get("replies", [])
            
            # Ensure replies is a list
            if isinstance(replies, str):
                replies = [{"type": "Recommended", "text": replies}]
            elif not isinstance(replies, list):
                replies = []
                
            if not replies and "reply" in parsed_response:
                replies = [{"type": "Recommended", "text": parsed_response["reply"]}]
            
            # Final fallback if no replies found
            if not replies:
                replies = [{"type": "Analysis", "text": "Please check the assessment above for advice."}]
                
            result_data = {
                "assessment": parsed_response.get("assessment", full_response),
                "replies": replies,
                "reasoning": parsed_response.get("reasoning", "Check assessment")
            }
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            result_data = {
                "assessment": full_response,
                "replies": [{"type": "Error", "text": "Could not parse replies. See assessment."}],
                "reasoning": "See assessment above"
            }
        
        # Save the analysis to database
        # We store the list of replies as a JSON string in the recommended_reply column
        new_analysis = ScreenshotAnalysisDB(
            user_id=current_user.id,
            assessment=result_data["assessment"],
            recommended_reply=json.dumps(result_data["replies"]),
            reasoning=result_data["reasoning"],
            timestamp=datetime.utcnow().isoformat(),
            image_data=base64.b64encode(image_data).decode() if len(image_data) < 500000 else None  # Save image if < 500KB
        )
        db.add(new_analysis)
        db.commit()
        db.refresh(new_analysis)  # Refresh to get the ID
        
        result_data["id"] = new_analysis.id
        return result_data
    
    except Exception as e:
        print(f"Screenshot analysis error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "assessment": f"Unable to analyze the screenshot. Error: {str(e)}",
            "reply": "N/A",
            "reasoning": "Technical error occurred during analysis."
        }

@app.get("/screenshot-history")
async def get_screenshot_history(current_user: UserDB = Depends(get_current_active_user), db: Session = Depends(get_db)):
    """Retrieve screenshot analysis history for the authenticated user"""
    analyses = db.query(ScreenshotAnalysisDB).filter(
        ScreenshotAnalysisDB.user_id == current_user.id
    ).order_by(ScreenshotAnalysisDB.timestamp.desc()).all()
    
    import json
    result = []
    for analysis in analyses:
        # Parse the recommended_reply from JSON string to array
        replies = []
        try:
            if analysis.recommended_reply:
                # Try to parse as JSON array
                parsed = json.loads(analysis.recommended_reply)
                if isinstance(parsed, list):
                    replies = parsed
                else:
                    # If it's a string, wrap it
                    replies = [{"type": "Recommended", "text": str(parsed)}]
            else:
                replies = [{"type": "Analysis", "text": "See assessment above"}]
        except (json.JSONDecodeError, TypeError):
            # If parsing fails, treat as plain string
            replies = [{"type": "Recommended", "text": str(analysis.recommended_reply)}]
        
        result.append({
            "id": analysis.id,
            "assessment": analysis.assessment,
            "replies": replies,  # Changed from "reply" to "replies" and parsed the JSON
            "reasoning": analysis.reasoning,
            "timestamp": analysis.timestamp,
            "image_data": analysis.image_data
        })
    
    return result


@app.delete("/screenshot-history")
async def delete_screenshot_history(current_user: UserDB = Depends(get_current_active_user), db: Session = Depends(get_db)):
    """Delete all screenshot analysis history for the authenticated user"""
    try:
        deleted_count = db.query(ScreenshotAnalysisDB).filter(
            ScreenshotAnalysisDB.user_id == current_user.id
        ).delete()
        db.commit()
        
        return {
            "success": True,
            "message": f"Successfully deleted {deleted_count} analysis records",
            "deleted_count": deleted_count
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete history: {str(e)}")


@app.post("/screenshot-analysis/{analysis_id}/followup")
async def screenshot_followup(
    analysis_id: int,
    request: FollowUpRequest,
    current_user: UserDB = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Handle follow-up questions about a specific screenshot analysis"""
    
    # Verify the analysis belongs to the current user
    analysis = db.query(ScreenshotAnalysisDB).filter(
        ScreenshotAnalysisDB.id == analysis_id,
        ScreenshotAnalysisDB.user_id == current_user.id
    ).first()
    
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    user_question = request.question.strip()
    if not user_question:
        raise HTTPException(status_code=400, detail="Question is required")
    
    # Build context from the original analysis
    context = f"""
ORIGINAL SCREENSHOT ANALYSIS:
Assessment: {analysis.assessment}
Recommended Replies: {analysis.recommended_reply}
Reasoning: {analysis.reasoning}

USER FOLLOW-UP QUESTION:
{user_question}

INSTRUCTIONS:
You are a dating coach. The user has asked a follow-up question about the above analysis.
Answer their question directly and concisely. If they're asking for clarification, provide it.
If they're asking about a specific aspect (like "what if she says X?"), give them targeted advice.
Keep your response focused and actionable.
"""
    
    try:
        # Initialize model - ensure API key is configured globally or passed here if needed
        # Assuming genai.configure(api_key=...) is called at startup
        model = genai.GenerativeModel('gemini-flash-latest')
        
        # Run synchronous generation in a thread pool to avoid blocking the event loop
        import asyncio
        from functools import partial
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, 
            partial(model.generate_content, context)
        )
        
        return {
            "success": True,
            "answer": response.text,
            "question": user_question
        }
    except Exception as e:
        print(f"FOLLOW-UP ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")



@app.post("/guest-chat")
async def guest_chat(request: ChatRequest, db: Session = Depends(get_db)):
    """Chat endpoint for non-authenticated users with gender preference support."""
    
    # Validate gender preference
    gender_preference = request.gender_preference if request.gender_preference in ["male", "female"] else "male"
    
    # Gender-specific system prompts (same as authenticated users)
    male_coach_prompt = """You are a direct, confident relationship coach specializing in female psychology and dating dynamics.

⚠️ OVERRIDE INSTRUCTION - READ THIS FIRST:
Questions about human biology and physiology (menstruation, periods, ovulation, hormones, fertility, arousal, erections, sexual health, etc.) are NORMAL and ALLOWED when discussed in the context of dating, attraction, and relationships. These are natural biological processes that affect dating dynamics. Answer these questions professionally and helpfully.

CRITICAL SAFETY & SCOPE RULES:
- REFUSE to answer questions about self-harm, suicide, violence, or murder.
- You CAN and SHOULD answer questions about human biology, physiology, and natural processes (periods, menstruation, ovulation, erections, arousal, hormones, fertility, sexual health, etc.) when they relate to dating, attraction, or relationships.
- You CAN answer questions about influence, persuasion, manipulation psychology, and dating tactics - these are legitimate parts of attraction dynamics.
- Only refuse questions COMPLETELY unrelated to human interaction, relationships, or social dynamics (e.g., "how to fix a car", "what's the weather").
- If a question is truly out of scope or harmful, respond with: "I'm specifically designed to help with dating and relationship challenges. I can't assist with that topic, but I'm here if you have questions about attraction, communication, or relationship dynamics."

IMPORTANT INSTRUCTIONS:
- Keep responses concise and readable (200-300 words maximum).
- Use natural paragraph breaks for readability.
- You MAY use bullet points (with hyphens) when listing action steps or key points.
- Do NOT use bold text or headers.
- Write in plain text, like you are texting a knowledgeable friend.
- Be direct and insightful. Break down the situation and explain the dynamics at play.
- NEVER mention any book names, PDF names, authors, or specific sources. Present all knowledge as your own expertise.
- NEVER use exact terminology or concept names from books/PDFs. Always paraphrase and adapt concepts into your own words.
- If a concept comes from a source, rephrase it naturally without attribution (e.g., instead of "the concept of XYZ," say "I've found that...").
- Give actionable advice with clear reasoning behind it.
- Explain WHY things work the way they do, not just WHAT to do.
- Focus on: behavioural psychology, evolutionary psychology, and attraction psychology.

Your goal is to sound like a real person who deeply understands male dating dynamics, not a robot giving one-liners."""


    female_coach_prompt = """You are a direct, confident relationship coach specializing in male psychology and dating dynamics.

⚠️ OVERRIDE INSTRUCTION - READ THIS FIRST:
Questions about human biology and physiology (menstruation, periods, ovulation, hormones, fertility, arousal, erections, sexual health, etc.) are NORMAL and ALLOWED when discussed in the context of dating, attraction, and relationships. These are natural biological processes that affect dating dynamics. Answer these questions professionally and helpfully.

CRITICAL SAFETY & SCOPE RULES:
- REFUSE to answer questions about self-harm, suicide, violence, or murder.
- You CAN and SHOULD answer questions about human biology, physiology, and natural processes (periods, erections, arousal, hormones, etc.) when they relate to dating, attraction, or relationships.
- You CAN answer questions about influence, persuasion, manipulation psychology, and dating tactics - these are legitimate parts of attraction dynamics.
- Only refuse questions COMPLETELY unrelated to human interaction, relationships, or social dynamics (e.g., "how to fix a car", "what's the weather").
- If a question is truly out of scope or harmful, respond with: "I'm specifically designed to help with dating and relationship challenges. I can't assist with that topic, but I'm here if you have questions about attraction, communication, or relationship dynamics."

IMPORTANT INSTRUCTIONS:
- Keep responses concise and readable (150-200 words maximum).
- Use natural paragraph breaks for readability.
- You MAY use bullet points when listing action steps or key points.
- Do NOT use bold text or headers.
- Write in plain text, like you are texting a supportive friend.
- Be direct. Break down the situation and explain the dynamics at play.
- NEVER mention any book names, PDF names, authors, or specific sources. Present all knowledge as your own expertise.
- NEVER use exact terminology or concept names from books/PDFs. Always paraphrase and adapt concepts into your own words.
- If a concept comes from a source, rephrase it naturally without attribution (e.g., instead of "the concept of XYZ," say "I've found that...").
- Give actionable advice with clear reasoning behind it.
- Explain WHY things work the way they do, and WHAT to do.
- Focus on: attraction psychology.
- ALWAYS end with a follow-up question to continue the conversation.

Your goal is to sound like a real person who deeply understands female dating dynamics, not a robot giving one-liners."""


    # Select prompt based on gender preference
    base_prompt = male_coach_prompt if gender_preference == "male" else female_coach_prompt

    # Persona overrides
    persona_prompts = {
        "drill_sergeant": "\n\nPERSONA OVERRIDE: You are a TOUGH LOVE 'Drill Sergeant' type coach. Be brutally honest. Stop the user from making excuses. Use short, punchy sentences. Don't coddle them. Focus on discipline and action. Your tone is commanding.",
        "wingman": "\n\nPERSONA OVERRIDE: You are an enthusiastic 'Wingman'. Use casual, bro-like language (like 'dude', 'bro', 'man'). Be hype and supportive. Focus on boosting their confidence and giving practical, 'street-smart' advice. Your tone is high-energy.",
        "therapist": "\n\nPERSONA OVERRIDE: You are an Empathetic Relationship Therapist. Focus on the emotional undercurrents, anxiety, and connection. Be gentle, validating, and ask deep questions about how they feel. Use softer language. Your tone is calming."
    }
    
    # Apply persona if selected (and not standard)
    persona_instruction = persona_prompts.get(request.persona, "")

    # Add instruction to NOT use a name for guests
    name_instruction = "\n\nIMPORTANT: You are chatting with a guest user whose name is unknown. Do NOT address them by any personal name. Do NOT make up a name. You can address them directly without a name."
    system_prompt = base_prompt + name_instruction + persona_instruction

    if not GEMINI_API_KEY:
        return {"response": "Gemini API key not configured. Please check .env file."}

    try:
        # Get stored files filtered by gender preference (same as authenticated users)
        stored_files = db.query(GeminiFileDB).filter(
            GeminiFileDB.gender_category == gender_preference
        ).all()
        
        # Initialize model with system instruction
        model = genai.GenerativeModel(
            model_name="gemini-flash-latest",
            system_instruction=system_prompt
        )
        
        # Build chat history from request
        chat_history = []
        if request.history:
            for msg in request.history:
                chat_history.append({
                    "role": "user" if msg.role == "user" else "model",
                    "parts": [msg.content]
                })
        
        # Start chat session with history
        chat = model.start_chat(history=chat_history)
        
        # Prepare content parts for current message
        content_parts = []
        
        # Always include PDFs for context on every query
        for file_record in stored_files:
            try:
                file_obj = genai.get_file(file_record.file_id)
                content_parts.append(file_obj)
            except Exception as e:
                print(f"Error fetching file {file_record.file_id}: {e}")
                continue
        
        # Add user message
        content_parts.append(request.message)
        
        # Generate response with context
        response = chat.send_message(content_parts)
        return {"response": response.text}

        
    except Exception as e:
        import traceback
        print(f"Guest chat Gemini API error: {str(e)}")
        print(f"Full traceback: {traceback.format_exc()}")
        # Fallback responses
        responses = [
            "Stop seeking validation. Focus on your mission.",
            "They are testing your boundaries. Do not react emotionally.",
            "You are the prize. Act like it.",
        ]
        return {"response": random.choice(responses)}

@app.post("/guest-analyze")
async def guest_analyze_screenshot(
    file: UploadFile = File(...), 
    user_color: str = Query("blue"),
    other_color: str = Query("gray"),
    user_gender: str = Query("male"),
    other_gender: str = Query("female"),
    goal: str = Query("Build attraction"),
    language: str = Query("English"),
    tone: str = Query("balanced")
):
    if not GEMINI_API_KEY:
        return {"error": "Gemini API key not configured"}
    
    try:
        # Read the uploaded image
        image_data = await file.read()
        
        # User color and gender clarification
        color_text = f"\n\n!!!CRITICAL INSTRUCTION - READ CAREFULLY!!!\n"
        color_text += f"The user asking for help has {user_color.upper()} colored message bubbles.\n"
        color_text += f"The other person has {other_color.upper()} colored message bubbles.\n"
        color_text += f"\n"
        color_text += f"YOU MUST:\n"
        color_text += f"1. Identify ALL messages with {user_color.upper()} color - these are the USER's messages\n"
        color_text += f"2. Identify ALL messages with {other_color.upper()} color - these are the OTHER PERSON's messages\n"
        color_text += f"3. Analyze ONLY the {user_color.upper()} messages (the user's messages)\n"
        color_text += f"4. DO NOT analyze the {other_color.upper()} messages\n"
        color_text += f"5. DO NOT assume based on left/right position - ONLY use the COLOR to identify messages\n"
        color_text += f"\n"
        color_text += f"Additional context:\n"
        color_text += f"- The user is {user_gender.upper()} talking to a {other_gender.upper()}\n"
        color_text += f"- Tailor advice for {user_gender}-{other_gender} dating dynamics\n"
        color_text += f"- Focus your analysis on what the {user_color.upper()} bubble person is saying/doing"
        
        # Map tone codes to descriptions
        tone_descriptions = {
            "balanced": "a balanced mix of different styles",
            "playful": "playful and teasing",
            "direct": "direct and confident",
            "mysterious": "mysterious and intriguing",
            "casual": "casual and friendly",
            "witty": "witty and clever",
            "romantic": "romantic and sweet",
            "cocky_funny": "cocky-funny (confident with humor)",
            "challenge": "challenge/push-pull dynamics"
        }
        tone_description = tone_descriptions.get(tone, "a balanced mix of different styles")
        
        # Create system prompt for screenshot analysis
        analysis_prompt = f"""You are an expert dating coach analyzing text message screenshots.

USER'S CONVERSATION GOAL: {goal}
IMPORTANT: Tailor your assessment and reply suggestions specifically to help them achieve this exact goal.

PREFERRED TONE/STYLE: The user wants replies that are {tone_description}.
Focus on this tone while still providing variety in the 3 reply options.

ANALYSIS TASK:
1. Read the conversation in the screenshot
2. Identify who sent the LAST message in the conversation
3. Assess the power dynamics and "frame" (who is chasing whom)
4. Identify signs of attraction, interest level, or red flags
5. Determine the appropriate next action based on conversation state
6. Provide context-aware reply options that align with the goal: {goal}

CRITICAL RULES:
- Keep your analysis concise (150-200 words total)
- Be direct and strategic
- Focus on attraction psychology and power dynamics
- Do NOT mention books, PDFs, or specific methodologies
- ANALYZE THE CONVERSATION STATE CAREFULLY:
  * If the USER sent the last message and it's a QUESTION → Recommend WAITING for their response
  * If the USER is double/triple texting → Flag this as desperate behavior
  * If the OTHER PERSON hasn't responded in a while → Provide "follow-up" or "move on" advice
  * If the OTHER PERSON responded → Provide immediate reply suggestions
{color_text}

OUTPUT FORMAT:
Return a JSON object with the following keys:
- "assessment": Your analysis of the situation and power dynamics. IMPORTANT:
    * If the user should WAIT before texting, clearly state this WARNING at the start: "⚠️ DO NOT TEXT YET - Wait for her response first."
    * Explain WHY waiting is strategically important
    * Then provide your full analysis of power dynamics, attraction signals, and what's happening
    * Keep assessment and reasoning in ENGLISH
- "replies": A list of 3 distinct reply options with different TONES/STYLES. Each should be an actual text message they could send:
    * Use descriptive types like: "Playful/Teasing", "Direct/Confident", "Mysterious/Intrigue", "Challenge", "Validation-Withdrawal", "Cocky-Funny", etc.
    * Each option must have:
      - "type": The tone/style of the reply (e.g., "Playful/Ambiguous", "Direct/Challenge", "Frame Control")
      - "text": The ACTUAL message text to copy and send - WRITE THIS IN {language.upper()}
- "reasoning": Brief explanation of the overall strategy behind these reply styles (in ENGLISH)

Do not include markdown formatting (like ```json) in the response, just the raw JSON string.
"""

        # Initialize Gemini model with vision capability
        model = genai.GenerativeModel('gemini-flash-latest')
        
        # Create the image part for Gemini
        import base64
        
        # Determine mime type based on file extension or default to jpeg
        mime_type = file.content_type or "image/jpeg"
        
        image_part = {
            "mime_type": mime_type,
            "data": image_data
        }
        
        # Run synchronous generation in a thread pool to avoid blocking the event loop
        import asyncio
        from functools import partial
        
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None, 
                partial(model.generate_content, [analysis_prompt, image_part])
            )
        except Exception as e:
            print(f"Gemini generation error: {e}")
            raise HTTPException(status_code=500, detail="AI generation failed. The image might be unclear or violate safety policies.")

        # Check for safety blocks or empty response
        if not response.parts:
            print(f"Safety ratings: {response.prompt_feedback}")
            raise HTTPException(status_code=400, detail="AI refused to analyze this image due to safety filters. Please try a different image.")
            
        # Parse the response to extract assessment, reply, and reasoning
        full_response = response.text
        
        # Clean up potential markdown code blocks
        import json
        cleaned_response = full_response.replace("```json", "").replace("```", "").strip()
        
        try:
            parsed_response = json.loads(cleaned_response)
            
            # Debug: print the parsed response
            print(f"Parsed response: {parsed_response}")
            
            # Handle both old (single reply) and new (multiple replies) formats
            replies = parsed_response.get("replies", [])
            
            # Ensure replies is a list
            if isinstance(replies, str):
                replies = [{"type": "Recommended", "text": replies}]
            elif not isinstance(replies, list):
                replies = []
                
            if not replies and "reply" in parsed_response:
                replies = [{"type": "Recommended", "text": parsed_response["reply"]}]
                
            # Final fallback if no replies found - extract from assessment or regenerate
            if not replies or len(replies) == 0:
                print("WARNING: No replies in AI response, checking assessment...")
                assessment = parsed_response.get("assessment", "")
                # If assessment mentions waiting, create appropriate reply
                if "DO NOT TEXT" in assessment or "WAIT" in assessment.upper():
                    replies = [{"type": "Strategy", "text": "Wait for their response before texting again."}]
                else:
                    # Generate generic playful reply as fallback
                    replies = [
                        {"type": "Playful/Teasing", "text": "Haha fair enough 😏"},
                        {"type": "Direct/Confident", "text": "Alright, let's make it happen then"},
                        {"type": "Mysterious/Intrigue", "text": "We'll see about that 👀"}
                    ]
                
            return {
                "assessment": parsed_response.get("assessment", full_response),
                "replies": replies,
                "reasoning": parsed_response.get("reasoning", "Strategy based on conversation dynamics")
            }
        except json.JSONDecodeError as e:
            # Fallback if JSON parsing fails
            print(f"JSON decode error: {e}")
            print(f"Raw response: {cleaned_response}")
            return {
                "assessment": full_response,
                "replies": [
                    {"type": "Playful", "text": "Haha that's interesting 😏"},
                    {"type": "Direct", "text": "I see what you mean"}
                ],
                "reasoning": "Generic replies due to parsing error"
            }
    
    except Exception as e:
        print(f"Guest screenshot analysis error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "assessment": f"Unable to analyze the screenshot. Error: {str(e)}",
            "reply": "N/A",
            "reasoning": "Technical error occurred during analysis."
        }

@app.post("/update-plan")
async def update_plan(plan_data: PlanUpdate, current_user: UserDB = Depends(get_current_active_user), db: Session = Depends(get_db)):
    valid_plans = ["Sleeper", "Initiate", "Master"]
    if plan_data.plan not in valid_plans:
        raise HTTPException(status_code=400, detail="Invalid plan")
    
    current_user.subscription_plan = plan_data.plan
    if plan_data.plan != "Sleeper":
        expiry_date = datetime.utcnow() + timedelta(days=7)
        current_user.plan_expiry = expiry_date.isoformat()
    else:
        current_user.plan_expiry = None
    
    current_user.queries_used = 0
    db.commit()
    db.refresh(current_user)
    
    return {
        "message": f"Plan updated to {plan_data.plan}",
        "plan": current_user.subscription_plan,
        "expiry": current_user.plan_expiry
    }

@app.post("/update-profile")
async def update_profile(profile_data: ProfileUpdate, current_user: UserDB = Depends(get_current_active_user), db: Session = Depends(get_db)):
    if profile_data.full_name:
        current_user.full_name = profile_data.full_name
    if profile_data.email:
        existing_user = db.query(UserDB).filter(UserDB.email == profile_data.email, UserDB.id != current_user.id).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already in use")
        current_user.email = profile_data.email
    if profile_data.gender_preference and profile_data.gender_preference in ["male", "female"]:
        current_user.gender_preference = profile_data.gender_preference
    
    db.commit()
    db.refresh(current_user)
    return {"message": "Profile updated successfully", "user": current_user}

# --- Gemini File Management ---

@app.post("/admin/upload-pdf", response_model=FileUploadResponse)
async def upload_pdf(
    file: UploadFile = File(...), 
    gender_category: str = Form("male"),  # Default to male if not specified
    db: Session = Depends(get_db)
):
    print(f"Received upload request for file: {file.filename}, gender_category: {gender_category}")
    
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail="Gemini API not configured")
    
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    if gender_category not in ["male", "female"]:
        raise HTTPException(status_code=400, detail="gender_category must be 'male' or 'female'")
    
    try:
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            print(f"Uploading to Gemini: {file.filename}")
            uploaded_file = genai.upload_file(tmp_path, mime_type="application/pdf")
            print(f"Gemini upload successful. URI: {uploaded_file.name}")
            
            # Store in DB with gender category
            file_record = GeminiFileDB(
                file_id=uploaded_file.name, # Stores 'files/xxxx'
                filename=file.filename,
                uploaded_at=datetime.utcnow().isoformat(),
                purpose="assistants",
                gender_category=gender_category
            )
            db.add(file_record)
            db.commit()
            db.refresh(file_record)
            
            return FileUploadResponse(
                file_id=uploaded_file.name,
                filename=file.filename,
                message=f"File uploaded successfully for {gender_category} coach. URI: {uploaded_file.name}"
            )
            
        finally:
            # Clean up temp file
            os.unlink(tmp_path)
            
    except Exception as e:
        print(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@app.get("/admin/list-files", response_model=FileListResponse)
async def list_files(db: Session = Depends(get_db)):
    files = db.query(GeminiFileDB).all()
    file_list = [
        {
            "id": f.id,
            "file_id": f.file_id,
            "filename": f.filename,
            "uploaded_at": f.uploaded_at,
            "purpose": f.purpose,
            "gender_category": f.gender_category
        }
        for f in files
    ]
    return FileListResponse(files=file_list)

@app.delete("/admin/delete-file/{file_id:path}")
async def delete_file(file_id: str, db: Session = Depends(get_db)):
    # Note: file_id here might be the DB ID or the Gemini Name. 
    # The frontend likely sends the Gemini Name (file_id in response).
    # But wait, the frontend might send the DB ID if it uses the list-files response 'id' field?
    # Let's check the list-files response. It returns "file_id" which is the Gemini Name.
    # So we expect Gemini Name here.
    
    # However, standard URL encoding might mess up slashes in 'files/xxxx'.
    # FastAPI handles path params, but 'files/xxxx' might be tricky.
    # Let's assume the frontend sends it correctly or we handle it.
    # Alternatively, we can accept the DB ID if that's easier, but let's stick to the existing pattern.
    # Actually, if the frontend sends "files/xxxx", it needs to be URL encoded.
    
    # Let's look up by file_id (Gemini Name)
    file_record = db.query(GeminiFileDB).filter(GeminiFileDB.file_id == file_id).first()
    if not file_record:
        # Try finding by ID if it's an integer
        try:
            int_id = int(file_id)
            file_record = db.query(GeminiFileDB).filter(GeminiFileDB.id == int_id).first()
        except:
            pass
            
    if not file_record:
        raise HTTPException(status_code=404, detail="File not found in database")
    
    try:
        # Delete from Gemini
        try:
            genai.delete_file(file_record.file_id)
        except Exception as e:
            print(f"Warning: Could not delete from Gemini (might already be gone): {e}")
        
        # Delete from DB
        db.delete(file_record)
        db.commit()
        
        return {"message": f"File {file_record.filename} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")

@app.post("/admin/replace-all-files")
async def replace_all_files(
    files: List[UploadFile] = File(...), 
    gender_category: str = Form("male"),
    db: Session = Depends(get_db)
):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail="Gemini API not configured")
    
    if gender_category not in ["male", "female"]:
        raise HTTPException(status_code=400, detail="gender_category must be 'male' or 'female'")
    
    try:
        # Delete all existing files FOR THIS CATEGORY
        # We should only replace files for the specific coach type, not ALL files
        existing_files = db.query(GeminiFileDB).filter(GeminiFileDB.gender_category == gender_category).all()
        for file_record in existing_files:
            try:
                genai.delete_file(file_record.file_id)
            except:
                pass
            db.delete(file_record)
        db.commit()
        
        # Upload new files
        uploaded_files = []
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                continue
            
            # Save to temp
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name
            
            try:
                uploaded_file = genai.upload_file(tmp_path, mime_type="application/pdf")
                
                file_record = GeminiFileDB(
                    file_id=uploaded_file.name,
                    filename=file.filename,
                    uploaded_at=datetime.utcnow().isoformat(),
                    purpose="assistants",
                    gender_category=gender_category
                )
                db.add(file_record)
                uploaded_files.append({
                    "file_id": uploaded_file.name,
                    "filename": file.filename
                })
            finally:
                os.unlink(tmp_path)
        
        db.commit()
        
        return {
            "message": f"Replaced all files. {len(uploaded_files)} files uploaded.",
            "files": uploaded_files
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error replacing files: {str(e)}")

# --- Blog Management ---

@app.get("/blogs", response_model=BlogList)
async def get_blogs(
    category: str = None,
    search: str = None,
    db: Session = Depends(get_db)
):
    """Get all published blogs for public viewing with optional filtering."""
    query = db.query(BlogDB).filter(BlogDB.published == True)

    # Filter by category if provided
    if category and category != "all":
        query = query.filter(BlogDB.categories.contains(category))

    # Filter by search term if provided
    if search:
        search_term = f"%{search}%"
        query = query.filter(
            (BlogDB.title.ilike(search_term)) |
            (BlogDB.content.ilike(search_term)) |
            (BlogDB.excerpt.ilike(search_term))
        )

    blogs = query.order_by(BlogDB.published_at.desc()).all()
    return {"blogs": blogs}

@app.get("/blogs/categories")
async def get_blog_categories(db: Session = Depends(get_db)):
    """Get all unique categories from published blogs."""
    blogs = db.query(BlogDB).filter(BlogDB.published == True).all()

    categories = set()
    for blog in blogs:
        if blog.categories:
            blog_categories = [cat.strip() for cat in blog.categories.split(',') if cat.strip()]
            categories.update(blog_categories)

    return {"categories": sorted(list(categories))}

@app.get("/blogs/{slug}")
async def get_blog(slug: str, db: Session = Depends(get_db)):
    """Get a single blog by slug."""
    blog = db.query(BlogDB).filter(BlogDB.slug == slug, BlogDB.published == True).first()
    if not blog:
        raise HTTPException(status_code=404, detail="Blog not found")
    return blog

@app.post("/admin/blogs", response_model=Blog)
async def create_blog(blog: BlogCreate, db: Session = Depends(get_db)):
    """Create a new blog (admin only)."""
    # Check if slug already exists
    existing_blog = db.query(BlogDB).filter(BlogDB.slug == blog.slug).first()
    if existing_blog:
        raise HTTPException(status_code=400, detail="Blog with this slug already exists")

    now = datetime.utcnow().isoformat()
    db_blog = BlogDB(
        **blog.dict(),
        created_at=now,
        updated_at=now,
        published_at=now if blog.published else None
    )
    db.add(db_blog)
    db.commit()
    db.refresh(db_blog)
    return db_blog

@app.put("/admin/blogs/{blog_id}", response_model=Blog)
async def update_blog(blog_id: int, blog_update: BlogUpdate, db: Session = Depends(get_db)):
    """Update a blog (admin only)."""
    db_blog = db.query(BlogDB).filter(BlogDB.id == blog_id).first()
    if not db_blog:
        raise HTTPException(status_code=404, detail="Blog not found")

    # Check if new slug conflicts with another blog
    if blog_update.slug != db_blog.slug:
        existing_blog = db.query(BlogDB).filter(BlogDB.slug == blog_update.slug, BlogDB.id != blog_id).first()
        if existing_blog:
            raise HTTPException(status_code=400, detail="Blog with this slug already exists")

    update_data = blog_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        if value is not None:
            setattr(db_blog, field, value)

    db_blog.updated_at = datetime.utcnow().isoformat()
    if blog_update.published and not db_blog.published_at:
        db_blog.published_at = datetime.utcnow().isoformat()

    db.commit()
    db.refresh(db_blog)
    return db_blog

@app.delete("/admin/blogs/{blog_id}")
async def delete_blog(blog_id: int, db: Session = Depends(get_db)):
    """Delete a blog (admin only)."""
    db_blog = db.query(BlogDB).filter(BlogDB.id == blog_id).first()
    if not db_blog:
        raise HTTPException(status_code=404, detail="Blog not found")

    # Delete associated image file if it exists
    if db_blog.featured_image:
        try:
            image_path = os.path.join("uploads", "blogs", db_blog.featured_image)
            if os.path.exists(image_path):
                os.remove(image_path)
        except Exception as e:
            print(f"Warning: Could not delete blog image: {e}")

    db.delete(db_blog)
    db.commit()
    return {"message": "Blog deleted successfully"}

@app.get("/admin/blogs", response_model=BlogList)
async def get_all_blogs(db: Session = Depends(get_db)):
    """Get all blogs including unpublished ones (admin only)."""
    blogs = db.query(BlogDB).order_by(BlogDB.created_at.desc()).all()
    return {"blogs": blogs}

@app.post("/admin/upload-blog-image")
async def upload_blog_image(file: UploadFile = File(...)):
    """Upload an image for blog posts."""
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
        raise HTTPException(status_code=400, detail="Only image files are allowed")

    # Create uploads/blogs directory if it doesn't exist
    upload_dir = os.path.join("uploads", "blogs")
    os.makedirs(upload_dir, exist_ok=True)

    # Generate unique filename
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}{file_extension}"
    file_path = os.path.join(upload_dir, unique_filename)

    # Save the file
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")

    return {
        "filename": unique_filename,
        "url": f"/uploads/blogs/{unique_filename}",
        "message": "Image uploaded successfully"
    }

# --- Blog Comment Endpoints ---

@app.post("/blogs/{slug}/comments")
async def create_comment(
    slug: str,
    comment: CommentCreate,
    current_user: UserDB = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create a new comment or reply on a blog post. Requires authentication."""
    # Get the blog
    blog = db.query(BlogDB).filter(BlogDB.slug == slug, BlogDB.published == True).first()
    if not blog:
        raise HTTPException(status_code=404, detail="Blog not found")
    
    # If this is a reply, verify parent comment exists
    if comment.parent_id:
        parent = db.query(BlogCommentDB).filter(BlogCommentDB.id == comment.parent_id).first()
        if not parent:
            raise HTTPException(status_code=404, detail="Parent comment not found")
    
    # Check if user is admin (you can add admin role check here)
    is_admin = current_user.username == "admin"  # Simple check, customize as needed
    
    # Create the comment
    db_comment = BlogCommentDB(
        blog_id=blog.id,
        user_id=current_user.id,
        username=current_user.username,
        content=comment.content,
        parent_id=comment.parent_id,
        is_admin=is_admin,
        created_at=datetime.utcnow().isoformat(),
        updated_at=datetime.utcnow().isoformat()
    )
    
    db.add(db_comment)
    db.commit()
    db.refresh(db_comment)
    
    return {"message": "Comment created successfully", "comment_id": db_comment.id}

@app.get("/blogs/{slug}/comments", response_model=List[CommentResponse])
async def get_comments(slug: str, db: Session = Depends(get_db)):
    """Get all comments for a blog post, organized hierarchically."""
    # Get the blog
    blog = db.query(BlogDB).filter(BlogDB.slug == slug, BlogDB.published == True).first()
    if not blog:
        raise HTTPException(status_code=404, detail="Blog not found")
    
    # Get all comments for this blog
    comments = db.query(BlogCommentDB).filter(BlogCommentDB.blog_id == blog.id).order_by(BlogCommentDB.created_at.asc()).all()
    
    # Build hierarchical structure
    comment_dict = {}
    root_comments = []
    
    for comment in comments:
        comment_data = CommentResponse(
            id=comment.id,
            blog_id=comment.blog_id,
            user_id=comment.user_id,
            username=comment.username,
            content=comment.content,
            parent_id=comment.parent_id,
            is_admin=comment.is_admin,
            created_at=comment.created_at,
            updated_at=comment.updated_at,
            replies=[]
        )
        comment_dict[comment.id] = comment_data
        
        if comment.parent_id is None:
            root_comments.append(comment_data)
    
    # Attach replies to their parents
    for comment in comments:
        if comment.parent_id and comment.parent_id in comment_dict:
            comment_dict[comment.parent_id].replies.append(comment_dict[comment.id])
    
    return root_comments

@app.delete("/blogs/{slug}/comments/{comment_id}")
async def delete_comment(
    slug: str,
    comment_id: int,
    current_user: UserDB = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete a comment. Users can delete their own comments, admins can delete any."""
    comment = db.query(BlogCommentDB).filter(BlogCommentDB.id == comment_id).first()
    if not comment:
        raise HTTPException(status_code=404, detail="Comment not found")
    
    # Check permission: user owns comment or user is admin
    is_admin = current_user.username == "admin"
    if comment.user_id != current_user.id and not is_admin:
        raise HTTPException(status_code=403, detail="Not authorized to delete this comment")
    
    # Delete all replies to this comment recursively
    def delete_replies(parent_id):
        replies = db.query(BlogCommentDB).filter(BlogCommentDB.parent_id == parent_id).all()
        for reply in replies:
            delete_replies(reply.id)
            db.delete(reply)
    
    delete_replies(comment_id)
    db.delete(comment)
    db.commit()
    
    return {"message": "Comment deleted successfully"}
