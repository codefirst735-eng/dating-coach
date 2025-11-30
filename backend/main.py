from fastapi import FastAPI, Request, UploadFile, File, Depends, HTTPException, status
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
SQLALCHEMY_DATABASE_URL = "sqlite:///./users.db"
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
    gender_preference: Optional[str] = "male"  # For guest users
    history: Optional[List[ChatMessage]] = []  # Conversation history for context


class PlanUpdate(BaseModel):
    plan: str

class ProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None
    gender_preference: Optional[str] = None  # 'male' or 'female'

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
        "Initiate": 100,
        "Master": 999999
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_origin_regex=r"https://.*\.vercel\.app",
)

# --- Routes ---

@app.get("/")
async def read_root():
    return {"message": "Welcome to FastAPI + Gemini API"}

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
    male_coach_prompt = """You are a direct, confident relationship coach specializing in male psychology and dating dynamics. You understand masculine energy, frame control, attraction triggers, and how men can build genuine confidence and success with women.

CONTEXT: Your users are from India. Use Indian names, cultural context, and dating scenarios relevant to Indian society when giving examples.

IMPORTANT INSTRUCTIONS:
- Keep responses concise and readable (150-200 words maximum).
- Use natural paragraph breaks for readability.
- You MAY use bullet points (with hyphens) when listing action steps or key points.
- Do NOT use bold text or headers.
- Write in plain text, like you are texting a knowledgeable friend.
- Be direct and insightful. Break down the situation and explain the dynamics at play.
- Refer PDFs uploaded but NEVER mention any book names, PDF names, authors, or specific sources but take your knowledge from them.
- NEVER use exact terminology or concept names from books/PDFs. Always paraphrase and adapt concepts into your own words.
- If a concept comes from a source, rephrase it naturally without attribution (e.g., instead of "the concept of XYZ," say "I've found that...").
- Give actionable advice with clear reasoning behind it.
- Explain WHY things work the way they do, not just WHAT to do.
- Focus on: masculine frame, leadership, attraction psychology, maintaining boundaries, building value.
- ALWAYS end with a follow-up question to continue the conversation.

Your goal is to sound like a real person who deeply understands male dating dynamics, not a robot giving one-liners."""


    female_coach_prompt = """You are a warm, insightful relationship coach specializing in female psychology and modern dating. You understand feminine energy, emotional intelligence, relationship dynamics, and how women can navigate dating while maintaining their standards and authenticity.

CONTEXT: Your users are from India. Use Indian names, cultural context, and dating scenarios relevant to Indian society when giving examples.

IMPORTANT INSTRUCTIONS:
- Keep responses concise and readable (150-200 words maximum).
- Use natural paragraph breaks for readability.
- You MAY use bullet points (with hyphens) when listing action steps or key points.
- Do NOT use bold text or headers.
- Write in plain text, like you are texting a supportive friend.
- Be direct yet empathetic. Break down the situation and explain the dynamics at play.
- NEVER mention any book names, PDF names, authors, or specific sources. Present all knowledge as your own expertise.
- NEVER use exact terminology or concept names from books/PDFs. Always paraphrase and adapt concepts into your own words.
- If a concept comes from a source, rephrase it naturally without attribution (e.g., instead of "the concept of XYZ," say "I've found that...").
- Give actionable advice with clear reasoning behind it.
- Explain WHY things work the way they do, not just WHAT to do.
- Focus on: recognizing red flags, maintaining standards, emotional intelligence, authentic connection, self-worth.
- ALWAYS end with a follow-up question to continue the conversation.

Your goal is to sound like a real person who deeply understands female dating dynamics, not a robot giving one-liners."""


    # Select prompt based on user's gender preference
    base_prompt = male_coach_prompt if current_user.gender_preference == "male" else female_coach_prompt
    
    # Add user's name to the prompt to prevent hallucinated names
    user_name = current_user.full_name if current_user.full_name else "User"
    user_name_instruction = f"\n\nIMPORTANT: The user's name is '{user_name}'. Address them by this name occasionally. Do NOT make up a name for the user."
    system_prompt = base_prompt + user_name_instruction

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
            model_name="gemini-2.0-flash",
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
        
        # Add files to context if this is the first message or if we want to always include them
        if len(history_messages) == 0:  # First message, include PDFs for context
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

@app.post("/analyze-screenshot")
async def analyze_screenshot(file: UploadFile = File(...), current_user: UserDB = Depends(get_current_active_user), db: Session = Depends(get_db)):
    current_user.screenshots_analyzed += 1
    db.commit()
    
    # For now, mock analysis or implement vision capability if needed
    # Gemini supports vision too!
    return {
        "assessment": "They are testing your compliance. This is a classic test designed to see if you will jump through their hoops.",
        "reply": "Haha, nice try. I'm busy tonight, but I might be free on Thursday.",
        "reasoning": "This reply maintains your frame, shows you have a life (high value), and sets the terms for the interaction on your schedule."
    }

@app.post("/guest-chat")
async def guest_chat(request: ChatRequest, db: Session = Depends(get_db)):
    """Chat endpoint for non-authenticated users with gender preference support."""
    
    # Validate gender preference
    gender_preference = request.gender_preference if request.gender_preference in ["male", "female"] else "male"
    
    # Gender-specific system prompts (same as authenticated users)
    male_coach_prompt = """You are a direct, confident relationship coach specializing in male psychology and dating dynamics. You understand masculine energy, frame control, attraction triggers, and how men can build genuine confidence and success with women.

CONTEXT: Your users are from India. Use Indian names, cultural context, and dating scenarios relevant to Indian society when giving examples.

IMPORTANT INSTRUCTIONS:
- Keep responses concise and readable (150-200 words maximum).
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
- Focus on: masculine frame, leadership, attraction psychology, maintaining boundaries, building value.
- ALWAYS end with a follow-up question to continue the conversation.

Your goal is to sound like a real person who deeply understands male dating dynamics, not a robot giving one-liners."""


    female_coach_prompt = """You are a warm, insightful relationship coach specializing in female psychology and modern dating. You understand feminine energy, emotional intelligence, relationship dynamics, and how women can navigate dating while maintaining their standards and authenticity.

CONTEXT: Your users are from India. Use Indian names, cultural context, and dating scenarios relevant to Indian society when giving examples.

IMPORTANT INSTRUCTIONS:
- Keep responses concise and readable (150-200 words maximum).
- Use natural paragraph breaks for readability.
- You MAY use bullet points (with hyphens) when listing action steps or key points.
- Do NOT use bold text or headers.
- Write in plain text, like you are texting a supportive friend.
- Be direct yet empathetic. Break down the situation and explain the dynamics at play.
- NEVER mention any book names, PDF names, authors, or specific sources. Present all knowledge as your own expertise.
- NEVER use exact terminology or concept names from books/PDFs. Always paraphrase and adapt concepts into your own words.
- If a concept comes from a source, rephrase it naturally without attribution (e.g., instead of "the concept of XYZ," say "I've found that...").
- Give actionable advice with clear reasoning behind it.
- Explain WHY things work the way they do, not just WHAT to do.
- Focus on: recognizing red flags, maintaining standards, emotional intelligence, authentic connection, self-worth.
- ALWAYS end with a follow-up question to continue the conversation.

Your goal is to sound like a real person who deeply understands female dating dynamics, not a robot giving one-liners."""


    # Select prompt based on gender preference
    base_prompt = male_coach_prompt if gender_preference == "male" else female_coach_prompt

    # Add instruction to NOT use a name for guests
    name_instruction = "\n\nIMPORTANT: You are chatting with a guest user whose name is unknown. Do NOT address them by any personal name. Do NOT make up a name. You can address them directly without a name."
    system_prompt = base_prompt + name_instruction

    if not GEMINI_API_KEY:
        return {"response": "Gemini API key not configured. Please check .env file."}

    try:
        # Get stored files filtered by gender preference (same as authenticated users)
        stored_files = db.query(GeminiFileDB).filter(
            GeminiFileDB.gender_category == gender_preference
        ).all()
        
        # Initialize model with system instruction
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
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
        
        # Add files to context if this is the first message
        if len(chat_history) == 0:  # First message, include PDFs for context
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
async def guest_analyze_screenshot(file: UploadFile = File(...)):
    return {
        "assessment": "They are testing your compliance. This is a classic test designed to see if you will jump through their hoops.",
        "reply": "Haha, nice try. I'm busy tonight, but I might be free on Thursday.",
        "reasoning": "This reply maintains your frame, shows you have a life (high value), and sets the terms for the interaction on your schedule."
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
    gender_category: str = "male",  # Default to male if not specified
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
async def replace_all_files(files: List[UploadFile] = File(...), db: Session = Depends(get_db)):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail="Gemini API not configured")
    
    try:
        # Delete all existing files
        existing_files = db.query(GeminiFileDB).all()
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
                    purpose="assistants"
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
