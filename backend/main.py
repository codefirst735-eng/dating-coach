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
ACCESS_TOKEN_EXPIRE_MINUTES = 30
SQLALCHEMY_DATABASE_URL = "sqlite:///./users.db"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Initialize Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("WARNING: GEMINI_API_KEY not set. AI features will be disabled.")

app = FastAPI()

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

class GeminiFileDB(Base):
    __tablename__ = "openai_files" # Keep table name to avoid migration issues
    id = Column(Integer, primary_key=True, index=True)
    file_id = Column(String, unique=True, index=True) # Stores Gemini file.name (e.g. files/xxxx)
    filename = Column(String)
    uploaded_at = Column(String)
    purpose = Column(String, default="assistants")
    
class MessageDB(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    role = Column(String) # user or assistant
    content = Column(String)
    timestamp = Column(String) # ISO date string

Base.metadata.create_all(bind=engine)

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

class ChatRequest(BaseModel):
    message: str

class PlanUpdate(BaseModel):
    plan: str

class ProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None

class FileUploadResponse(BaseModel):
    file_id: str
    filename: str
    message: str

class FileListResponse(BaseModel):
    files: List[dict]

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
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:8081",
        "http://127.0.0.1:8081",
        "https://*.vercel.app",  # Allow all Vercel preview deployments
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_origin_regex=r"https://.*\.vercel\.app",  # Regex for Vercel domains
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
    
    # System prompt
    system_prompt = """You are a direct, confident, redpill no bullshit dating coach. You specialize in clarity, masculine energy, male and female psychology and outcome-focused guidance. 

IMPORTANT INSTRUCTIONS:
- Keep your responses CONCISE and HUMAN-LIKE.
- Do NOT use Markdown formatting (no bold, no lists, no headers).
- Write in plain text, like you are texting a friend.
- Be direct and to the point. Avoid long lectures.
- Use the uploaded PDFs as knowledge sources but don't explicitly mention them unless asked.
- Focus on actionable advice.
- If the user asks a simple question, give a simple answer.

Your goal is to sound like a real person, not a robot."""

    if not GEMINI_API_KEY:
        return {"response": "Gemini API key not configured. Please check .env file."}

    try:
        # Get stored files
        stored_files = db.query(GeminiFileDB).all()
        
        # Prepare content parts
        content_parts = []
        
        # Add files to context
        for file_record in stored_files:
            try:
                # Fetch file metadata from Gemini to ensure it exists and get the object
                # In Gemini Python SDK, we can pass the file name (URI) or the file object.
                # Passing the name string "files/..." works in some contexts, but getting the object is safer.
                file_obj = genai.get_file(file_record.file_id)
                content_parts.append(file_obj)
            except Exception as e:
                print(f"Error fetching file {file_record.file_id}: {e}")
                # If file is missing in Gemini but in DB, we might want to ignore it
                continue
        
        # Add user message
        content_parts.append(request.message)
        
        # Initialize model
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=system_prompt
        )
        
        # Generate response
        response = model.generate_content(content_parts)
        ai_response_text = response.text
        
        # Save message to history
        user_msg = MessageDB(
            user_id=current_user.id,
            role="user",
            content=request.message,
            timestamp=datetime.utcnow()
        )
        ai_msg = MessageDB(
            user_id=current_user.id,
            role="assistant",
            content=ai_response_text,
            timestamp=datetime.utcnow()
        )
        db.add(user_msg)
        db.add(ai_msg)
        db.commit()
        
        return {"response": ai_response_text}
        
    except Exception as e:
        print(f"Gemini API error: {str(e)}")
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
            timestamp=datetime.utcnow()
        )
        ai_msg = MessageDB(
            user_id=current_user.id,
            role="assistant",
            content=fallback_response,
            timestamp=datetime.utcnow()
        )
        db.add(user_msg)
        db.add(ai_msg)
        db.commit()
        
        return {"response": fallback_response}

@app.get("/chat/history")
async def get_chat_history(current_user: UserDB = Depends(get_current_active_user), db: Session = Depends(get_db)):
    messages = db.query(MessageDB).filter(MessageDB.user_id == current_user.id).order_by(MessageDB.timestamp).all()
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
async def guest_chat(request: ChatRequest):
    # Same logic as chat but without user tracking/files for now, or limited
    # For simplicity, just return a canned response or use Gemini without files
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
    
    db.commit()
    db.refresh(current_user)
    return {"message": "Profile updated successfully", "user": current_user}

# --- Gemini File Management ---

@app.post("/admin/upload-pdf", response_model=FileUploadResponse)
async def upload_pdf(file: UploadFile = File(...), db: Session = Depends(get_db)):
    print(f"Received upload request for file: {file.filename}")
    
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail="Gemini API not configured")
    
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
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
            
            # Store in DB
            file_record = GeminiFileDB(
                file_id=uploaded_file.name, # Stores 'files/xxxx'
                filename=file.filename,
                uploaded_at=datetime.utcnow().isoformat(),
                purpose="assistants"
            )
            db.add(file_record)
            db.commit()
            db.refresh(file_record)
            
            return FileUploadResponse(
                file_id=uploaded_file.name,
                filename=file.filename,
                message=f"File uploaded successfully. URI: {uploaded_file.name}"
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
            "purpose": f.purpose
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
