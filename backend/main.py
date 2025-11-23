from fastapi import FastAPI, Request, UploadFile, File, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import create_engine, Column, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import random

# --- Configuration ---
SECRET_KEY = "your-secret-key-change-this-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
SQLALCHEMY_DATABASE_URL = "sqlite:///./users.db"

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
    subscription_plan = Column(String, default="Sleeper")  # Sleeper, Initiate, Master
    plan_expiry = Column(String, nullable=True)  # ISO date string

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
    plan: str  # Sleeper, Initiate, Master

class ProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None

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
    """Returns the query limit for a given subscription plan."""
    limits = {
        "Sleeper": 10,
        "Initiate": 100,
        "Master": 999999  # Unlimited
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
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Routes ---

@app.get("/")
async def read_root():
    return {"message": "Welcome to FastAPI + Bootstrap API"}

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
async def analyze_screenshot(file: UploadFile = File(...), current_user: UserDB = Depends(get_current_active_user), db: Session = Depends(get_db)):
    # Increment usage
    current_user.screenshots_analyzed += 1
    db.commit()

    # Mock analysis logic
    return {
        "assessment": "They are testing your compliance. This is a classic test designed to see if you will jump through their hoops.",
        "reply": "Haha, nice try. I'm busy tonight, but I might be free on Thursday.",
        "reasoning": "This reply maintains your frame, shows you have a life (high value), and sets the terms for the interaction on your schedule."
    }

@app.post("/guest-chat")
async def guest_chat(request: ChatRequest):
    """Chat endpoint for non-authenticated users (limited to 5 queries via frontend)."""
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

@app.post("/guest-analyze")
async def guest_analyze_screenshot(file: UploadFile = File(...)):
    """Screenshot analysis endpoint for non-authenticated users (limited to 5 queries via frontend)."""
    return {
        "assessment": "They are testing your compliance. This is a classic test designed to see if you will jump through their hoops.",
        "reply": "Haha, nice try. I'm busy tonight, but I might be free on Thursday.",
        "reasoning": "This reply maintains your frame, shows you have a life (high value), and sets the terms for the interaction on your schedule."
    }

@app.post("/update-plan")
async def update_plan(plan_data: PlanUpdate, current_user: UserDB = Depends(get_current_active_user), db: Session = Depends(get_db)):
    """Update user's subscription plan."""
    valid_plans = ["Sleeper", "Initiate", "Master"]
    if plan_data.plan not in valid_plans:
        raise HTTPException(status_code=400, detail="Invalid plan")
    
    current_user.subscription_plan = plan_data.plan
    
    # Set plan expiry (7 days from now for paid plans)
    if plan_data.plan != "Sleeper":
        expiry_date = datetime.utcnow() + timedelta(days=7)
        current_user.plan_expiry = expiry_date.isoformat()
    else:
        current_user.plan_expiry = None
    
    # Reset query counter when upgrading
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
    """Update user profile information."""
    if profile_data.full_name:
        current_user.full_name = profile_data.full_name
    if profile_data.email:
        # Check if email is already taken
        existing_user = db.query(UserDB).filter(UserDB.email == profile_data.email, UserDB.id != current_user.id).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already in use")
        current_user.email = profile_data.email
    
    db.commit()
    db.refresh(current_user)
    
    return {"message": "Profile updated successfully", "user": current_user}
