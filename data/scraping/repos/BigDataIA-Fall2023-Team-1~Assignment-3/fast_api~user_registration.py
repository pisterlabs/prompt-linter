from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from passlib.context import CryptContext
from datetime import datetime, timedelta
import jwt
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, MetaData, Table
from sqlalchemy.orm import Session
from databases import Database
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
import logging
from dotenv import load_dotenv

# Import SequenceMatcher for text similarity
from difflib import SequenceMatcher  

#QA_Openai
from fastapi import FastAPI
from pydantic import BaseModel
import openai
import pinecone

# Load environment variables from the .env file
load_dotenv()

# Initialize a dictionary to store previous questions and their answers
previous_questions = {}

# Environment Variables
api_key = os.getenv("OPENAI_KEY")
pinecone_api_key = os.getenv("PINECONE_API")
postgres_user = os.getenv("POSTGRES_USER")
postgres_password = os.getenv("POSTGRES_PASSWORD")
host_ip_address = os.getenv("POSTGRES_HOST")
port = os.getenv("POSTGRES_PORT")
postgres_db = os.getenv("POSTGRES_DB")

# Log metrics
logging.basicConfig(level=logging.INFO)

# Fast API
app = FastAPI()

# Initialize Pinecone
pinecone.init(api_key=pinecone_api_key, environment="gcp-starter")
index = pinecone.Index(index_name='my-index')

class UserInput(BaseModel):
    forms: list  # Change from 'form' to 'forms' to accept a list of selected forms
    question: str

def generate_answer(question: str):
    try:
        # Create embeddings for the given 'question' using the specified EMBEDDING_MODEL
        openai.api_key = api_key
        EMBEDDING_MODEL = "text-embedding-ada-002"
        response = openai.Embedding.create(model=EMBEDDING_MODEL, input=question)

        # Extract the embeddings from the API response
        embeddings = response["data"][0]["embedding"]

        return embeddings
    except Exception as e:
        return str(e)


# Database setup
DATABASE_URL = f"postgresql://{postgres_user}:{postgres_password}@{host_ip_address}:{port}/{postgres_db}"
database = Database(DATABASE_URL)
metadata = MetaData()
Base = declarative_base()
engine = create_engine(DATABASE_URL)

# Define the 'users' table
users = Table(
    "users",
    metadata,
    Column("id", Integer, primary_key=True, index=True),
    Column("username", String, unique=True, index=True),
    Column("full_name", String),
    Column("email", String, unique=True, index=True),
    Column("hashed_password", String),
    Column("active", Boolean, default=True),
    Column("created_at", DateTime, default=datetime.utcnow),
)

# JWT
SECRET_KEY = "e41f6b654b3e3f41e3a030ef783cbe2fec5824d5543a0d408ee3ba0677e1750a" 
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Initialize the database
Base.metadata.create_all(bind=engine)

# Hashing password
password_hash = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Initialize sessionmaker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# User Pydantic model
class User(BaseModel):
    username: str
    full_name: str
    email: str

# Token Pydantic model
class Token(BaseModel):
    access_token: str
    token_type: str

# User registration
class UserInDB(User):
    password: str

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# User model
class UserDB(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    full_name = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

# Get user data
def get_user(db, username: str):
    return db.query(UserDB).filter(UserDB.username == username).first()

def verify_password(plain_password, hashed_password):
    return password_hash.verify(plain_password, hashed_password)

def create_access_token(data, expires_delta):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# OAuth2 password scheme for token generation
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Register a new user
@app.post("/register", response_model=User)
async def register(user: UserInDB, db: Session = Depends(get_db)):
    existing_user = get_user(db, user.username)
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = password_hash.hash(user.password)
    new_user = UserDB(username=user.username, full_name=user.full_name, email=user.email, hashed_password=hashed_password)
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return User(username=new_user.username, full_name=new_user.full_name, email=new_user.email)

# Login and get JWT token
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    db = SessionLocal()  # Get the database session
    user = get_user(db, form_data.username)
    if user is None or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}

# Protected route
@app.get("/protected")
async def get_protected_data(current_user: User = Depends(oauth2_scheme)):
    return current_user

# QA Processing

# Function to find the most similar previous question
def find_similar_previous_question(question):
    max_similarity = 0
    best_match_question = None

    for prev_question in previous_questions.keys():
        similarity = SequenceMatcher(None, question, prev_question).ratio()
        if similarity > max_similarity:
            max_similarity = similarity
            best_match_question = prev_question

    if max_similarity >= 0.85:
        return best_match_question
    else:
        return None

@app.post("/process_question")
async def process_question(input_data: UserInput, current_user: User = Depends(oauth2_scheme)):
    try:
        # Check for similar previous questions
        similar_question = find_similar_previous_question(input_data.question)

        if similar_question is not None:
            # Retrieve the answer from the dictionary
            similar_answer = previous_questions[similar_question]
            return {"answer": similar_answer}

        # If no similar previous question is found, continue with OpenAI processing
        embeddings = generate_answer(input_data.question)

        if isinstance(embeddings, str):
            return {"error": embeddings}

        filter_condition = {"form_title": {"$in": input_data.forms}}
        results = index.query(embeddings, top_k=1, include_metadata=True, filter=filter_condition)

        logging.info(f"User Question: {input_data.question}")
        logging.info(f"Selected Forms: {input_data.forms}")
        logging.info(f"Embeddings: {embeddings}")
        logging.info(f"filter_condition: {filter_condition}")
        logging.info(f"results: {results}")

        if results['matches'][0]['score'] < 0.74:
            return {"answer": "Your question is out of scope"}

        best_match_question = results['matches'][0]['metadata']['content']

        answer = openai.Completion.create(
            engine="text-davinci-002",
            temperature=0.3,
            n=1,
            prompt=f"Answer the following question: {best_match_question}",
            max_tokens=100
        )
        logging.info(f"OpenAI Response: {answer}")

        # Store the current question and its answer in the dictionary
        previous_questions[input_data.question] = answer.choices[0].text

        return {"answer": answer.choices[0].text}
    except Exception as e:
        return {"error": str(e)}
