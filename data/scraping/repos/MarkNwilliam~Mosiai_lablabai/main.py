from fastapi import FastAPI, HTTPException, status, Depends  # Added Depends here
from pydantic import BaseModel, EmailStr  
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional
from typing import List
import cohere
from fastapi.middleware.cors import CORSMiddleware
from bson import ObjectId
from fastapi.encoders import jsonable_encoder
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
import uuid
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from datetime import timedelta
from datetime import datetime


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


co = cohere.Client('')
uri = ""

# Create a new client and connect to the server
client2 = MongoClient(uri, server_api=ServerApi('1'))

client = AsyncIOMotorClient(uri)


db = client.MosiAi
collection = db.users
company_collection = db.companies 
document_collection = db.documents
chatbot_collection = db.chatbots
chat_session_collection = db.chat_sessions


class Prompt(BaseModel):
    prompt: str


try:
    client2.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)


# General function to call Cohere API
async def call_cohere(prompt: str):
    try:
        response = co.generate(prompt=prompt, model='large', max_tokens=50)
        return response.generations[0].text
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class Product(BaseModel):
    productName: str
    productDescription: str
    industry: str

class DeleteChatbotRequest(BaseModel):
    id: str

# Define the CompanyData model
class CompanyData(BaseModel):
    email: str
    companyName: str
    companyDescription: str
    products: List[Product]

class ChatMessage(BaseModel):
    session_id: str
    user_name: str
    text: str
    timestamp: datetime = datetime.now()


class FinancialData(BaseModel):
    data: str  # CSV data as a string

class LegalDocumentRequest(BaseModel):
    email: str
    description: str
    additionalInfo: Optional[str] = None  # Additional details for document generation


class DocumentAnalysisRequest(BaseModel):
    email: str  # User's email to fetch additional user data
    text: str   # Text content of the document
    description: Optional[str] = None  # Optional description or instructions

class FeedbackAnalysisRequest(BaseModel):
    email: str
    data: str  # CSV data as a string
    context: Optional[str] = None

class CustomerData(BaseModel):
    data: str  # CSV data as a string
    context: Optional[str] = None

# Pydantic models
class Document(BaseModel):
    title: str
    snippet: str
    category: Optional[str] = "general"  # default to general

class UserDocument(BaseModel):
    email: str
    documents: List[Document]

class ChatSession(BaseModel):
    session_id: str
    chatbot_id: str
    chat_history: List[ChatMessage]
    last_active: datetime = datetime.now()


class EmailContentRequest(BaseModel):
    email: str
    description: str
    prompt: str

class LegalDocumentAnalysisRequest(BaseModel):
    email: str
    text: str  # Text content of the legal document
    description: Optional[str] = None  # Optional description or instructions


class DeleteDocumentRequest(BaseModel):
    email: str
    document_title: str
    category: str  # Add this line


class StartChatSessionRequest(BaseModel):
    chatbot_id: str


class BlogContentRequest(BaseModel):
    email: str
    description: str

class ChatbotCreationRequest(BaseModel):
    email: str
    type: str
    goals: Optional[str] = None
    description: Optional[str] = None

class Chatbot(BaseModel):
    id: str
    email: str
    type: str
    goals: Optional[str] = None
    description: Optional[str] = None

class SocialMediaContentRequest(BaseModel):
    email: str
    type: str  # e.g., "twitter", "facebook", "linkedin", etc.
    prompt: str

class User(BaseModel):
    full_name: str
    email: str
    password: str  # In a real-world scenario, ensure this is hashed
    company_name: Optional[str] = None
    job_title: Optional[str] = None
    industry: Optional[str] = None
    country: Optional[str] = None
    phone_number: Optional[str] = None

    def to_collection_dict(self):
        return {k: (str(v) if isinstance(v, ObjectId) else v) for k, v in self.dict().items()}

@app.post("/signup/", status_code=status.HTTP_201_CREATED)
async def create_user(user: User):
    if await collection.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email already registered")

    new_user = await collection.insert_one(user.dict())
    created_user = await collection.find_one({"_id": new_user.inserted_id})

    # Convert ObjectId to str for JSON serialization
    created_user['_id'] = str(created_user['_id'])
    return created_user


@app.post("/login/")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    print("Login attempt:", form_data.username, form_data.password)  # Log credentials attempt
    user = await collection.find_one({"email": form_data.username})
    if not user:
        print("User not found for email:", form_data.username)
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if user['password'] != form_data.password:
        print("Incorrect password for email:", form_data.username)
        raise HTTPException(status_code=401, detail="Invalid email or password")

    print("Login successful for email:", form_data.username)
    return {"message": "Login successful"}


@app.post("/my_company/")
async def my_company(data: CompanyData):
    user = await collection.find_one({"email": data.email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Check if company data already exists
    existing_company = await company_collection.find_one({"email": data.email})
    if existing_company:
        # Update existing company data
        await company_collection.update_one({"email": data.email}, {"$set": data.dict()})
        return {"message": "Company data updated successfully"}
    else:
        # Insert new company data
        await company_collection.insert_one(data.dict())
        return {"message": "Company data created successfully"}



@app.get("/get_company/")
async def get_company(email: str):
    company_data = await company_collection.find_one({"email": email})
    if not company_data:
        raise HTTPException(status_code=404, detail="Company data not found")

    # Convert MongoDB document to a dictionary and ObjectId to string
    company_data_dict = {k: (str(v) if isinstance(v, ObjectId) else v) for k, v in company_data.items()}
    return JSONResponse(content=company_data_dict)


@app.post("/upload_documents/", status_code=status.HTTP_201_CREATED)
async def upload_documents(user_documents: UserDocument):
    user_documents_dict = jsonable_encoder(user_documents)
    
    # Check if a document entry already exists for the user
    existing_entry = await document_collection.find_one({"email": user_documents.email})
    if existing_entry:
        # Append new documents to the existing array
        await document_collection.update_one(
            {"email": user_documents.email},
            {"$push": {"documents": {"$each": user_documents_dict["documents"]}}}
        )
    else:
        # Create a new entry if none exists
        await document_collection.insert_one(user_documents_dict)
    
    return {"message": "Documents uploaded successfully"}

@app.post("/create_chatbot/", response_model=Chatbot)
async def create_chatbot(request: ChatbotCreationRequest):
    # Generate a unique ID for the chatbot
    chatbot_id = str(uuid.uuid4())

    # Create chatbot object
    new_chatbot = Chatbot(
        id=chatbot_id,
        email=request.email,
        type=request.type,
        goals=request.goals,
        description=request.description
    )

    # Store in the database
    await chatbot_collection.insert_one(new_chatbot.dict())

    return new_chatbot

scheduler = AsyncIOScheduler()
scheduler.start()

def delete_old_sessions():
    threshold = datetime.now() - timedelta(minutes=20)
    chat_session_collection.delete_many({"last_active": {"$lt": threshold}})

scheduler.add_job(delete_old_sessions, 'interval', minutes=5)



# Endpoint to get user documents
@app.get("/get_documents/{email}", response_model=UserDocument)
async def get_documents(email: str):
    document_data = await document_collection.find_one({"email": email})
    if document_data:
        return UserDocument(**document_data)
    else:
        raise HTTPException(status_code=404, detail="Documents not found")


@app.put("/update_document/")
async def update_document(email: str, document_title: str, updated_document: Document):
    # Fetch the user's document data
    user_data = await document_collection.find_one({"email": email})
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")

    # Update the specific document
    documents = user_data.get("documents", [])
    for doc in documents:
        if doc["title"] == document_title:
            doc.update(updated_document.dict())
            break
    else:
        raise HTTPException(status_code=404, detail="Document not found")

    await document_collection.update_one({"email": email}, {"$set": {"documents": documents}})
    return {"message": "Document updated successfully"}

@app.delete("/delete_document/")
async def delete_document(request: DeleteDocumentRequest):
    await document_collection.update_one(
        {"email": request.email},
        {"$pull": {"documents": {"title": request.document_title, "category": request.category}}}
    )
    return {"message": "Document deleted successfully"}




@app.post("/customer_analysis/")
async def financial_analysis(data: CustomerData):
    try:
        print("Received data:", data.data)
        analysis_prompt = "Process this, do calculations using metrics and ratios and write a good report with emojis for the following Customer data for the company and suggest any chnages and improvemnets that can be done you are part of the company as a CMO:\n\n" + data.data
        response = co.generate(prompt=analysis_prompt)  
        print(response.generations[0].text)
        return {"analysis": response.generations[0].text}
    except Exception as e:
        print("Error during analysis:", str(e))
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/financial_analysis/")
async def financial_analysis(data: FinancialData):
    try:
        print("Received data:", data.data)
        analysis_prompt = "Process this, do finanical calculations using metrics and ratios and write a good report with emojis for the following financial data for the company and suggest any chnages and improvemnets that can be done you are part of the company as a CFO:\n\n" + data.data
        response = co.generate(prompt=analysis_prompt)  
        print(response.generations[0].text)
        return {"analysis": response.generations[0].text}
    except Exception as e:
        print("Error during analysis:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get_user/")
async def get_user(email: str):
    user = await collection.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user['_id'] = str(user['_id'])  # Convert ObjectId to string for JSON serialization
    return user


@app.put("/update_user/")
async def update_user(user: User):
    updated_user = await collection.update_one({"email": user.email}, {"$set": user.dict()})
    if updated_user.matched_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "User updated successfully"}




@app.post("/document_analysis/")
async def document_analysis(request: DocumentAnalysisRequest):
    print(request)
    # Fetch user details if needed
    user_data = await collection.find_one({"email": request.email})
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")

    # Here, you can include logic to process the document text
    # For example, using Cohere API with the provided text and user data
    analysis_prompt = f"Summaries this document for me: {request.description}\n\nDocument Content:\n{request.text}\n\nUser's Company: {user_data.get('company_name', '')}"

    try:
        print("Now calling cohere")
        response = co.generate(prompt=analysis_prompt)
        print(response.generations[0].text)
        return {"analysis": response.generations[0].text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/feedback_analysis/")
async def feedback_analysis(request: FeedbackAnalysisRequest):

    # Fetch user details from the database
    user_data = await collection.find_one({"email": request.email})
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")

    # Include user details in the analysis
    user_details_prompt = f"Company Name: {user_data.get('company_name', '')}\n" \
                          f"Industry: {user_data.get('industry', '')}\n\n"

    analysis_prompt = "Process and write a good report with emojis for the following feedback data as the Chief Growth officer:\n\n" + request.data + " company details: "+ user_details_prompt

    try:
        response = co.generate(prompt=analysis_prompt)
       
        return {"analysis": response.generations[0].text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_blog_content/")
async def generate_blog_content(request: BlogContentRequest):
    user_data = await collection.find_one({"email": request.email})
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")

    blog_prompt = f"Generate a blog post about: {request.description} \nUser's Company: {user_data.get('company_name', '')}"
    response = co.generate(prompt=blog_prompt )
    return {"content": response.generations[0].text}

@app.post("/generate_social_media_content/")
async def generate_social_media_content(request: SocialMediaContentRequest):
    user_data = await collection.find_one({"email": request.email})
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")

    social_media_prompt = f"{request.prompt} \nPlatform: {request.type}. \nUser's Company: {user_data.get('company_name', '')} . Use make it trendy and relevant to our company and add appropriate emojis and hastags"
    response = co.chat(message=social_media_prompt,  connectors=[{"id": "web-search"}] )
    return {"content": response.text}

@app.post("/generate_email_content/")
async def generate_email_content(request: EmailContentRequest):
    user_data = await collection.find_one({"email": request.email})
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")

    email_prompt = f"Generate an email regarding: {request.description}\nDetails: {request.prompt}\nUser's Company: {user_data.get('company_name', '')}"
    response = co.generate(prompt=email_prompt )
    return {"content": response.generations[0].text}


@app.post("/legal_doc_generation/")
async def legal_doc_generation(request: LegalDocumentRequest):
    user_data = await collection.find_one({"email": request.email})
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")

    prompt = f"Generate a legal document based on: {request.description}. \nUser's Company: {user_data.get('company_name', '')}. Additional Info: {request.additionalInfo}"
    try:
        response = co.generate(prompt=prompt)
        return {"document": response.generations[0].text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/legal_doc_analysis/")
async def legal_doc_analysis(request: LegalDocumentAnalysisRequest):
    print(f"Received request: {request}")
    # Fetch user details if needed
    user_data = await collection.find_one({"email": request.email})
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")

    # Here, you can include logic to process the legal document text
    # For example, using Cohere API with the provided text and user data
    analysis_prompt = f"Legal document analysis:\n\n{request.text}\n\nUser's Company: {user_data.get('company_name', '')}\nDescription: {request.description}"

    try:
        response = co.generate(prompt=analysis_prompt )
        return {"analysis": response.generations[0].text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_user_chatbots/", response_model=List[Chatbot])
async def get_user_chatbots(email: str):
    # Query the database for chatbots associated with the given email
    chatbots_cursor = chatbot_collection.find({"email": email})
    chatbots = await chatbots_cursor.to_list(length=100)  # Adjust the length as needed

    # Convert to list of Chatbot objects
    return [Chatbot(**chatbot) for chatbot in chatbots]


@app.delete("/delete_chatbot/", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chatbot(request: DeleteChatbotRequest):
    delete_result = await chatbot_collection.delete_one({"id": request.id})

    if delete_result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Chatbot not found")

    return {"message": "Chatbot deleted successfully"}

@app.post("/start_chat_session/", response_model=ChatSession)
async def start_chat_session(request: StartChatSessionRequest):
    print(f"Starting chat session for chatbot_id: {request.chatbot_id}")
    session_id = str(uuid.uuid4())
    new_session = ChatSession(session_id=session_id, chatbot_id=request.chatbot_id, chat_history=[])
    await chat_session_collection.insert_one(new_session.dict())
    return new_session



@app.post("/send_chat_message/")
async def send_chat_message(message: ChatMessage):
    print("Received a new chat message request: ", message.text)

    # Find the chat session using the session_id from the ChatMessage model
    session = await chat_session_collection.find_one({"session_id": message.session_id})
    if not session:
        print("Chat session not found.")
        raise HTTPException(status_code=404, detail="Chat session not found")

    # Retrieve the chatbot using the chatbot_id from the session
    chatbot = await chatbot_collection.find_one({"id": session["chatbot_id"]})
    if not chatbot:
        print("Chatbot not found.")
        raise HTTPException(status_code=404, detail="Chatbot not found")

    # Extract chatbot details
    chatbot_type = chatbot.get("type", "Unknown type")
    chatbot_description = chatbot.get("description", "No description")
    chatbot_email = chatbot.get("email", "No email")
    chatbot_goals = chatbot.get("goals", "No goals")

    print(f"Chatbot type: {chatbot_type}")
    print(f"Chatbot description: {chatbot_description}")
    print(f"Chatbot email: {chatbot_email}")
    print(f"Chatbot goals: {chatbot_goals}")
    # Using Cohere to generate a response

     # Update chat history
    updated_chat_history = session.get("chat_history", [])
    updated_chat_history.append({
        "user_name": message.user_name,
        "text": message.text,
        "timestamp": message.timestamp
    })

    # Fetch user details using the chatbot_email
    user = await collection.find_one({"email": chatbot_email})
    if user:
        user_details = {
        "full_name": user.get("full_name", "No name"),
        "email": user.get("email", "No email"),
        "company_name": user.get("company_name", "No company name"),
        "job_title": user.get("job_title", "No job title"),
        "industry": user.get("industry", "No industry"),
        "country": user.get("country", "No country"),
        "phone_number": user.get("phone_number", "No phone number")
        }
        print("User details:", user_details)
    else:
        print("User not found for email:", chatbot_email)



     # Fetch company details including products using the chatbot_email
    company = await company_collection.find_one({"email": chatbot_email})
    if company:
        company_name = company.get("companyName", "No company name")
        company_description = company.get("companyDescription", "No description")
        products = company.get("products", [])
        print(f"Company: {company_name}, Description: {company_description}, Products: {products}")
    else:
        print("Company not found for email:", chatbot_email)
        products = []


    # Fetch and print documents associated with the email from the frontend
    # Fetch the document associated with the email
    user_document = await document_collection.find_one({"email": chatbot_email})
    if user_document:
        # Extract the documents array
        documents = user_document.get("documents", [])
        print("Extracted documents:", documents)
    else:
        print("No documents found for this email")
        documents = []

    try:
        enhanced_message = f"You are a {chatbot_type} for the company {user_details.get('company_name')},this is your goal as our representative {chatbot_goals}, our products are: {products} , so use proper sales techniques and  the user has asked: {message.text}"
        # You might want to prepend some context or previous messages if needed
        cohere_response = co.chat(message=enhanced_message, documents=documents)
        response_text = cohere_response.text

        updated_chat_history.append({
            "user_name": "Chatbot",
            "text": response_text,
            "timestamp": datetime.now().isoformat()
        })

        # Update the session in the database
        await chat_session_collection.update_one(
            {"session_id": message.session_id},
            {"$set": {"chat_history": updated_chat_history, "last_active": datetime.now()}}
        )
        print("Cohere generated response: ", response_text)
    except Exception as e:
        print(f"Error calling Cohere: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    # Return the response from Cohere
    return {"response": response_text}




@app.get("/")
async def root():
    return {"message": "Hello, World!"}
