from fastapi import FastAPI, Depends, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI, HTTPException
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy.orm import sessionmaker, declarative_base
from databases import Database
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Boolean, select, delete, update, func, and_, or_
import openai
from langchain import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
import json
import re
import os
import time
import tiktoken


DATABASE_URL = "sqlite:///./test7.db"
database = Database(DATABASE_URL)
metadata = MetaData()
Base = declarative_base()

#MAIN SITE
class BetaRequest(Base):
    __tablename__ = "beta_requests"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)  # Make sure email is unique

class BetaRequestData(BaseModel):
    email: str


# This is the function table
class Function(Base):
    __tablename__ = "functions"
    id = Column(Integer, primary_key=True, index=True)
    input = Column(String, index=True)
    output = Column(String)
    is_reviewed = Column(Boolean, default=False, index=True)
    llm_name = Column(String)
    llm_short_summary = Column(String)
    llm_step_by_step_description = Column(String)

class EvaluationLog(Base):
    __tablename__ = "evaluation_logs"
    id = Column(Integer, primary_key=True, index=True)
    request_data = Column(String)
    response_data = Column(String)

class Suggestion(Base):
    __tablename__ = "suggestions"
    id = Column(Integer, primary_key=True, index=True)
    sha256 = Column(String, index=True)
    offset = Column(String, index=True)
    code = Column(String)
    suggestion = Column(String)

class SuggestionData(BaseModel):
    sha256: str
    offset: str
    code: str
    suggestion: str

class ActionData(BaseModel):
    action_type: str
    function_id: int

class DataPoint(BaseModel):
    input: str
    output: str

class EvaluateLocalData(BaseModel):
    sha256: str
    function_offset: str
    code: str


openai_key = os.environ.get('OPENAI_KEY')
if not openai_key:
    print("OPENAI_KEY not set in environment variables.")
else:
    print("OPENAI_KEY retrieved successfully.")
    

app = FastAPI()

approved_dataset = []

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
llm4 = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=openai_key)
llm3516 = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k", openai_api_key=openai_key)
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=openai_key)

# Load the dataset
dataset = load_dataset("dyngnosis/function_names_v2", split="train")

# Load tokenizer and model
output_dir = "/home/gpu/instruct-decode-llama-4096-34b/checkpoint-10000"
base_model = "codellama/CodeLlama-34b-Instruct-hf"
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-34b-Instruct-hf")

print("loading peft model")
from peft import PeftModel
model = PeftModel.from_pretrained(model, output_dir)
model.eval()


engine = create_engine(DATABASE_URL)
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(bind=engine)



@app.on_event("startup")
async def startup():
    await database.connect()
    load_data_to_db() 

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

#sometimes OpenAI json responses  for step-by-step instructions are formatted as a list of strings instead of a single string.  This function flattens the list of strings into a single string.
def flatten_list(lst):
    """Flatten a potentially nested list or dictionary."""
    for item in lst:
        if isinstance(item, list):
            yield from flatten_list(item)
        elif isinstance(item, dict):
            yield from item.values()  # or item.items() if you want key-value pairs
        else:
            yield item


# Define a function to get llm_name with retries
def get_llm_name_with_retry(code):
    max_retries = 3  # Define the maximum number of retries
    retries = 0
    while retries < max_retries:
        prompt = f"The following code was found in a malware sample.  Provide only JSON response in your response.  Ensure the JSON is properly formatted and control characters are escaped.  It must contain 'short_summary', 'step_by_step_description', and 'new_function_name' that provides a long descriptive name that describes the purpose of the function. The step-by-step description must use Markdown syntax.  Identify any constants like windows error codes, HRESULT, and encryption constants.  Here is the code:\r\n {code}"
        #use tiktoken to get the token count of prompt
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        print(prompt)

        token_count = len(encoding.encode(prompt))
        if token_count > 4096:
            llm_name = llm3516.predict(prompt)
        else:
            llm_name = llm.predict(prompt)
        print(llm_name)
        try:
            response_json = json.loads(llm_name)
            return response_json  # Return response_json if successfully obtained
        except json.decoder.JSONDecodeError:
            corrected_data = llm_name.replace("\\", "\\\\")
            print(llm_name)
            try:
                response_json = json.loads(corrected_data)
                print("[!] WARNING: Had to fix escapes")
                return response_json  # Return response_json if successfully obtained
            except:
                print("--------------", llm_name, "----------------")
        
        # Increment the retry count and wait for a moment before retrying
        retries += 1
        time.sleep(1)  # Wait for 1 second before retrying
    
    # If maximum retries are reached without success, return None
    return None

def generate_decode_response(code):
    full_prompt = f"""<s>[INST]<<SYS>> You are an advanced malware reverse engineer capable of understanding decompiled C code and identifying malicious functionality.  Be sure to mention any constants related to encryption algoritms Windows error codes and HRESULT codes.  Be descriptive about what API calls do.<</SYS>>

You must output a descriptive. You should use Markdown syntax. ### Function Summary that describes the following decompiled code followed by a descriptive ### New Function Name 

Do not provide any extra information in the ### New Function Name. Only provide the name of the function. Do not include any extra information.


[/INST]

### Code:
{code}

### Function Summary:
"""

    #we only want a few tokens from the model but we need to add the prompt length to the max_new_tokens
    max_new_tokens = 4096-len(tokenizer.encode(full_prompt))
    print("max_new_tokens: ", max_new_tokens)
    model_input = tokenizer(full_prompt, return_tensors="pt", max_length=4000, truncation=True).to("cuda")
    model.eval()
    with torch.no_grad():
        #top_k = 50 # Top-k sampling
        #top_p = 0.95 # Nucleus sampling
        #repetition_penalty = 1.8
        #repetition_penalty_sustain
        #token_repetition_penalty_decay
        try:        
            response = tokenizer.decode(\
                model.generate(\
                    **model_input,\
                    max_new_tokens=1500)[0],\
                    skip_special_tokens=True,\
                    repetition_penalty=1.8,\
                    temperature=0)
        except torch.cuda.CudaError:
            print("Out of memory error. Telling user to try a shorter function.")
            return {}
            
        try:
            extract_function_info(response)
        except:
            print("--------------", response, "----------------")
    return response

#This is for extracting the function summary and new function name from the decode-llm response
def extract_function_info(content: str):
    """
    Extract the Function Summary and New Function Name from the provided content.

    Args:
    - content (str): The content string containing the Function Summary and New Function Name.

    Returns:
    - dict: A dictionary containing the extracted Function Summary and New Function Name.
    """
    
    # Regex pattern to capture Function Summary and New Function Name
    print("content -----------------------------------\r\n", content)
    summary_pattern = r"### Function Summary:\n+(.*?)\n+### New Function Name:"
    name_pattern = r"### New Function Name:\n+(.*?)(?:\n|$)"
    # Extracting the content
    summary_match = re.search(summary_pattern, content, re.DOTALL)
    name_match = re.search(name_pattern, content, re.DOTALL)
    print("summary_match, name_match")
    print(summary_match, name_match)

    function_info = {
        "Function Summary": summary_match.group(1) if summary_match else None,
        "New Function Name": name_match.group(1) if name_match else None
    }

    return function_info

# Modify llm_rename_function to use get_llm_name_with_retry
def llm_rename_function(code):
    response_json = get_llm_name_with_retry(code)

    if response_json is None:
        return {
            "llm_name": "LLM_ERROR",
            "llm_short_summary": "LLM_ERROR",
            "llm_step_by_step_description": "LLM_ERROR"
        }

    # Handle the "step_by_step_description" if it's a list
    step_description = response_json.get("step_by_step_description", "LLM_ERROR")
    if isinstance(step_description, list):
        step_description = "\n".join(flatten_list(step_description))

    analysis_results = {
        "llm_name": response_json.get("new_function_name", "LLM_ERROR"),
        "llm_short_summary": response_json.get("short_summary", "LLM_ERROR"),
        "llm_step_by_step_description": step_description
    }
    
    return analysis_results

@app.post('/beta')
async def request_beta_key(data: BetaRequestData):
    # Check if email already exists
    query = select(BetaRequest).where(BetaRequest.email == data.email)
    existing_email = await database.fetch_one(query)
    
    if existing_email:
        raise HTTPException(status_code=400, detail="Email already registered for beta.")

    # Add email to the beta_requests table
    beta_request_entry = BetaRequest(email=data.email)
    db = SessionLocal()
    db.add(beta_request_entry)
    db.commit()
    db.close()

    return {"status": "success", "message": "Beta key request registered successfully."}

@app.get('/')
async def index():
    return templates.TemplateResponse("index.html", {"request": {}})

@app.get('/sample')
async def sample():
    # Return a single sample from the dataset for demonstration purposes
    return dataset[0] 

@app.post('/evaluate')
async def evaluate(data: EvaluateLocalData):
    response_json = llm_rename_function(data.code)
    analysis_results = {
        "llm_name": response_json.get("llm_name", "LLM_ERROR"),
        "llm_short_summary": response_json.get("llm_short_summary", "LLM_ERROR"),
        "llm_step_by_step_description": response_json.get("llm_step_by_step_description", "LLM_ERROR"),
    }

    # Store the request and response data in the EvaluationLog table
    log = EvaluationLog(request_data=json.dumps(data.dict()), response_data=json.dumps(analysis_results))
    db = SessionLocal()
    db.add(log)
    db.commit()
    db.close()
    return analysis_results    

@app.post('/evaluate_local')
async def evaluate_local(data: EvaluateLocalData):
    # Get local llm response
    response = generate_decode_response(data.code)
    extracted_data = extract_function_info(response)
    analysis_results = {
        'llm_name': extracted_data["New Function Name"],
        'llm_short_summary': extracted_data["Function Summary"],
        'llm_step_by_step_description': response
    }

    # Store the request and response data in the EvaluationLog table
    log = EvaluationLog(request_data=json.dumps(data.dict()), response_data=json.dumps(analysis_results))
    db = SessionLocal()
    db.add(log)
    db.commit()
    db.close()
    return analysis_results

@app.get('/review')
async def review(category: str = "all"):
    category_dict = {
    "enumeration": ["CreateToolhelp32Snapshot","EnumDeviceDrivers","EnumProcesses","EnumProcessModules","EnumProcessModulesEx","FindFirstFileA","FindNextFileA","GetLogicalProcessorInformtion","GetLogicalProcessorInformationEx","GetModuleBaseNameA","GetSystemDefaultLangId","GetVersionExA","GetWindowsDirectoryA","IsWoW64Process","Module32First","Module32Next","Process32First","Process32Next","ReadProcessMemory","Thread32First","Thread32Next","GetSystemDirectoryA","GetSystemTime","ReadFile","GetComputerNameA","Vi","tualQueryEx","GetProcessIdOfThread","GetProcessId","GetCurrentThread","GetCurrentThreadId","GetThreadId","GetThreadInformation","GetCurrentProcess","GetCurrentProcessId","SearchPathA","GetFileTime","GetFileAttributesA","LookupPrivilegeValueA","LookupAccountNameA","GetCurrentHwProfileA","GetUserNameA","RegEnumKeyExA","RegEnumValueA","RegQueryInfoKeyA","RegQueryMultipleValuesA","RegQueryValueExA","NtQueryDirectoryFile","NtQueryInformationProcess","NtQuerySystemEnvironmentValueEx","EnumDesktop","EnumWindows","NetShareEnum","NetShareGetInfo","NetShareCheck","GetAdaptersInfo","PathFileExistsA","GetNativeSystemInfo","RtlGetVersion","GetIpNetTable","GetLogicalDrives","GetDriveTypeA","RegEnumKeyA","WNetEnumResourceA","WNetCloseEnum","FindFirstUrlCacheEntryA","FindNextUrlCacheEntryA","WNetAddConnection2A","WNetAddConnectionA","EnumResourceTypesA","EnumResourceTypesExA","GetSystemTimeAsFileTime","GetThreadLocale","EnumSystemLocalesA"],
    "injection" : ["CreateFileMappingA","CreateProcessA","CreateRemoteThread","CreateRemoteThreadEx","GetModuleHandleA","GetProcAddress","GetThreadContext","HeapCreate","LoadLibraryA","LoadLibraryExA","LocalAlloc","MapViewOfFile","MapViewOfFile2","MapViewOfFile3","MapViewOfFileEx","OpenThread","Process32First","Process32Next","QueueUserAPC","ReadProcessMemory","ResumeThread","SetProcessDEPPolicy","SetThreadContext","SuspendThread","Thread32First","Thread32Next","Toolhelp32ReadProcessMemory","VirtualAlloc","VirtualAllocEx","VirtualProtect","VirtualProtectEx","WriteProcessMemory","VirtualAllocExNuma","VirtualAlloc2","VirtualAlloc2FromApp","VirtualAllocFromApp","VirtualProtectFromApp","CreateThread","WaitForSingleObject","OpenProcess","OpenFileMappingA","GetProcessHeap","GetProcessHeaps","HeapAlloc","HeapReAlloc","GlobalAlloc","AdjustTokenPrivileges","CreateProcessAsUserA","OpenProcessToken","CreateProcessWithTokenW","NtAdjustPrivilegesToken","NtAllocateVirtualMemory","NtContinue","NtCreateProcess","NtCreateProcessEx","NtCreateSection","NtCreateThread","NtCreateThreadEx","NtCreateUserProcess","NtDuplicateObject","NtMapViewOfSection","NtOpenProcess","NtOpenThread","NtProtectVirtualMemory","NtQueueApcThread","NtQueueApcThreadEx","NtQueueApcThreadEx2","NtReadVirtualMemory","NtResumeThread","NtUnmapViewOfSection","NtWaitForMultipleObjects","NtWaitForSingleObject","NtWriteVirtualMemory","RtlCreateHeap","LdrLoadDll","RtlMoveMemory","RtlCopyMemory","SetPropA","WaitForSingleObjectEx","WaitForMultipleObjects","WaitForMultipleObjectsEx","KeInsertQueueApc","Wow64SetThreadContext","NtSuspendProcess","NtResumeProcess","DuplicateToken","NtReadVirtualMemoryEx","CreateProcessInternal","EnumSystemLocalesA","UuidFromStringA"],
    "evasion" : ["CreateFileMappingA","DeleteFileA","GetModuleHandleA","GetProcAddress","LoadLibraryA","LoadLibraryExA","LoadResource","SetEnvironmentVariableA","SetFileTime","Sleep","WaitForSingleObject","SetFileAttributesA","SleepEx","NtDelayExecution","NtWaitForMultipleObjects","NtWaitForSingleObject","CreateWindowExA","RegisterHotKey","timeSetEvent","IcmpSendEcho","WaitForSingleObjectEx","WaitForMultipleObjects","WaitForMultipleObjectsEx","SetWaitableTimer","CreateTimerQueueTimer","CreateWaitableTimer","SetWaitableTimer","SetTimer","Select","ImpersonateLoggedOnUser","SetThreadToken","DuplicateToken","SizeOfResource","LockResource","CreateProcessInternal","TimeGetTime","EnumSystemLocalesA","UuidFromStringA"],
    "spying" : ["AttachThreadInput","CallNextHookEx","GetAsyncKeyState","GetClipboardData","GetDC","GetDCEx","GetForegroundWindow","GetKeyboardState","GetKeyState","GetMessageA","GetRawInputData","GetWindowDC","MapVirtualKeyA","MapVirtualKeyExA","PeekMessageA","PostMessageA","PostThreadMessageA","RegisterHotKey","RegisterRawInputDevices","SendMessageA","SendMessageCallbackA","SendMessageTimeoutA","SendNotifyMessageA","SetWindowsHookExA","SetWinEventHook","UnhookWindowsHookEx","BitBlt","StretchBlt","GetKeynameTextA"],
    "internet" : ["WinExec","FtpPutFileA","HttpOpenRequestA","HttpSendRequestA","HttpSendRequestExA","InternetCloseHandle","InternetOpenA","InternetOpenUrlA","InternetReadFile","InternetReadFileExA","InternetWriteFile","URLDownloadToFile","URLDownloadToCacheFile","URLOpenBlockingStream","URLOpenStream","Accept","Bind","Connect","Gethostbyname","Inet_addr","Recv","Send","WSAStartup","Gethostname","Socket","WSACleanup","Listen","ShellExecuteA","ShellExecuteExA","DnsQuery_A","DnsQueryEx","WNetOpenEnumA","FindFirstUrlCacheEntryA","FindNextUrlCacheEntryA","InternetConnectA","InternetSetOptionA","WSASocketA","Closesocket","WSAIoctl","ioctlsocket","HttpAddRequestHeaders"],
    "anti_debugging" : ["CreateToolhelp32Snapshot","GetLogicalProcessorInformation","GetLogicalProcessorInformationEx","GetTickCount","OutputDebugStringA","CheckRemoteDebuggerPresent","Sleep","GetSystemTime","GetComputerNameA","SleepEx","IsDebuggerPresent","GetUserNameA","NtQueryInformationProcess","ExitWindowsEx","FindWindowA","FindWindowExA","GetForegroundWindow","GetTickCount64","QueryPerformanceFrequency","QueryPerformanceCounter","GetNativeSystemInfo","RtlGetVersion","GetSystemTimeAsFileTime","CountClipboardFormats"],
    "ransomware" :["CryptAcquireContextA","EncryptFileA","CryptEncrypt","CryptDecrypt","CryptCreateHash","CryptHashData","CryptDeriveKey","CryptSetKeyParam","CryptGetHashParam","CryptSetKeyParam","CryptDestroyKey","CryptGenRandom","DecryptFileA","FlushEfsCache","GetLogicalDrives","GetDriveTypeA","CryptStringToBinary","CryptBinaryToString","CryptReleaseContext","CryptDestroyHash","EnumSystemLocalesA"],
    "helper" : ["ConnectNamedPipe","CopyFileA","CreateFileA","CreateMutexA","CreateMutexExA","DeviceIoControl","FindResourceA","FindResourceExA","GetModuleBaseNameA","GetModuleFileNameA","GetModuleFileNameExA","GetTempPathA","IsWoW64Process","MoveFileA","MoveFileExA","PeekNamedPipe","WriteFile","TerminateThread","CopyFile2","CopyFileExA","CreateFile2","GetTempFileNameA","TerminateProcess","SetCurrentDirectory","FindClose","SetThreadPriority","UnmapViewOfFile","ControlService","ControlServiceExA","CreateServiceA","DeleteService","OpenSCManagerA","OpenServiceA","RegOpenKeyA","RegOpenKeyExA","StartServiceA","StartServiceCtrlDispatcherA","RegCreateKeyExA","RegCreateKeyA","RegSetValueExA","RegSetKeyValueA","RegDeleteValueA","RegOpenKeyExA","RegEnumKeyExA","RegEnumValueA","RegGetValueA","RegFlushKey","RegGetKeySecurity","RegLoadKeyA","RegLoadMUIStringA","RegOpenCurrentUser","RegOpenKeyTransactedA","RegOpenUserClassesRoot","RegOverridePredefKey","RegReplaceKeyA","RegRestoreKeyA","RegSaveKeyA","RegSaveKeyExA","RegSetKeySecurity","RegUnLoadKeyA","RegConnectRegistryA","RegCopyTreeA","RegCreateKeyTransactedA","RegDeleteKeyA","RegDeleteKeyExA","RegDeleteKeyTransactedA","RegDeleteKeyValueA","RegDeleteTreeA","RegDeleteValueA","RegCloseKey","NtClose","NtCreateFile","NtDeleteKey","NtDeleteValueKey","NtMakeTemporaryObject","NtSetContextThread","NtSetInformationProcess","NtSetInformationThread","NtSetSystemEnvironmentValueEx","NtSetValueKey","NtShutdownSystem","NtTerminateProcess","NtTerminateThread","RtlSetProcessIsCritical","DrawTextExA","GetDesktopWindow","SetClipboardData","SetWindowLongA","SetWindowLongPtrA","OpenClipboard","SetForegroundWindow","BringWindowToTop","SetFocus","ShowWindow","NetShareSetInfo","NetShareAdd","NtQueryTimer","GetIpNetTable","GetLogicalDrives","GetDriveTypeA","CreatePipe","RegEnumKeyA","WNetOpenEnumA","WNetEnumResourceA","WNetAddConnection2A","CallWindowProcA","NtResumeProcess","lstrcatA","ImpersonateLoggedOnUser","SetThreadToken","SizeOfResource","LockResource","UuidFromStringA"]
    }
    # If the user selects "all", combine all lists
    if category == "all":
        conditions = [Function.input.contains(string) for sublist in category_dict.values() for string in sublist]
    else:
        conditions = [Function.input.contains(string) for string in category_dict.get(category, [])]

    count_query = select(func.count()).where(
        and_(
            Function.is_reviewed == False, 
            Function.llm_short_summary != '',
            or_(*conditions)
        )
    )
    unreviewed_count = await database.fetch_val(count_query)
    query = select(Function).where(
        and_(
            Function.is_reviewed == False, 
            Function.llm_short_summary != '',
            or_(*conditions)
        )
    ).limit(1)
    result = await database.fetch_one(query)

    if not result:
        return {"message": "No more items to review"}

    if result:
        # send the source code to hf model
        code_to_analyze = result.input
        #llm_response_str = generate_decode_response(code_to_analyze)
        #print(extract_function_info(llm_response_str))

        function_name = result.output
        if result.llm_step_by_step_description is not None:
            analysis_results = {
                "llm_name": result.llm_name,
                "llm_short_summary": result.llm_short_summary,
                "llm_step_by_step_description": result.llm_step_by_step_description
            }
        else:
            analysis_results = llm_rename_function(code_to_analyze)

        # Store the new analysis results in the database
            query = (update(Function)
                        .where(Function.id == result.id)
                        .values(
                            llm_name=analysis_results.get("llm_new_function_name", "LLM_ERROR"),
                            llm_short_summary=analysis_results.get("llm_short_summary", "LLM_ERROR"),
                            llm_step_by_step_description=analysis_results.get("llm_step_by_step_description", "LLM_ERROR")
                        ))
            await database.execute(query)

    return templates.TemplateResponse(
        "review.html",
        {
            "request": {}, 
            "data_item": result, 
            "unreviewed_count": unreviewed_count,
            "selected_category": category,
            "analysis_results": analysis_results
        }
    )

class UpdateOutputData(BaseModel):
    function_id: int
    new_output: str
    new_short_summary: str  
    new_step_by_step_description: str  

@app.post('/update-output')
async def update_output(data: UpdateOutputData):
    query = (update(Function).where(Function.id == data.function_id)
                             .values(output=data.new_output, \
                                     llm_short_summary=data.new_short_summary,\
                                     llm_step_by_step_description=data.new_step_by_step_description))
    await database.execute(query)
    return {"status": "success"}

@app.post('/suggest')
async def suggest(data: SuggestionData):
    # Create a new Suggestion entry
    suggestion_entry = Suggestion(
        sha256=data.sha256,
        offset=data.offset,
        code=data.code,
        suggestion=data.suggestion
    )
    
    db = SessionLocal()
    db.add(suggestion_entry)
    db.commit()
    db.close()

    return {"status": "success", "message": "Suggestion added successfully."}

@app.post('/action')
async def action(data: ActionData):
    if data.action_type == "approve":
        print("approve")
        # Mark the item as reviewed
        query = (update(Function).where(Function.id == data.function_id)
                                 .values(is_reviewed=True))
        await database.execute(query)
    elif data.action_type == "remove":
        # Delete the function from the database
        query = delete(Function).where(Function.id == data.function_id)
        await database.execute(query)
    else:
        raise HTTPException(status_code=400, detail="Invalid action type")
    return {"status": "success"}


def load_data_to_db():
    db = SessionLocal()
    if not db.query(Function).first():
        for item in dataset:
            function = Function(input=item['input'], output=item['output'], is_reviewed=False)
            db.add(function)
        db.commit()
    db.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
