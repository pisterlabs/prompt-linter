
import os
import openai
from langchain.chat_models import ChatOpenAI
import time
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
import subprocess
import json
import logging
import hashlib
import shutil
results = {}


chat = ChatOpenAI(temperature=0)
MODEL = "gpt-3.5-turbo"
def compile_code(code, compiler_options=''):
    """
    This function accepts valid C code and attempts to compile it using the x86_64-w64-mingw32-gcc compiler.
    If the code compiles successfully, the function returns the stdout and stderr of the compilation process.
    """
    # Write the C source code to a file
    filename = 'temp.c'
    with open(filename, 'w') as file:
        file.write(code)
    

    # Prepare the compiler command
    command = ['x86_64-w64-mingw32-gcc', '-o', 'output_file.exe', filename]
    #open output_file.exe and calculate hash with hashlib
    if compiler_options:
        #add the options to the command
        command += compiler_options.split(' ')
    print(command)
    # Compile the code
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Print the result
    print('stdout:', result.stdout.decode())
    print('stderr:', result.stderr.decode())
    try:
        data = open('output_file.exe', 'rb').read()
        hash = hashlib.sha256(data).hexdigest()
        subprocess.run(['mv', 'output_file.exe', f'{hash}.exe'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #wr
    except Exception as e:
        print('failed to compile', e)

def get_alternates_convo(code):
    # Step 1: send the conversation and available functions to GPT
    messages=[
        {"role": "system", "content": "You are a C programming assistant designed to create new, diverse implementations of supplied code.  There is no restriction on how you implement the functionality, only that you may not use exactly the same windows API calls.  You must always test your code with a compiler."},
        {"role": "user", "content": f"Please provide an alternatie implementation on the following code and make sure it compiles:\r\n{code}"}
    ]    
    functions = [
        {
            "name": "compile_code",
            "description": "This function will test to make sure code provided compiles source code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "C source code to compile",
                    },
                    "compiler_options": {
                        "type": "string",
                        "description": "arguments to pass the compiler",
                    } 
                },
                "required": ["code"],
            },
        }
    ]
    response = openai.ChatCompletion.create(
        #model="gpt-3.5-turbo-0613",
        model="gpt-4-0613",
        messages=messages,
        functions=functions,
        function_call="auto",  # auto is default, but we'll be explicit
    )
    response_message = response["choices"][0]["message"]
    print(response_message)

    # Step 2: check if GPT wanted to call a function
    if response_message.get("function_call"):
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "compile_code": compile_code,
        }  # only one function in this example, but you can have multiple
        function_name = response_message["function_call"]["name"]
        fuction_to_call = available_functions[function_name]
        try:
            function_args = json.loads(response_message["function_call"]["arguments"])
        except:
            # return fail
            return
        try:
            compiler_options = function_args.get("compiler_options")
        except:
            compiler_options = ''
        function_response = fuction_to_call(
            code=function_args.get("code"),
            compiler_options=compiler_options,
        )
    else:
        logging.error("GPT did not request a function call")
    



# import os
# import shutil
# import json
# import hashlib
# import subprocess

def cl_compile_ttp_code(code, api, action, compiler_options=''):
    """
    This function accepts valid C code and attempts to compile it using the cl.exe compiler.
    If the code compiles successfully, the function returns the stdout and stderr of the compilation process.
    """
    results = {}

    # Load existing results if available
    if os.path.exists('cl_results.json'):
        with open('.json', 'r') as file:
            results = json.load(file)
    
    # Write the C source code to a file
    filename = 'temp.c'
    with open(filename, 'w') as file:
        file.write(code)

    # Prepare the compiler command
    command = ['cl', '/Feoutput_file.exe', filename]

    if compiler_options:
        # add the options to the command
        command += compiler_options.split(' ')

    # Compile the code
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Print the result
    print('stdout:', result.stdout.decode())
    print('stderr:', result.stderr.decode())

    output_filename = 'output_file.exe'
    
    # Check if compilation was successful
    if os.path.exists(output_filename):
        # Open output_file.exe and calculate hash with hashlib
        with open(output_filename, 'rb') as file:
            data = file.read()
        hash = hashlib.sha256(data).hexdigest()
        
        new_result = {
            'code': code,
            'compiler_options': compiler_options,
            'stdout': result.stdout.decode(),
            'stderr': result.stderr.decode(),
            'output_file': f'output/{action}_{api}_{hash}.exe'
        }
        
        results[new_result['output_file']] = new_result

        # Rename the output file
        shutil.move(output_filename, new_result['output_file'])
    else:
        print('Compilation failed')

    # Save results to file
    with open('cl_results.json', 'w') as file:
        json.dump(results, file, indent=4)

def get_ttp_code_for_api(api, action):
    # Step 1: send the conversation and available functions to GPT
    
    ip_address = "172.105.16.201"
    port = "9000"
    domain_name = "apt.watch"
    
    sp_content = "You are a C programming assistant designed to create working C code that "
    sp_content +="produces specific funtionality using a specific windows API. You must always "
    sp_content +="test your code with a compiler and it must compile with x86_64-w64-mingw32-gcc. "
    sp_content +="Comment the code and print debug messages to help the user understand how "
    sp_content +="the code works. Try to make the code as unique and different as possible. "
    sp_content +="When generating code that demonstrates connection to a server use the "
    sp_content +=f"IP Address: {ip_address} and Port: {port}.  When contacting a domain name use "
    sp_content +=f"Domain name: {domain_name}.  When sending data to a server send *.log files. "


    
    
    messages=[
        {"role": "system", "content": f"{sp_content}"},
        {"role": "user", "content": f"Please provide example code for how an attacker might abuse the api \"{api}\" to provide \"{action}\" functionality. and test it with a compiler.  You will need to pass compiler flags like -lpsapi -lwininet -lws2_32 in some cases"}
    ]
    functions = [
        {
            "name": "compile_ttp_code",
            "description": "This function will test to make sure code provided compiles.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "C source code to compile",
                    },
                    "action": {
                        "type": "string",
                        "description": "functionmality provided",
                    },
                    "api": {
                        "type": "string",
                        "description": "the Function name being abused",
                    },
                    "compiler_options": {
                        "type": "string",
                        "description": "arguments to pass the compiler",
                    }
                },
                "required": ["code", "action", "api", "compiler_options"],
            },
        }
    ]
    response = openai.ChatCompletion.create(
        #model="gpt-3.5-turbo-0613",
        model="gpt-4-0613",
        messages=messages,
        functions=functions,
        function_call="auto",  # auto is default, but we'll be explicit
    )
    response_message = response["choices"][0]["message"]
    print(response_message)

    # Step 2: check if GPT wanted to call a function
    if response_message.get("function_call"):
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "compile_ttp_code": compile_ttp_code,
        }  # only one function in this example, but you can have multiple
        function_name = response_message["function_call"]["name"]
        fuction_to_call = available_functions[function_name]
        try:
            function_args = json.loads(response_message["function_call"]["arguments"])
        except:
            # return fail
            return

        try:
            api = function_args.get("api")
            action = function_args.get("action")
            compiler_options = function_args.get("compiler_options")
        except:
            return
        function_response = fuction_to_call(
            code=function_args.get("code"),
            api=api,
            action=action,
            compiler_options=compiler_options
        )
    else:
        print("No function call detected")
    print(function_response)
    return function_response

def get_ming_ttp_code_for_api(api, action):
    # Step 1: send the conversation and available functions to GPT
    messages=[
        {"role": "system", "content": "You are a C programming assistant designed to create working C code that produces specific funtionality using a specific windows API. You must always test your code with a compiler and it must compile with x86_64-w64-mingw32-gcc.  Comment the code and print debug messages to help the user understand how the code works. Try to make the code as unique and different as possible. When generating code that demonstrates connection to a server use 172.105.16.201 running on TCP port 9000.  Do not use google.com or example.com, use the domain name \"apt.watch\""},
        {"role": "user", "content": f"Please provide example code for how an attacker might abuse the api \"{api}\" to provide \"{action}\" functionality. and test it with a compiler.  You will need to pass compiler flags like -lpsapi -lwininet -lws2_32 in some cases"}
    ]    
    functions = [
        {
            "name": "compile_ttp_code",
            "description": "This function will test to make sure code provided compiles.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "C source code to compile",
                    },
                    "action": {
                        "type": "string",
                        "description": "functionmality provided",
                    },
                    "api": {
                        "type": "string",
                        "description": "the Function name being abused",
                    },
                    "compiler_options": {
                        "type": "string",
                        "description": "arguments to pass the compiler",
                    }                    
                },
                "required": ["code", "action", "api", "compiler_options"],
            },
        }
    ]
    response = openai.ChatCompletion.create(
        #model="gpt-3.5-turbo-0613",
        model="gpt-4-0613",
        messages=messages,
        functions=functions,
        function_call="auto",  # auto is default, but we'll be explicit
    )
    response_message = response["choices"][0]["message"]
    print(response_message)

    # Step 2: check if GPT wanted to call a function
    if response_message.get("function_call"):
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "compile_ttp_code": compile_ttp_code,
        }  # only one function in this example, but you can have multiple
        function_name = response_message["function_call"]["name"]
        fuction_to_call = available_functions[function_name]
        try:
            function_args = json.loads(response_message["function_call"]["arguments"])
        except:
            # return fail
            return
        
        try:
            api = function_args.get("api")
            action = function_args.get("action")
            compiler_options = function_args.get("compiler_options")    
        except:
            return
        function_response = fuction_to_call(
            code=function_args.get("code"),
            api=api,
            action=action,
            compiler_options=compiler_options
        )
    else:
        print("No function call detected")
    print(function_response)
    return function_response


enumeration = ["CreateToolhelp32Snapshot","EnumDeviceDrivers","EnumProcesses","EnumProcessModules","EnumProcessModulesEx","FindFirstFileA","FindNextFileA","GetLogicalProcessorInform","tion","GetLogicalProcessorInformationEx","GetModuleBaseNameA","GetSystemDefaultLangId","GetVersionExA","GetWindowsDirectoryA","IsWoW64Process","Module32First","Module32Next","Process32First","Process32Next","ReadProcessMemory","Thread32First","Thread32Next","GetSystemDirectoryA","GetSystemTime","ReadFile","GetComputerNameA","Vi","tualQueryEx","GetProcessIdOfThread","GetProcessId","GetCurrentThread","GetCurrentThreadId","GetThreadId","GetThreadInformation","GetCurrentProcess","GetCurrentProcessId","SearchPathA","GetFileTime","GetFileAttributesA","LookupPrivilegeValueA","LookupAccountNameA","GetCurrentHwProfileA","GetUserNameA","RegEnumKeyExA","RegEnumValueA","RegQueryInfoKeyA","RegQueryMultipleValuesA","RegQueryValueExA","NtQueryDirectoryFile","NtQueryInformationProcess","NtQuerySystemEnvironmentValueEx","EnumDesktop","EnumWindows","NetShareEnum","NetShareGetInfo","NetShareCheck","GetAdaptersInfo","PathFileExistsA","GetNativeSystemInfo","RtlGetVersion","GetIpNetTable","GetLogicalDrives","GetDriveTypeA","RegEnumKeyA","WNetEnumResourceA","WNetCloseEnum","FindFirstUrlCacheEntryA","FindNextUrlCacheEntryA","WNetAddConnection2A","WNetAddConnectionA","EnumResourceTypesA","EnumResourceTypesExA","GetSystemTimeAsFileTime","GetThreadLocale","EnumSystemLocalesA"]
injection = ["CreateFileMappingA","CreateProcessA","CreateRemoteThread","CreateRemoteThreadEx","GetModuleHandleA","GetProcAddress","GetThreadContext","HeapCreate","LoadLibraryA","LoadLibraryExA","LocalAlloc","MapViewOfFile","MapViewOfFile2","MapViewOfFile3","MapViewOfFileEx","OpenThread","Process32First","Process32Next","QueueUserAPC","ReadProcessMemory","ResumeThread","SetProcessDEPPolicy","SetThreadContext","SuspendThread","Thread32First","Thread32Next","Toolhelp32ReadProcessMemory","VirtualAlloc","VirtualAllocEx","VirtualProtect","VirtualProtectEx","WriteProcessMemory","VirtualAllocExNuma","VirtualAlloc2","VirtualAlloc2FromApp","VirtualAllocFromApp","VirtualProtectFromApp","CreateThread","WaitForSingleObject","OpenProcess","OpenFileMappingA","GetProcessHeap","GetProcessHeaps","HeapAlloc","HeapReAlloc","GlobalAlloc","AdjustTokenPrivileges","CreateProcessAsUserA","OpenProcessToken","CreateProcessWithTokenW","NtAdjustPrivilegesToken","NtAllocateVirtualMemory","NtContinue","NtCreateProcess","NtCreateProcessEx","NtCreateSection","NtCreateThread","NtCreateThreadEx","NtCreateUserProcess","NtDuplicateObject","NtMapViewOfSection","NtOpenProcess","NtOpenThread","NtProtectVirtualMemory","NtQueueApcThread","NtQueueApcThreadEx","NtQueueApcThreadEx2","NtReadVirtualMemory","NtResumeThread","NtUnmapViewOfSection","NtWaitForMultipleObjects","NtWaitForSingleObject","NtWriteVirtualMemory","RtlCreateHeap","LdrLoadDll","RtlMoveMemory","RtlCopyMemory","SetPropA","WaitForSingleObjectEx","WaitForMultipleObjects","WaitForMultipleObjectsEx","KeInsertQueueApc","Wow64SetThreadContext","NtSuspendProcess","NtResumeProcess","DuplicateToken","NtReadVirtualMemoryEx","CreateProcessInternal","EnumSystemLocalesA","UuidFromStringA"]
evasion = ["CreateFileMappingA","DeleteFileA","GetModuleHandleA","GetProcAddress","LoadLibraryA","LoadLibraryExA","LoadResource","SetEnvironmentVariableA","SetFileTime","Sleep","WaitForSingleObject","SetFileAttributesA","SleepEx","NtDelayExecution","NtWaitForMultipleObjects","NtWaitForSingleObject","CreateWindowExA","RegisterHotKey","timeSetEvent","IcmpSendEcho","WaitForSingleObjectEx","WaitForMultipleObjects","WaitForMultipleObjectsEx","SetWaitableTimer","CreateTimerQueueTimer","CreateWaitableTimer","SetWaitableTimer","SetTimer","Select","ImpersonateLoggedOnUser","SetThreadToken","DuplicateToken","SizeOfResource","LockResource","CreateProcessInternal","TimeGetTime","EnumSystemLocalesA","UuidFromStringA"]
spying = ["AttachThreadInput","CallNextHookEx","GetAsyncKeyState","GetClipboardData","GetDC","GetDCEx","GetForegroundWindow","GetKeyboardState","GetKeyState","GetMessageA","GetRawInputData","GetWindowDC","MapVirtualKeyA","MapVirtualKeyExA","PeekMessageA","PostMessageA","PostThreadMessageA","RegisterHotKey","RegisterRawInputDevices","SendMessageA","SendMessageCallbackA","SendMessageTimeoutA","SendNotifyMessageA","SetWindowsHookExA","SetWinEventHook","UnhookWindowsHookEx","BitBlt","StretchBlt","GetKeynameTextA"]
internet = ["WinExec","FtpPutFileA","HttpOpenRequestA","HttpSendRequestA","HttpSendRequestExA","InternetCloseHandle","InternetOpenA","InternetOpenUrlA","InternetReadFile","InternetReadFileExA","InternetWriteFile","URLDownloadToFile","URLDownloadToCacheFile","URLOpenBlockingStream","URLOpenStream","Accept","Bind","Connect","Gethostbyname","Inet_addr","Recv","Send","WSAStartup","Gethostname","Socket","WSACleanup","Listen","ShellExecuteA","ShellExecuteExA","DnsQuery_A","DnsQueryEx","WNetOpenEnumA","FindFirstUrlCacheEntryA","FindNextUrlCacheEntryA","InternetConnectA","InternetSetOptionA","WSASocketA","Closesocket","WSAIoctl","ioctlsocket","HttpAddRequestHeaders"]
anti_debugging = ["CreateToolhelp32Snapshot","GetLogicalProcessorInformation","GetLogicalProcessorInformationEx","GetTickCount","OutputDebugStringA","CheckRemoteDebuggerPresent","Sleep","GetSystemTime","GetComputerNameA","SleepEx","IsDebuggerPresent","GetUserNameA","NtQueryInformationProcess","ExitWindowsEx","FindWindowA","FindWindowExA","GetForegroundWindow","GetTickCount64","QueryPerformanceFrequency","QueryPerformanceCounter","GetNativeSystemInfo","RtlGetVersion","GetSystemTimeAsFileTime","CountClipboardFormats"]
ransomware =["CryptAcquireContextA","EncryptFileA","CryptEncrypt","CryptDecrypt","CryptCreateHash","CryptHashData","CryptDeriveKey","CryptSetKeyParam","CryptGetHashParam","CryptSetKeyParam","CryptDestroyKey","CryptGenRandom","DecryptFileA","FlushEfsCache","GetLogicalDrives","GetDriveTypeA","CryptStringToBinary","CryptBinaryToString","CryptReleaseContext","CryptDestroyHash","EnumSystemLocalesA"]
helper = ["ConnectNamedPipe","CopyFileA","CreateFileA","CreateMutexA","CreateMutexExA","DeviceIoControl","FindResourceA","FindResourceExA","GetModuleBaseNameA","GetModuleFileNameA","GetModuleFileNameExA","GetTempPathA","IsWoW64Process","MoveFileA","MoveFileExA","PeekNamedPipe","WriteFile","TerminateThread","CopyFile2","CopyFileExA","CreateFile2","GetTempFileNameA","TerminateProcess","SetCurrentDirectory","FindClose","SetThreadPriority","UnmapViewOfFile","ControlService","ControlServiceExA","CreateServiceA","DeleteService","OpenSCManagerA","OpenServiceA","RegOpenKeyA","RegOpenKeyExA","StartServiceA","StartServiceCtrlDispatcherA","RegCreateKeyExA","RegCreateKeyA","RegSetValueExA","RegSetKeyValueA","RegDeleteValueA","RegOpenKeyExA","RegEnumKeyExA","RegEnumValueA","RegGetValueA","RegFlushKey","RegGetKeySecurity","RegLoadKeyA","RegLoadMUIStringA","RegOpenCurrentUser","RegOpenKeyTransactedA","RegOpenUserClassesRoot","RegOverridePredefKey","RegReplaceKeyA","RegRestoreKeyA","RegSaveKeyA","RegSaveKeyExA","RegSetKeySecurity","RegUnLoadKeyA","RegConnectRegistryA","RegCopyTreeA","RegCreateKeyTransactedA","RegDeleteKeyA","RegDeleteKeyExA","RegDeleteKeyTransactedA","RegDeleteKeyValueA","RegDeleteTreeA","RegDeleteValueA","RegCloseKey","NtClose","NtCreateFile","NtDeleteKey","NtDeleteValueKey","NtMakeTemporaryObject","NtSetContextThread","NtSetInformationProcess","NtSetInformationThread","NtSetSystemEnvironmentValueEx","NtSetValueKey","NtShutdownSystem","NtTerminateProcess","NtTerminateThread","RtlSetProcessIsCritical","DrawTextExA","GetDesktopWindow","SetClipboardData","SetWindowLongA","SetWindowLongPtrA","OpenClipboard","SetForegroundWindow","BringWindowToTop","SetFocus","ShowWindow","NetShareSetInfo","NetShareAdd","NtQueryTimer","GetIpNetTable","GetLogicalDrives","GetDriveTypeA","CreatePipe","RegEnumKeyA","WNetOpenEnumA","WNetEnumResourceA","WNetAddConnection2A","CallWindowProcA","NtResumeProcess","lstrcatA","ImpersonateLoggedOnUser","SetThreadToken","SizeOfResource","LockResource","UuidFromStringA"]



while True: # This will loop forever
    for inter in internet:
        print(inter)
        try:
            get_ttp_code_for_api(inter, "internet")
        except Exception as e:
            print('failed to run conversation', e)
            pass
        time.sleep(1)

    for enum in enumeration: # This will cycle through your enumerations
        print(enum)
        try:
            get_ttp_code_for_api(enum, "enumeration")
        except Exception as e:
            print('failed to run conversation', e)
            pass
        time.sleep(1)
    for inj in injection:
        print(inj)
        try:
            get_ttp_code_for_api(inj, "injection")
        except Exception as e:
            print('failed to run conversation', e)
            pass
        time.sleep(1)
    for eva in evasion:
        print(eva)
        try:
            get_ttp_code_for_api(eva, "evasion")
        except Exception as e:
            print('failed to run conversation', e)
            pass
        time.sleep(1)
    for spy in spying:
        print(spy)
        try:
            get_ttp_code_for_api(spy, "spying")
        except Exception as e:
            print('failed to run conversation', e)
            pass
        time.sleep(1)
    for inter in internet:
        print(inter)
        try:
            get_ttp_code_for_api(inter, "internet")
        except Exception as e:
            print('failed to run conversation', e)
            pass
        time.sleep(1)
    for anti in anti_debugging:
        print(anti)
        try:
            get_ttp_code_for_api(anti, "anti_debugging")
        except Exception as e:
            print('failed to run conversation', e)
            pass
        time.sleep(1)
    for ransom in ransomware:
        print(ransom)
        try:
            get_ttp_code_for_api(ransom, "ransomware")
        except Exception as e:
            print('failed to run conversation', e)
            pass
        time.sleep(1)
    for help in helper:
        print(help)
        try:
            get_ttp_code_for_api(help, "helper")
        except Exception as e:
            print('failed to run conversation', e)
            pass
        time.sleep(1)
