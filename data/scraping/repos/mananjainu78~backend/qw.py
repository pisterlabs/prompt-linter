
import asyncio
import logging
import json
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
from openaiManager import completion_endpoint_plain
# app = FastAPI()

# # Configure CORS
# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],  # Replace with your frontend origin
#     allow_credentials=True,
#     allow_methods=["GET", "POST", "OPTIONS"],
#     allow_headers=["*"],
# )
# # Configure CORS

# messages = []

# Configure logging format
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)

# @app.get("/")
# def root():
#     if messages:
#         return {"message": messages[-1]}
#     else:
#         return {"message": "No messages yet"}

# @app.get("/api/messages")
# def get_messages():
#     return {"messages": "hello there"}


# @app.post("/api/messages")
# def add_message(message: dict):
#     new_message = message.get("message")
#     print("fdgdfgfdf fgfhf")
#     logging.info(f"Received new message: {new_message}")
#     messages.append(new_message)
#     # backend_response = completion_endpoint_plain(new_message)
#     return {"message": "\nimport sys\nsys.path.append('/path/to/folder')\nfrom my_module import my_function"}
#     # return {"message": backend_response}



from clone_repo import extract_user_repo_github
from ast_parser import create_repo_ast
from fastapi import FastAPI, WebSocket, HTTPException, File, UploadFile
from pydantic import BaseModel
import os, re
import zipfile
import pandas as pd
from pathlib import Path

from fastapi.responses import JSONResponse
app = FastAPI()
no_Df=False
# df = pd.DataFrame(columns=["code_chunk", "file_name", "file_path", "path_to_code_chunk","parent","prev_sibling","next_sibling","start_point","end_point","has_error","code_node_type","code_identifier","is_chunked","num_tokens","uuid_str"])

async def simulate_processing_stage_1():
    # Simulate some processing for stage 1
    await asyncio.sleep(1)

async def simulate_processing_stage_2():
    # Simulate some processing for stage 2
    await asyncio.sleep(1)

async def simulate_processing_stage_3():
    # Simulate some processing for stage 3
    await asyncio.sleep(1)


@app.websocket("/ws/initiate")
async def initiate_websocket(websocket: WebSocket):
    conversation_data1 = [{'chatId': '1692436170698', 'messages': [{'id': 1692436175759, 'avatar': 'https://example.com/avatar.png', 'username': 'User', 'message': 'hello', 'timestamp': 1692436175759}, {'id': 1692436181744, 'avatar': 'https://example.com/avatar.png', 'username': 'AI Assistant', 'message': 'Hello! How can I assist you today?', 'timestamp': 1692436181744}, {'id': 1692436191162, 'avatar': 'https://example.com/avatar.png', 'username': 'User', 'message': 'how are you', 'timestamp': 1692436191162}, {'id': 1692436198048, 'avatar': 'https://example.com/avatar.png', 'username': 'AI Assistant', 'message': "As an AI, I don't have feelings, but I'm here to help you with any questions or tasks you have. How can I assist you today?", 'timestamp': 1692436198048}, {'id': 1692436211157, 'avatar': 'https://example.com/avatar.png', 'username': 'User', 'message': 'what is the capital of denmark', 'timestamp': 1692436211157}, {'id': 1692436216969, 'avatar': 'https://example.com/avatar.png', 'username': 'AI Assistant', 'message': 'The capital of Denmark is Copenhagen.', 'timestamp': 1692436216969}, {'id': 1692436225940, 'avatar': 'https://example.com/avatar.png', 'username': 'User', 'message': 'best cricker of world', 'timestamp': 1692436225940}, {'id': 1692436233668, 'avatar': 'https://example.com/avatar.png', 'username': 'AI Assistant', 'message': 'There have been many great cricketers throughout history, and it is subjective to determine the best cricketer of all time. However, some of the most highly regarded cricketers include Sir Donald Bradman, Sachin Tendulkar, Sir Vivian Richards, Sir Garfield Sobers, and Sir Jack Hobbs. These players have achieved remarkable records and have left a lasting impact on the game of cricket.', 'timestamp': 1692436233668}]}, {'chatId': '1692436239709', 'messages': []}, {'chatId': '1692436243005', 'messages': [{'id': 1692436251519, 'avatar': 'https://example.com/avatar.png', 'username': 'User', 'message': 'some icream flovors', 'timestamp': 1692436251519}, {'id': 1692436259952, 'avatar': 'https://example.com/avatar.png', 'username': 'AI Assistant', 'message': '1. Vanilla\n2. Chocolate\n3. Strawberry\n4. Mint chocolate chip\n5. Cookies and cream\n6. Butter pecan\n7. Rocky road\n8. Coffee\n9. Pistachio\n10. Salted caramel\n11. Neapolitan\n12. Cherry Garcia (cherry ice cream with chocolate chunks and cherries)\n13. Peanut butter cup\n14. Coconut\n15. Birthday cake\n16. Matcha green tea\n17. Black raspberry\n18. Dulce de leche\n19. Red velvet\n20. Lemon sorbet', 'timestamp': 1692436259952}, {'id': 1692436267399, 'avatar': 'https://example.com/avatar.png', 'username': 'User', 'message': 'different color of apple', 'timestamp': 1692436267399}, {'id': 1692436279132, 'avatar': 'https://example.com/avatar.png', 'username': 'AI Assistant', 'message': 'There are several different colors of apples, including:\n\n1. Red: This is the most common color for apples, with varieties such as Red Delicious, Gala, and Fuji.\n\n2. Green: Green apples, such as Granny Smith and Golden Delicious, have a tart flavor and are often used in baking or for making apple cider.\n\n3. Yellow: Yellow apples, like Yellow Transparent and Yellow Newtown Pippin, have a sweet and tangy taste.\n\n4. Pink: Pink Lady apples have a distinctive pinkish-red skin and a crisp, sweet-tart flavor.\n\n5. Bi-colored: Some apples have a combination of colors, such as the popular Honeycrisp apple, which has a red and yellow skin.\n\n6. Striped: Certain apple varieties, like the McIntosh apple, have a striped pattern on their skin, with a mix of red and green.\n\n7. Purple: There are a few varieties of purple apples, such as the Black Diamond apple, which have a deep purple or almost black skin.\n\nThese are just a few examples of the different colors of apples available. The specific color and appearance of an apple can vary depending on the variety and ripeness.', 'timestamp': 1692436279132}, {'id': 1692436280326, 'avatar': 'https://example.com/avatar.png', 'username': 'User', 'message': 'japan to india distance', 'timestamp': 1692436280326}, {'id': 1692436286987, 'avatar': 'https://example.com/avatar.png', 'username': 'AI Assistant', 'message': 'The distance between Japan and India is approximately 4,500 kilometers (2,800 miles) when measured in a straight line. However, the actual distance may vary depending on the route taken for travel.', 'timestamp': 1692436286987}, {'id': 1692436288296, 'avatar': 'https://example.com/avatar.png', 'username': 'User', 'message': 'bye bye', 'timestamp': 1692436288296}, {'id': 1692436294137, 'avatar': 'https://example.com/avatar.png', 'username': 'AI Assistant', 'message': 'Goodbye! Have a great day!', 'timestamp': 1692436294137}]}]

    await websocket.accept()
    # await asyncio.sleep(1)
    # await websocket.send_text("Hello  now from FastAPI!")

    await websocket.send_text(json.dumps({"chat_data":conversation_data1,"action": "initiate", "message": "ploaded Successfully"}))
    
    csv_files = []
    root1 = Path(__file__).parent
    csv_directory_path = f"{root1}/repositories"  # Replace with the actual path to your directory

    for filename in os.listdir(csv_directory_path):
        if filename.endswith(".csv"):
            csv_files.append(os.path.splitext(filename)[0])
    print(csv_files)       
    await websocket.send_text(json.dumps({"chat_data":csv_files,"action": "cvalue", "message": "ploaded Successfully"}))

    while True:
    # progress_message = {}
        message = await websocket.receive_text()
        print(message)

@app.websocket("/ws/process")
async def status_websocket(websocket: WebSocket):
    await websocket.accept()
    while True:
        progress_message = {}
        message = await websocket.receive_text()
        message_data = json.loads(message)
        filename = message_data.get('filename')
        print("llll",filename)
        if message_data['type'] == 'filename':
            data = await websocket.receive_bytes()  # Receive binary data in chunks
            await websocket.send_text(json.dumps({"action": "process", "message": f"{filename} Uploaded Successfully"}))
            if data:
                print("ertre")
                file_path=f'repositories/data/{filename}'
                # Process the received binary data as a chunk of a larger file
                with open(file_path, "ab") as f:
                    print("opop")
                    f.write(data)
                await asyncio.sleep(4)
                await websocket.send_text(json.dumps({"action": "process", "message":f"{filename} Processed Successfully"}))
                if filename.endswith('.zip'):
                    with zipfile.ZipFile(f'repositories/data/{filename}', 'r') as zip_ref:
                        zip_ref.extractall(f'repositories/data/')
                        print("Zip file extracted successfully.")
                    os.remove(file_path)
                    await asyncio.sleep(4)
                    await websocket.send_text(json.dumps({"action": "process", "message": "{filename} Extracted Successfully"}))
                await websocket.send_text(json.dumps({"type": "success"}))
                progress_message['action'] = 'process'
                progress_message['message'] = 'success'
                print(progress_message)
                await websocket.send_json(progress_message)
                await websocket.send_text(json.dumps({"action": "process", "message": "# Simulate some processing for stage 1"}))
                await simulate_processing_stage_1()  # Call your processing function
                await websocket.send_text(json.dumps({"action": "process", "message": "Stage 2              "}))
        elif message_data['type'] == 'url':
            source_url=message_data['url_string']
            if message_data['sourcetype'] == 'Github':
                pattern = r'https://github.com/([^/]+)/([^/]+)'
    
                    # Search for the pattern in the input URL
                await asyncio.sleep(4)
                await websocket.send_text(json.dumps({"action": "process", "message": "Search for the pattern in the input URL"}))
                
                match = re.search(pattern, source_url)
                print(match)
                if match:
                    # Extract the username and repo name from the matched groups
                    user_name = match.group(1)
                    repo_name = match.group(2)
                    if user_name and repo_name:
                        print("Username:", user_name)
                        print("Repository:", repo_name)
                        print(source_url)
                        await asyncio.sleep(1)
                        await websocket.send_text(json.dumps({"action": "process", "message": "Username and Repo name extracted Successfully"}))

                        extract_user_repo_github(source_url,user_name,repo_name)
                        await asyncio.sleep(2)
                        await websocket.send_text(json.dumps({"action": "process", "message": "GitHub Repository cloned Successfully"}))

                        print("nice")
                        await asyncio.sleep(3)
                        await websocket.send_text(json.dumps({"action": "process", "message": f"Parsing source code of {user_name}/{repo_name}"}))

                        create_repo_ast(f"{user_name}_{repo_name}")
                        await asyncio.sleep(4)
                        await websocket.send_text(json.dumps({"action": "process", "message": f" Source code of {user_name}/{repo_name} Parsed Successfully"}))
                        await asyncio.sleep(5)
                        await websocket.send_text(json.dumps({"data_source_name": f"{user_name}_{repo_name}","action": "process", "message": f" Data Source {user_name}/{repo_name} Successfully"}))


                        print("ioio")
                    else:
                        print("Invalid GitHub repository URL")
                        await asyncio.sleep(6)
                        await websocket.send_text(json.dumps({"action": "process", "message": "Invalid GitHub Repository URL"}))
        simulate_processing_stage_2()  # Call another processing function
        await websocket.send_text(json.dumps({"action": "process", "message": ""}))    
                          
    # aweeeeait websocket.accept()
    # await websocket.send_text("WebSocket connection established.")

    # file_data = bytearray()
    # while True:
    #     message = await websocket.receive_text()
    #     if message == "File upload complete":
    #         break
    #     file_data.extend(message.encode())  # Convert string message to bytes and extend bytearray

    # # Save the received file data to a file
    # file_path = os.path.join(UPLOAD_FOLDER, "data")
    # with file_path.open("wb") as f:
    #     f.write(file_data)

    # await websocket.send_text("File received and saved.")
    # await websocket.close()
    # await websocket.accept()
    # logging.info(f"hello: ")
    # try:
    #     while True:
    #         logging.info(f"hello1: ")
    #         file_data = await websocket.receive_text()
    #         logging.info(f"hello2: ")
    #         print(file_data)
    #         await websocket.send_text("File received and processed")

        # Process the received file data here
        # You can save it to a file or perform any other necessary actions

                # # await websocket.send_text(f"Received file name: {data}")
                # file_path = os.path.join(UPLOAD_FOLDER, "data")
                # print(file_path)
                # async with websocket.stream(filename=file_path) as upload_file:
                #     await shutil.copyfileobj(upload_file, open(file_path, "wb"))

                # repo_parts = "qwe"

                # if len(repo_parts) != 2:
                #     logging.info(f"Received chat message: ")

                #     message['action'] = 'status'
                #     message['message'] = 'Invalid link'
                #     await websocket.send_json(message)
                # else:
                #     username, repo_name = repo_parts
                #     # Perform the extraction and cloning here
                #     try:
                #         # Extract info
                #         message['action'] = 'status'
                #         message['message'] = 'Extraction complete'
                #         await websocket.send_json(message)

                #         # Clone the repository

                #         message['action'] = 'status'
                #         message['message'] = 'Cloned successfully'
                #         await websocket.send_json(message)

                #     except Exception as e:
                #         message['action'] = 'status'
                #         message['message'] = f'Error: {str(e)}'
                #         await websocket.send_json(message)
    # except Exception:
    #     pass




@app.websocket("/ws/chat")
async def chat_socket(websocket: WebSocket):
    await websocket.accept()
    logging.info(f"hello: ")
    try:
        while True:
            data = await websocket.receive_text()
            if data:
                  query_data = json.loads(data)
                  if query_data["type"]=="activate":
                      print(query_data["active_df"])
                      root = Path(__file__).parent
                      if len(query_data["active_df"])!=0:
                        data_frame=query_data["active_df"][0]
                        df=pd.read_csv(f"{root}/repositories/{data_frame}.csv")
                        print("outer")
                        for i in range(1, len(query_data["active_df"])):
                            print("Inner")
                            data_frame = query_data["active_df"][i]
                            print("olko",data_frame)
                            print(f"{root}/repositories/{data_frame}.csv")
                            df1=pd.read_csv(f"{root}/repositories/{data_frame}.csv")
                            df = pd.concat([df, df1], ignore_index=True)
                            print(df)
                            print("nowu")
                            print(df1)
                        df.to_csv(f"{root}/repositories/qwe.csv", index=False)                      
                      else:
                        no_Df=True
                      continue
            print("fdgdfgfdf fgfhf")
            logging.info(f"Received new message: ")
            query_data = json.loads(data)
            message = {}

            message['action'] = 'chat'
            message['message'] = 'valid link'
            message['chatId'] = query_data["chatId"]
            message['progressbar'] = True

            # await websocket.send_json(message)
            # await asyncio.sleep(8)
            if data:

                logging.info(f"Received new message: {data} ")

                repo_parts = "qwe"

                if len(repo_parts) != 2:
                    logging.info(f"Received chat message: ")
               
                    backend_response = """
In Java, the `TypeReference` class is used to capture the generic type information at runtime. It is commonly used when working with libraries or frameworks that require generic type information, such as JSON parsing libraries like Jackson or Gson.

Here's an example of how to use `TypeReference` in Java:

1. Import the necessary classes:
```java
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
```

2. Create an instance of `ObjectMapper`:
```java
ObjectMapper objectMapper = new ObjectMapper();
```

3. Define a generic type using `TypeReference`:
```java
TypeReference<List<String>> typeReference = new TypeReference<List<String>>() {};
```
In this example, we are defining a `TypeReference` for a `List` of `String` objects.

4. Use the `ObjectMapper` to read or write JSON data:
```java
String json = "[\"apple\", \"banana\", \"orange\"]";

try {
    List<String> fruits = objectMapper.readValue(json, typeReference);
    System.out.println(fruits); // Output: [apple, banana, orange]
} catch (IOException e) {
    e.printStackTrace();
}
```
In this example, we are using the `readValue()` method of `ObjectMapper` to parse the JSON string into a `List` of `String` objects, using the `TypeReference` to preserve the generic type information.

Note that the `TypeReference` is an abstract class, so we need to create an anonymous subclass by using `{}` at the end of the declaration.

By using `TypeReference`, we can work with generic types at runtime without losing the type information.
fgdfgd

"""     
                    # backend_response = completion_endpoint_plain(query_data["data"])
                    print(query_data["data"],backend_response)
                    print(query_data["ch"])
                    message['action'] = 'chat'
                    message['message'] = backend_response
                    message['chatId'] = query_data["chatId"]
                    message['newvalue'] = "acha"
                    message['progressbar'] = False
                    await asyncio.sleep(5)

                    await websocket.send_json(message)
                else:
                    username, repo_name = repo_parts
                    # Perform the extraction and cloning here
                    try:
                        # Extract info
                        message['action'] = 'status'
                        message['message'] = 'Extraction complete'
                        await websocket.send_json(message)

                        # Clone the repository

                        message['action'] = 'status'
                        message['message'] = 'Cloned successfully'
                        await websocket.send_json(message)

                    except Exception as e:
                        message['action'] = 'status'
                        message['message'] = f'Error: {str(e)}'
                        await websocket.send_json(message)
    except Exception:
        pass