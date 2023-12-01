import os
import openai
import dotenv  # Import the dotenv module

# Load environment variables from the .env file
dotenv.load_dotenv()

# Set the API key using os.environ
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or "YOUR_API_KEY"
openai.api_key = os.environ["OPENAI_API_KEY"]

res = openai.File.create(
    file=open("training_data.jsonl", "r"),
    purpose='fine-tune'
)

file_id = res["id"]
print(f'File ID: {file_id}.')
