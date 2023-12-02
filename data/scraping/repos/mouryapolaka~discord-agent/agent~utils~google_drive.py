from langchain.document_loaders import GoogleDriveLoader
from pathlib import Path
import os

# Ensure the ~/.credentials directory exists
credentials_dir = Path.home() / '.credentials'
credentials_dir.mkdir(exist_ok=True)

# Set the path for credentials.json and token.json
credentials_path = credentials_dir / 'credentials.json'
token_path = credentials_dir / 'token.json'

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(credentials_path)

# Initialize GoogleDriveLoader
loader = GoogleDriveLoader(
    folder_id="root",  # Replace with your actual folder ID
    credentials_path=credentials_path,
    token_path=token_path,
    recursive=False,
)

# Attempt to load documents
docs = loader.load()
print(docs)