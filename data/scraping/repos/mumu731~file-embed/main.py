import re
import os
import uuid
import tiktoken
from fastapi import FastAPI, File, UploadFile, HTTPException
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import CohereEmbeddings

# Cohere密钥
cohere_api_key = os.getenv('COHERE_API_KEY', "")

app = FastAPI()

# Tiktoken计算
encoding_name = "cl100k_base"
encoding = tiktoken.get_encoding(encoding_name)

def num_tokens_from_string(string: str) -> int:
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Create a directory to store uploaded files
os.makedirs("uploads", exist_ok=True)

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InvalidFileFormatError(HTTPException):
    def __init__(self):
        super().__init__(status_code=400, detail='无效的文件格式')

class EmptyFileError(HTTPException):
    def __init__(self):
        super().__init__(status_code=400, detail='文件内容为空')

@app.post('/file_embed')
async def file_embed(file: UploadFile):
    try:
        # 检查是否上传了文件
        if not file:
            raise HTTPException(status_code=400, detail='未上传文件')

        # 检查文件扩展名
        if not file.filename.endswith('.pdf') and not file.filename.endswith('.txt'):
            raise InvalidFileFormatError()

        # 将文件保存到临时目录
        uuid_filename = str(uuid.uuid4())
        filename_with_uuid = f'{uuid_filename}_{file.filename}'
        temp_file_path = f'uploads/{filename_with_uuid}'
        with open(temp_file_path, 'wb') as f:
            f.write(file.file.read())

        splitte_text = []
        data_text = ""  # 存储所有metadata的字符串变量
        tokens = 0  # Tokens

        if file.filename.endswith('.pdf'):
            # 创建一个PDF加载器并加载文件
            loader = PyPDFLoader(temp_file_path)
            pages = loader.load_and_split()
            for page in pages:
                data_text += page.page_content + " "
                tokens += num_tokens_from_string(page.page_content)
        if file.filename.endswith('.txt'):
            loader = UnstructuredFileLoader(temp_file_path)
            pages = loader.load()
            for page in pages:
                data_text += page.page_content + " "
                tokens += num_tokens_from_string(page.page_content)

        # 检查内容是否为空
        if not data_text.strip():
            raise EmptyFileError()

        # 删除上传的文件
        os.remove(temp_file_path)

        fromtext = re.sub(r'\s+', '\n', data_text.strip())

        # 创建一个文本分割器并分割文本
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=20,
            length_function=len,
        )
        texts = text_splitter.create_documents([fromtext])

        # 创建一个嵌入器并嵌入文档
        embeddings = CohereEmbeddings(cohere_api_key=cohere_api_key)
        # 将分割后的文本存储到列表中
        for page in texts:
            splitte_text.append({
                'content': page.page_content,
                'embedddoc': embeddings.embed_documents([page.page_content])
            })

        # 返回转换后的内容
        return {
            'preview': splitte_text,
            'tokens': tokens
        }

    except InvalidFileFormatError as e:
        raise e
    except EmptyFileError as e:
        raise e
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail='Internal Server Error')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
