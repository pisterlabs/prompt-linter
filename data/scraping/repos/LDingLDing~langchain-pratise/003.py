from dotenv import load_dotenv
load_dotenv()

# langchain 没有兼容最新版本 HuggingFaceHub 0.19.0
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# from langchain import HuggingFaceHub
# llm = HuggingFaceHub(model_id="bigscience/bloom-1b7")
from langchain.llms import HuggingFaceHub

llm = HuggingFaceHub(repo_id="google/flan-t5-large")
text = llm("请给我写一句情人节红玫瑰的中文宣传语")
print(text)