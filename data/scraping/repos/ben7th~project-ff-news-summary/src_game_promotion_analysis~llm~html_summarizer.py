import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.request_gpt import request_gpt
from utils.remove_html_tags import get_text_content
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

from utils.print_color import prYellow


def __split(text):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name='gpt-3.5-turbo', chunk_size=2000, chunk_overlap=100
    )
    return splitter.split_text(text)



PROMPT = """请对以下内容进行要点总结:


标题: `{title}`

```{text}```


请按以下格式输出：```
- 要点
- 要点
```


总结:"""


def LLM_html_summary(title, html_text, verbose=False):
    """针对长网页生成要点总结"""

    text_content = get_text_content(html_text)
    splitted_texts = __split(text_content)

    # prYellow(PROMPT)

    results = []
    count = len(splitted_texts)
    for index, text in enumerate(splitted_texts, start=1):
        prompt = PROMPT.format(title=title, text=text)
        if verbose:
            print(prompt)
        print(f'请求 {index}/{count}')
        result = request_gpt(prompt)
        results.append(result)

    return "\n\n".join(results)