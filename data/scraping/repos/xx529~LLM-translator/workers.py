from src.worker.models import Model
from src.utils.tools import Log
from langchain import PromptTemplate


class Worker:
    def __init__(self, llm_model, temperature=None, top_k=None, top_p=None):
        self.llm = llm_model
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.model_name = Model(self.llm).name

    def __call__(self, **kwargs):
        Log.info(f'using model "{self.model_name}"')
        Log.info(f'temperature: {self.temperature}')
        Log.info(f'top_k: {self.top_k}')
        Log.info(f'top_p: {self.top_p}')

        p = PromptTemplate.from_template(self.get_template()).format(**kwargs)
        Log.info(f'prompt: {p}')

        res = self.llm(p, temperature=self.temperature, top_k=self.top_k, top_p=self.top_p)
        res = self.clean_process(res)
        res = self.custom_process(res)
        Log.info(f'response: {res}')
        return res

    def get_template(self):
        template_func = getattr(self, f'template_{self.model_name.lower()}', self.template_default)
        Log.info(f'using template "{template_func.__name__}"')
        return template_func()

    def clean_process(self, res: str):
        res = res.replace('\\n', '')

        if res.startswith('"'):
            res = res[1:]
        if res.endswith('"'):
            res = res[:-1]

        res = res.strip()
        return res

    def custom_process(self, res: str):
        return res

    def template_default(self):
        raise NotImplementedError


class Translator(Worker):

    def __init__(self, llm_model, temperature=None, top_k=None, top_p=None):
        super().__init__(llm_model, temperature, top_k, top_p)

    def template_default(self):
        return '原文: {content}，请翻译成{dst_lang}: '

    @staticmethod
    def template_zhipu():
        return '现在你是一个翻译专家，根据给出的原文，给出对应的{dst_lang}翻译，原文: {content}，翻译: '


class Summarizer(Worker):

    def __init__(self, llm_model, temperature=None, top_k=None, top_p=None):
        super().__init__(llm_model, temperature, top_k, top_p)

    def template_default(self):
        return '用{length}个左右的文字为已下内容提取摘要: {content}，摘要：'


class Extractor(Worker):

    def __init__(self, llm_model, temperature=None, top_k=None, top_p=None):
        super().__init__(llm_model, temperature, top_k, top_p)

    def template_default(self):
        t = """
        提取以下文本中的人名，并以提供的json格式返回结果。
        
        文本如下：
        ```text
        {content}
        ```

        提取结果json是：
        """
        return t

    def clean_process(self, res):
        return res
