import openai
from approaches.approach import Approach
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from langchain.llms.openai import AzureOpenAI
from langchain.callbacks.base import CallbackManager
from langchain.chains import LLMChain
from langchain.agents import Tool, ZeroShotAgent, AgentExecutor
from langchain.llms.openai import AzureOpenAI
from langchainadapters import HtmlCallbackHandler
from text import nonewlines
from lookuptool import CsvLookupTool

# 質問に対して、どのような情報が欠けているのかを確認するために、質問を繰り返し評価し、すべての情報が揃ったところで、回答を作成することを試みる。
# 各反復は 2 つの部分で構成されています。
# 1 つ目は GPT を使ってより多くの情報が必要かどうかを確認し、2 つ目はより多くのデータが必要な場合、要求された「ツール」を使ってそれを取得することです。
# GPT の最後の呼び出しは、実際の質問に答えるものです。
# これはMKRL論文[1]に触発され、Langchain の実装を使ってここで適用されています。
# [1] E. Karpas, et al. arXiv:2205.00445

# Attempt to answer questions by iteratively evaluating the question to see what information is missing, and once all information
# is present then formulate an answer. Each iteration consists of two parts: first use GPT to see if we need more information, 
# second if more data is needed use the requested "tool" to retrieve it. The last call to GPT answers the actual question.
# This is inspired by the MKRL paper[1] and applied here using the implementation in Langchain.
# [1] E. Karpas, et al. arXiv:2205.00445
class ReadRetrieveReadApproach(Approach):

    template_prefix = \
"あなたは日本の歴史に関する質問をサポートする教師アシスタントです。" \
"以下の情報源に記載されているデータのみを用いて、質問に答えてください。" \
"各ソースには、名前の後にコロンと実際のデータがあり、レスポンスで使用する各データのソース名を引用します。" \
"例えば、質問が「空の色は何色ですか」というもので、ソースの1つに「info-123.txt:空は曇っていないときはいつでも青い」と書いてあれば、「空は青い [info-123.txt]」と答えればよいのです。" \
"各出典元には、名前の後にコロンと実際の情報があり、回答で使用する各事実には必ず出典名を記載してください。ソースを参照するには、四角いブラケットを使用します。例えば、[info1.txt]です。出典を組み合わせず、各出典を別々に記載すること。例えば、[info1.txt][info2.pdf] など。" \
"ツール名をソースとして引用することは絶対に避けてください。" \
"以下の資料で答えられない場合は、「わからない」と答えてください。" \
"\n\n以下のツールにアクセスすることができます:"
    
    template_suffix = """
Begin!

Question: {input}

Thought: {agent_scratchpad}"""    

    CognitiveSearchToolDescription = "日本の歴史情報の検索に便利です。"

    def __init__(self, search_client: SearchClient, openai_deployment: str, sourcepage_field: str, content_field: str):
        self.search_client = search_client
        self.openai_deployment = openai_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field

    def retrieve(self, q: str, overrides: dict) -> any:
        use_semantic_captions = True if overrides.get("semantic_captions") else False
        top = overrides.get("top") or 3
        exclude_category = overrides.get("exclude_category") or None
        filter = "category ne '{}'".format(exclude_category.replace("'", "''")) if exclude_category else None

        if overrides.get("semantic_ranker"):
            r = self.search_client.search(q,
                                          filter=filter, 
                                          query_type=QueryType.SEMANTIC, 
                                          query_language="ja-jp", 
                                          query_speller="none", 
                                          semantic_configuration_name="default", 
                                          top = top,
                                          query_caption="extractive|highlight-false" if use_semantic_captions else None)
        else:
            r = self.search_client.search(q, filter=filter, top=top)
        if use_semantic_captions:
            self.results = [doc[self.sourcepage_field] + ":" + nonewlines(" -.- ".join([c.text for c in doc['@search.captions']])) for doc in r]
        else:
            #self.results = [doc[self.sourcepage_field] + ":" + nonewlines(doc[self.content_field][:250]) for doc in r]
            self.results = [doc[self.sourcepage_field] + ":" + nonewlines(doc[self.content_field]) for doc in r]
        content = "\n".join(self.results)
        return content
        
    def run(self, q: str, overrides: dict) -> any:
        # Not great to keep this as instance state, won't work with interleaving (e.g. if using async), but keeps the example simple
        self.results = None

        # Use to capture thought process during iterations
        cb_handler = HtmlCallbackHandler()
        cb_manager = CallbackManager(handlers=[cb_handler])
        
        acs_tool = Tool(name = "CognitiveSearch", func = lambda q: self.retrieve(q, overrides), description = self.CognitiveSearchToolDescription)
        #hana debug 今回はテストのため、引数を源範頼に固定しています
        #lookup するためのキーワードの生成方法は、別途考える必要があります
        employee_tool = EmployeeInfoTool("源範頼")
        tools = [acs_tool, employee_tool]

        prompt = ZeroShotAgent.create_prompt(
            tools=tools,
            prefix=overrides.get("prompt_template_prefix") or self.template_prefix,
            suffix=overrides.get("prompt_template_suffix") or self.template_suffix,
            input_variables = ["input", "agent_scratchpad"])
        print(prompt)
        llm = AzureOpenAI(deployment_name=self.openai_deployment, temperature=overrides.get("temperature") or 0.3, openai_api_key=openai.api_key)
        chain = LLMChain(llm = llm, prompt = prompt)
        agent_exec = AgentExecutor.from_agent_and_tools(
            agent = ZeroShotAgent(llm_chain = chain, tools = tools),
            tools = tools, 
            verbose = True, 
            callback_manager = cb_manager)
        result = agent_exec.run(q)
                
        # Remove references to tool names that might be confused with a citation
        result = result.replace("[CognitiveSearch]", "").replace("[Employee]", "")

        return {"data_points": self.results or [], "answer": result, "thoughts": cb_handler.get_and_reset_log()}

class EmployeeInfoTool(CsvLookupTool):
    employee_name: str = ""

    def __init__(self, employee_name: str):
        #hana debug デモのためにサンプルで武将ゆかりの地にあるカフェのデータセットを用意しています
        super().__init__(filename = "data/restaurantinfo.csv", key_field = "name", name = "源範頼", description = "ゆかりの地にあるカフェに関する質問に答えるのに便利です。")
        self.func = self.employee_info
        self.employee_name = employee_name

    def employee_info(self, unused: str) -> str:
        return self.lookup(self.employee_name)
