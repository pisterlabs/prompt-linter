from langchain import HuggingFacePipeline, PromptTemplate, LLMChain
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# GPUの確認
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n!!! current device is {device} !!!\n")

# モデルのダウンロード
model_id = "bigscience/bloom-1b7"
# model_id = "inu-ai/dolly-japanese-gpt-1b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

# LangChainのLLMとして利用
task = "text-generation"
pipe = pipeline(
    task, 
    model=model,
    tokenizer=tokenizer,
    device="cuda:0",
    framework='pt',
    max_length=32,
    temperature=0,
    repetition_penalty=1.0,
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

# LLMs: langchainで上記モデルを利用する
llm = HuggingFacePipeline(pipeline=pipe)

# Prompts: プロンプトを作成
# template = """Question: {question}
# Answer: Let's think step by step."""
template = """<s>\n以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。
要求を適切に満たす応答を書きなさい。
\n[SEP]\n指示:\n{instruction}\n[SEP]\n入力:\n{input}\n[SEP]\n応答:\n"""

instruction = """
    "ユーザ:あなたはユーザの質問に回答するアシスタントです。"\n
    "アシスタント:はい。何でも質問してください。"\n
    "ユーザ:同じ言葉を繰り返さないこと。何でも正確に要約して答えること。"\n
    "アシスタント:了解しました。同じ言葉を繰り返さず、何でも正確に要約して答えます。"\n
"""

prompt = PromptTemplate(template=template, input_variables=["instructions ""question"])

# Chains: llmを利用可能な状態にする
llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

# 質問を投げる
# question = "How can I get the end of the list in Python?"
question = "Pythonでリストの最後尾を取得するには？"
generated_text = llm_chain.run(instruction,question)
print(generated_text)