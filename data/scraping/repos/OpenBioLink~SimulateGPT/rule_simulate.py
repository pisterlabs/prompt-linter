from langchain.schema import AIMessage, HumanMessage, SystemMessage
from utils import load_openai_api
from pathlib import Path

batch_chat, stream_chat = load_openai_api()

request = [
    SystemMessage(content=Path(snakemake.input.system_message).read_text()),
    HumanMessage(content=Path(snakemake.input.human_message).read_text()),
]
result = stream_chat(request)

out_path = Path(snakemake.output[0])
out_path.parent.mkdir(exist_ok=True)
out_path.write_text(result.content)
