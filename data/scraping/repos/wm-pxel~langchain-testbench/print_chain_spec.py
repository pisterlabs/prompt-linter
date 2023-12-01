from langchain.llms import OpenAI
from lib.model.chain_revision import ChainRevision
from lib.model.chain_spec import LLMSpec, SequentialSpec
import dotenv

dotenv.load_dotenv()


llm = OpenAI(temperature=0.8)

chain = SequentialSpec(
  chain_id = 0,
  input_keys = ["input", "memory_in"],
  output_keys = ["output", "memory_out"],
  chain_type = "sequential_spec",
  chains = [
    LLMSpec(
      chain_id = 1,
      input_keys = ["input", "memory_in"],
      output_key = "output",
      prompt = "Context: {memory_in}\n\nYou are a witty but kind professor. Respond in a single paragraph to the student's question or statement.\n\nStudent: {input}\n\nResponse:",
      llm_key = "llm",
      chain_type = "llm_spec",
    ),
    LLMSpec(
      chain_id = 2,
      input_keys = ["input", "output", "memory_in"],
      output_key = "memory_out",
      prompt = "You are a witty but kind professor. Summarize in a single paragraph the conversation up to this point including the context.\n\nContext: {memory_in}\n\nStudent: {input}\n\nProfessor: {output}\n\nSummary:",
      llm_key = "llm",
      chain_type = "llm_spec",
    ),
  ]
)

chain_revision = ChainRevision(
  llms = {"llm": llm},
  chain = chain
)

print(chain_revision.json())