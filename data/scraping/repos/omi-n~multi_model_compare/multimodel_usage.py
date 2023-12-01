from multimodel import multi_model as mm
from langchain import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

config = mm.MultiModelPrompterConfig(
    gpt4_8k=True,
    llama="1000",
    gpt35_turbo=True
)

template = """Question: {question}

Answer: Let's think step by step."""

templates = [PromptTemplate(template=template, input_variables=["question"])]

callback_manager = CallbackManager([])

multimodel = mm.MultiModelPrompter(config)
question1 = "What is the shortest possible number of moves in which a knight on an empty chessboard can move from d4 to g5? What about g5 to f6?"
question2 = "What is the shortest possible number of moves in which a knight on an empty chessboard can move from d4 to f6?"
question3 = "What is the shortest possible number of moves in which a knight on an empty chessboard can move from d4 to g4?"
questions = [(0, question1), (0, question2)]

multimodel.run_prompt(
    "test_out_2.txt",
    questions,
    templates,
    None,
    512,
    False,
    ".env",
    3
)