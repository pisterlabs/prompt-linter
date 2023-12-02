# pip install semantic-kernel
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai.services.open_ai_text_completion import OpenAITextCompletion

# Creating the kernel
kernel = sk.Kernel()
api_key, org_id = sk.openai_settings_from_dot_env()
kernel.add_text_completion_service("OpenAI_davinci", OpenAITextCompletion("text-davinci-003", api_key))

# Craft your prompt here

prompt = """
You are a helpful chatbot. Please answer the question based on the provided context below.
Do not make up your answer or add anything which is not in the context. If the answer is not provided in the context, politely say that
you do not know.
context :{{$context_str}}
User: {{$question_str}}
"""

# Instantiate the semantic function
qa_chat_bot = kernel.create_semantic_function(
    prompt_template=prompt,
    description="Answers question based on provided context",
    max_tokens=1000
)

# This is the context to be used to answer question
context_str = "Semantic Kernel is an SDK that integrates Large Language Models (LLMs) " \
              "like OpenAI, Azure OpenAI, and Hugging Face with conventional programming languages " \
              "like C#, Python, and Java. Semantic Kernel achieves this by allowing " \
              "you to define plugins that can be chained together " \
              "in just a few lines of code.What makes Semantic Kernel special, " \
              "however, is its ability to automatically orchestrate plugins with AI. " \
              "With Semantic Kernel planners, you can ask an LLM to generate a plan " \
              "that achieves a user's unique goal. Afterwards, Semantic Kernel will execute the plan for the user."

# This is something unique. It returns the SKContext object
sk_context = kernel.create_new_context()
sk_context["context_str"] = context_str

while True:
    question_str = input("Enter your Question\n\n")
    sk_context["question_str"] = question_str
    answer = qa_chat_bot.invoke(context=sk_context)
    print(answer)
