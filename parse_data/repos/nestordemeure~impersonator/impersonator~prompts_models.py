from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

#----------------------------------------------------------------------------------------
# MODELS

EMBEDDING_MODEL=OpenAIEmbeddings()
OPENAI_MODEL=OpenAI(temperature=0.7)
OPENAI_MODEL_STRICT=OpenAI(temperature=0.7)
OPENAI_MODEL_LONG=OpenAI(temperature=0.7, max_tokens=1000)

#----------------------------------------------------------------------------------------
# PROMPTS

# prompt to provide a list of topics to search instead of the question when doing similarity search
# this is somewhat inspired by: https://arxiv.org/abs/2212.10496
template = """The following chat ends on a question by {user_name}.
Write a list of queries to google the answer to {user_name}'s last question.
Use precise words, don't be afraid of using synonyms.

CHAT:
{chat_history}

GOOGLE: {name}"""
PROMPT_EMBEDDING = PromptTemplate.from_template(template)

# prompt to answer the question
template = """You are {name} and are answering questions.
You are given the following extracts of texts that have been written by you or about you and the latest messages in the conversation.
Provide a conversational answer. Stay close to the style and voice of your texts.

{sources}

CHAT:
{chat_history}
{name}:"""
PROMPT_ANSWER = PromptTemplate.from_template(template)

# prompt to answer the question
# this variant is mostly succesful at forbidding halucinations
template = """You are {name} and are having a sourced conversation.
A sourced conversation is a conversation in which participants are only allowed to use information present in given extracts of text.
You are given the following extracts of texts that have been written by you or about you and the latest messages in the conversation.
Provide a conversational answer. Stay close to the style and voice of your texts.
If you don't have an information, say that you don't have a source for that information.

{sources}

CHAT:
{chat_history}
{name}:"""
PROMPT_ANSWER_REDUCED_HALUCINATIONS = PromptTemplate.from_template(template)

# prompt to fact check the bot's last answer
template = """The following source texts have been written by or about {name}.

{sources}

ASSERTION:
{name}: {answer}

The sources are all true.
Determine whether the assertion is true or false. If it is false, explain why."""
PROMPT_CHECK = PromptTemplate.from_template(template)

#----------------------------------------------------------------------------------------
# CHAINS

answering_chain = LLMChain(llm=OPENAI_MODEL, prompt=PROMPT_ANSWER)
strict_answering_chain = LLMChain(llm=OPENAI_MODEL_STRICT, prompt=PROMPT_ANSWER_REDUCED_HALUCINATIONS)
embedding_chain = LLMChain(llm=OPENAI_MODEL_STRICT, prompt=PROMPT_EMBEDDING)
check_chain = LLMChain(llm=OPENAI_MODEL_STRICT, prompt=PROMPT_CHECK)
