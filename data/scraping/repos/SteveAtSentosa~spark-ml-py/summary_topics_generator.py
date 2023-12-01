from typing import Dict
from langchain import PromptTemplate, LLMChain
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from domain.db_handler import Database


def extract_info_from_db_data(problem: Dict) -> str:
    text = f"Problem: {problem['name']}\nStatus: {problem['status']}\n"
    for variant in problem["variants"]:
        text += f"- Variant: {variant['content']} (Type: {variant['type']}, Order: {variant['order']})\n"
        for sub_problem in variant["sub_problems"]:
            text += f"    - Sub-problem: {sub_problem}\n"
    return text


def _generate_keywords(llm: ChatOpenAI, text: str) -> str:
    template = (
        "See the summary of a problem from our ideation platform. Choose EXACTLY 4 KEYWORDS that better describe "
        "the problem below: \n\n{text}"
    )
    prompt = PromptTemplate(template=template, input_variables=["text"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    keywords = llm_chain.run(text)
    return keywords


def _generate_topics(llm: ChatOpenAI, text: str) -> str:
    template = (
        "You have below the summary of several problems from our ideation platform and some "
        "metadata about them. Choose EXACTLY 4 KEY TOPICS that better describe "
        "the problems below: \n\n{text}"
    )
    prompt = PromptTemplate(template=template, input_variables=["text"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    topics = llm_chain.run(text)
    return topics


def daily_digest_llm(user_id: str) -> Dict:
    db = Database()
    user_name = db.fetch_username(user_id)
    sample_data = db.fetch_data(user_id)

    doc_splits = [extract_info_from_db_data(problem) for problem in sample_data]
    docs = [Document(page_content=split_text) for split_text in doc_splits]
    llm = ChatOpenAI(temperature=0, max_tokens=2000, model_name="gpt-3.5-turbo")

    summary_prompt_template = "Write a summary focusing only on the relevant information from the text below \n\n{text}."
    daily_digest_template = (
        f"As Spark, your role is to be {user_name}'s personal buddy. You are tasked with creating "
        f"daily summaries of the latest problems created in our ideation platform in {user_name}'s organization. "
        f"Your aim is to highlight the crucial topics being addressed by their team, starting with some background "
        f"and followed by an enumerated list of discussion points. Keep the tone light and engaging. "
        f"This message will be featured on Spark's welcome page. "
        "Here's the information about the latest problems: \n\n{text}."
    )

    summary_prompt = PromptTemplate(
        template=summary_prompt_template, input_variables=["text"]
    )
    daily_digest_template = PromptTemplate(
        template=daily_digest_template, input_variables=["text"]
    )

    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        return_intermediate_steps=True,
        map_prompt=summary_prompt,
        combine_prompt=daily_digest_template,
    )

    res = chain({"input_documents": docs}, return_only_outputs=True)
    topics = _generate_topics(llm, res["intermediate_steps"])

    return {
        # "intermediate_steps": res["intermediate_steps"][0],
        "output": res["output_text"],
        "topics": topics,
    }
