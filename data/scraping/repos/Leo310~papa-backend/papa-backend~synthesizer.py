from dotenv import load_dotenv

from llama_index.llms import OpenAI
from llama_index.prompts import PromptTemplate

from retriever import run_retrieval

import nest_asyncio
import asyncio

nest_asyncio.apply()


async def acombine_results(
    texts,
    query_str,
    qa_prompt,
    llm,
    cur_prompt_list,
    num_children,
):
    fmt_prompts = []
    for idx in range(0, len(texts), num_children):
        text_batch = texts[idx : idx + num_children]
        context_str = "\n\n".join([t for t in text_batch])
        fmt_qa_prompt = qa_prompt.format(context_str=context_str, query_str=query_str)
        # print(f"*****Prompt******:\n{fmt_qa_prompt}\n\n")
        fmt_prompts.append(fmt_qa_prompt)
        cur_prompt_list.append(fmt_qa_prompt)

    tasks = [llm.acomplete(p) for p in fmt_prompts]
    combined_responses = await asyncio.gather(*tasks)
    new_texts = [str(r) for r in combined_responses]

    if len(new_texts) == 1:
        return new_texts[0]
    else:
        return await acombine_results(
            new_texts,
            query_str,
            qa_prompt,
            llm,
            cur_prompt_list,
            num_children=num_children,
        )


async def agenerate_response_hs(retrieved_nodes, query_str, qa_prompt, llm):
    """Generate a response using hierarchical summarization strategy.
    Combine num_children nodes hierarchically until we get one root node.
    """
    fmt_prompts = []
    node_responses = []
    for node in retrieved_nodes:
        context_str = str(node.metadata) + "\n" + node.get_content()
        fmt_qa_prompt = qa_prompt.format(context_str=context_str, query_str=query_str)
        print(f"*****Prompt******:\n{fmt_qa_prompt}\n\n")
        fmt_prompts.append(fmt_qa_prompt)

    tasks = [llm.acomplete(p) for p in fmt_prompts]
    node_responses = await asyncio.gather(*tasks)

    response_txt = await acombine_results(
        [str(r) for r in node_responses],
        query_str,
        qa_prompt,
        llm,
        fmt_prompts,
        num_children=10,
    )

    return response_txt, fmt_prompts


async def run_synthesizer(query_str):
    llm = OpenAI(model_name="gpt-3.5-turbo")
    qa_prompt = PromptTemplate(
        """\
        Your are a personal assistant that should answer a query based on the users obsidian notes. 
        The context information from these notes is below.
        ---------------------
        {context_str}
        ---------------------
        Provide a response based on the context provided, without fabricating information.
        If you lack the necessary information, simply state 'I don't know.'
        You may include additional information in your response,
        but clearly indicate that it is a personal assistant's addition.
        Query: {query_str}
        Answer: \
        """
    )

    retrieved_nodes = run_retrieval(query_str)
    # context_str = "\n\n".join(
    #     ["%s\n%s" % (str(r.metadata), r.get_content()) for r in retrieved_nodes]
    # )
    # fmt_qa_prompt = qa_prompt.format(context_str=context_str, query_str=query_str)
    # response = llm.complete(fmt_qa_prompt)

    response, fmt_prompts = await agenerate_response_hs(
        retrieved_nodes, query_str, qa_prompt, llm
    )
    # print(f"*****Prompt******:\n{fmt_prompts}\n\n")
    print(f"*****Response******:\n{response}\n\n")
    return str(response)


if __name__ == "__main__":
    load_dotenv()
    response = run_synthesizer("Write a technical Web3 blog post in my style.")
    # print(f"*****Response******:\n{response}\n\n")
