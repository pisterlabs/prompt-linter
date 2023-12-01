import langchain
from langchain.docstore.document import Document
from cat.mad_hatter.decorators import hook
from cat.log import log


@hook
def before_rabbithole_stores_documents(docs, cat):
    # Load settings
    settings = cat.mad_hatter.plugins["ccat_summarization"].load_settings()
    group_size = settings["group_size"]

    # LLM Chain from Langchain
    summarization_prompt = f"{settings['summarization_prompt']}\n {{text}}"
    summarization_chain = langchain.chains.LLMChain(
        llm=cat._llm,
        verbose=False,
        prompt=langchain.PromptTemplate(template=summarization_prompt,
                                        input_variables=["text"]),
    )

    notification = f"Starting to summarize {len(docs)}",
    log(notification, "INFO")
    cat.send_ws_message(notification, msg_type="notification")

    # we will store iterative summaries all together in a list
    all_summaries = []

    # Compute total summaries for progress notification
    n_summaries = len(docs) // group_size

    # make summaries of groups of docs
    for n, i in enumerate(range(0, len(docs), group_size)):
        # Notify the admin of the progress
        progress = (n * 100) // n_summaries
        message = f"{progress}% of summarization"
        cat.send_ws_message(message, msg_type="notification")
        log(message, "INFO")

        # Get the text from groups of docs and join to string
        group = docs[i: i + group_size]
        group = list(map(lambda d: d.page_content, group))
        text_to_summarize = "\n".join(group)

        # Summarize and add metadata
        summary = summarization_chain.run(text_to_summarize)
        summary = Document(page_content=summary)
        summary.metadata["is_summary"] = True

        # add summary to list of all summaries
        all_summaries.append(summary)

    docs.extend(all_summaries)

    return docs
