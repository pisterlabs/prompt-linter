import datetime
import time

import asyncio
from typing import List

import config
import google.cloud.bigquery as bq
import google.cloud.bigquery.dbapi as bqapi
import numpy as np
import pandas as pd
import re
import tqdm
import tqdm.asyncio as tqao
from asyncio import Semaphore
from contextlib import closing
from google.api_core.exceptions import InvalidArgument, ResourceExhausted, InternalServerError
from google.oauth2 import service_account
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.chat_models import ChatOpenAI, ChatVertexAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage
from openai.error import RateLimitError, Timeout, APIConnectionError, APIError
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split

tqdm.tqdm.pandas()


class AugmentedOutput(BaseModel):
    answerable: str = Field(description="The answerable question", default="FAILED")
    unanswerable: str = Field(description="The unanswerable question", default="FAILED")


class FilteredOutput(BaseModel):
    relevant: bool = Field(description="Whether the summary is relevant to investor", default=False)
    movement: bool = Field(description="Whether the summary talks about the movement of share prices, index, "
                                       "or information that can easily be extracted from a stock screener.",
                           default=False)


class AugmentedQA(BaseModel):
    answer: str = Field(description="the given answer", default="FAILED")
    answerable: str = Field(description="the given answerable question", default="FAILED")
    unanswerable: str = Field(description="the given unanswerable question", default="FAILED")


def create_full_prompt(system, user, examples):
    final_examples = []
    for e in examples:
        final_examples.append(HumanMessage(content=e["user"]))
        final_examples.append(AIMessage(content=e["assistant"]))
    chat_prompt = ChatPromptTemplate.from_messages([system, *final_examples, user])
    return chat_prompt


def create_summary_aug_chain(llm):
    system_prompt = SystemMessagePromptTemplate.from_template(config.SUMMARY_AUG_SYSTEM)
    user_prompt = HumanMessagePromptTemplate.from_template(config.SUMMARY_AUG_USER)
    chat_prompt = create_full_prompt(system_prompt, user_prompt, config.SUMMARY_AUG_EXAMPLES)
    augment_chain = LLMChain(llm=llm, prompt=chat_prompt)
    return augment_chain


def create_aug_parse_chain(llm):
    output_parser = PydanticOutputParser(pydantic_object=AugmentedOutput)
    system_prompt = SystemMessagePromptTemplate.from_template(config.SUMMARY_AUG_PARSE_SYSTEM)
    user_prompt = HumanMessagePromptTemplate.from_template(config.SUMMARY_AUG_PARSE_USER)
    chat_prompt = create_full_prompt(system_prompt, user_prompt, config.SUMMARY_AUG_PARSE_EXAMPLE)
    chat_prompt = chat_prompt.partial(format_instructions=output_parser.get_format_instructions())
    parse_chain = LLMChain(llm=llm, prompt=chat_prompt, output_parser=output_parser)
    return parse_chain


def create_rewrite_reltime_chain(llm):
    system_prompt = SystemMessagePromptTemplate.from_template(config.SUMMARY_REWRITE_SYSTEM)
    user_prompt = HumanMessagePromptTemplate.from_template(config.SUMMARY_REWRITE_USER)
    chat_prompt = create_full_prompt(system_prompt, user_prompt, config.SUMMARY_REWRITE_EXAMPLES)
    augment_chain = LLMChain(llm=llm, prompt=chat_prompt)
    return augment_chain


def create_summary_filter_parse_chain(llm):
    output_parser = PydanticOutputParser(pydantic_object=FilteredOutput)
    system_prompt = SystemMessagePromptTemplate.from_template(config.SUMMARY_AUG_PARSE_SYSTEM)
    user_prompt = HumanMessagePromptTemplate.from_template(config.SUMMARY_AUG_PARSE_USER)
    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
    chat_prompt = chat_prompt.partial(format_instructions=output_parser.get_format_instructions())
    parse_chain = LLMChain(llm=llm, prompt=chat_prompt, output_parser=output_parser)
    return parse_chain


def create_summary_filter_text_chain(llm):
    system_prompt = SystemMessagePromptTemplate.from_template(config.SUMMARY_FILTER_SYSTEM)
    user_prompt = HumanMessagePromptTemplate.from_template(config.SUMMARY_FILTER_USER)
    chat_prompt = create_full_prompt(system_prompt, user_prompt, config.SUMMARY_FILTER_EXAMPLE)
    filter_chain = LLMChain(llm=llm, prompt=chat_prompt)
    return filter_chain


def create_rewrite_headline_chain(llm):
    system_prompt = SystemMessagePromptTemplate.from_template(config.SUMMARY_SUMMARY_SYSTEM)
    user_prompt = HumanMessagePromptTemplate.from_template(config.SUMMARY_SUMMARY_USER)
    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
    augment_chain = LLMChain(llm=llm, prompt=chat_prompt)
    return augment_chain


def create_augment_qa_chain(llm):
    output_parser = PydanticOutputParser(pydantic_object=AugmentedQA)
    system_prompt = SystemMessagePromptTemplate.from_template(config.QA_AUG_SYSTEM)
    user_prompt = HumanMessagePromptTemplate.from_template(config.QA_AUG_USER)
    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
    system_parse_prompt = SystemMessagePromptTemplate.from_template(config.QA_PARSE_SYSTEM)
    user_parse_prompt = HumanMessagePromptTemplate.from_template(config.QA_PARSE_USER)
    parse_chat_prompt = create_full_prompt(system_parse_prompt, user_parse_prompt, config.QA_PARSE_EXAMPLE)
    parse_chat_prompt = parse_chat_prompt.partial(format_instructions=output_parser.get_format_instructions())
    augment_chain = {"text": chat_prompt | llm} | parse_chat_prompt | llm | output_parser
    return augment_chain


def get_llm(kind="vertexai"):
    if kind == "openai":
        with open(config.OPENAI_KEY_PATH) as fp:
            key = fp.read().strip()

        plan_llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=key,
                              temperature=0, max_tokens=1024, max_retries=1, request_timeout=10)
        return plan_llm
    elif kind == "vertexai":
        credentials = service_account.Credentials.from_service_account_file(config.VERTEX_AI_KEY_PATH)
        plan_llm = ChatVertexAI(temperature=config.SUMMARY_AUG_TEMPERATURE,
                                model_name=config.SUMMARY_AUG_MODEL,
                                project=config.VERTEX_AI_PROJECT,
                                credentials=credentials,
                                max_retries=1, request_timeout=10,
                                max_output_tokens=1024)
        return plan_llm
    else:
        raise NotImplemented()


async def async_retry_with_backoff(func, *params, limiter=None, start=1, factor=2, max_retry=6,
                                   exceptions=(Timeout, RateLimitError,
                                               APIConnectionError, APIError,
                                               ResourceExhausted, InternalServerError)):
    retry_time = 0
    while True:
        if retry_time > max_retry:
            raise ValueError()
        try:
            if limiter:
                async with limiter:
                    return await func(*params)
            else:
                return await func(*params)
        except exceptions:
            amount = start * factor ** retry_time
            delta = amount * 0.2
            noise = (amount - delta) + np.random.random() * delta * 2
            await asyncio.sleep(noise)


async def filter_summary_complete(summary, llm, limiter):
    filter_chain = create_summary_filter_text_chain(llm)
    parse = create_summary_filter_parse_chain(llm)
    combined = SimpleSequentialChain(chains=[filter_chain, parse])
    try:
        return await async_retry_with_backoff(combined.acall, summary, limiter=limiter)
    except (ValueError, InvalidArgument):
        return {"output": FilteredOutput()}


async def augment_summary_complete(text, plan_llm, limiter):
    augment = create_summary_aug_chain(plan_llm)
    parse = create_aug_parse_chain(plan_llm)
    combined = SimpleSequentialChain(chains=[augment, parse])
    try:
        return await async_retry_with_backoff(combined.acall, text, limiter=limiter)
    except (ValueError, InvalidArgument):
        return {"output": AugmentedOutput()}


async def rewrite_summary_time(title, text, date, llm, limiter):
    rewriter = create_rewrite_reltime_chain(llm)
    date_str = date.strftime('%Y-%m-%d, %A')
    try:
        return await async_retry_with_backoff(rewriter.acall,
                                              {"ipt_text": text, "title": title, "date": date_str}, limiter=limiter)
    except (ValueError, InvalidArgument):
        return {"text": "FAILED"}


async def summary_summary_headline(text, llm, limiter):
    rewriter = create_rewrite_headline_chain(llm)
    try:
        return await async_retry_with_backoff(rewriter.acall, {"summary": text}, limiter=limiter)
    except (ValueError, InvalidArgument):
        return {"text": "FAILED"}


async def augment_qa_complete(text, plan_llm, limiter):
    augment = create_augment_qa_chain(plan_llm)
    try:
        return await async_retry_with_backoff(augment.ainvoke, {"input_text": text}, limiter=limiter)
    except (ValueError, InvalidArgument):
        return {"output": AugmentedQA()}


async def filter_summary(df, max_request=25):
    plan_llm = get_llm()
    relevant = []
    stat = []
    limiter = Semaphore(value=max_request)
    flist = [filter_summary_complete(i, plan_llm, limiter) for i in df.summary.tolist()]
    for i, augmentation in enumerate(await tqao.tqdm_asyncio.gather(*flist)):
        output = augmentation["output"]
        relevant.append(output.relevant)
        stat.append(output.movement)
    result = df.copy()
    result["relevant"] = relevant
    result["stat"] = stat
    return result.loc[result.relevant & ~result.stat]


async def rewrite_summary(df, max_request=25):
    plan_llm = get_llm()
    errors = []
    summaries = []
    limiter = Semaphore(value=max_request)
    flist = [rewrite_summary_time(row["title"], row["summary"],
                                  row["published"], plan_llm, limiter) for _, row in
             df.iterrows()]
    for i, augmentation in enumerate(await tqao.tqdm_asyncio.gather(*flist)):
        output = augmentation["text"]
        if "please send us your feedback" in output.lower() or "as a large language model" in output.lower():
            errors.append(i)
        else:
            summaries.append(output)
    result = df.copy()
    if len(errors) > 0:
        result = result.drop(index=result.iloc[errors].index)
    result["summary"] = summaries
    return result


async def headline_summary(df, max_request=25):
    plan_llm = get_llm()
    errors = []
    summaries = []
    limiter = Semaphore(value=max_request)
    flist = [summary_summary_headline(row["summary"], plan_llm, limiter) for _, row in df.iterrows()]
    for i, augmentation in enumerate(await tqao.tqdm_asyncio.gather(*flist)):
        output = augmentation["text"]
        if "please send us your feedback" in output.lower() or "as a large language model" in output.lower():
            errors.append(i)
        else:
            summaries.append(output)
    result = df.copy()
    if len(errors) > 0:
        result = result.drop(index=result.iloc[errors].index)
    result["headline"] = summaries
    result["summary"] = result["headline"].str.strip() + "\n" + result.summary.str.strip()
    result = result.drop("headline", axis=1)
    return result


async def augment_summary(df, max_request=25):
    plan_llm = get_llm()
    questions = []
    reverse_questions = []
    errors = []
    limiter = Semaphore(value=max_request)
    flist = [augment_summary_complete(i, plan_llm, limiter) for i in df.summary.tolist()]
    for i, augmentation in enumerate(await tqao.tqdm_asyncio.gather(*flist)):
        output = augmentation["output"]
        if output.unanswerable == "FAILED" or output.answerable == "FAILED":
            errors.append(i)
        else:
            questions.append(output.answerable)
            reverse_questions.append(output.unanswerable)
    result = df.copy()
    if len(errors) > 0:
        result = result.drop(index=result.iloc[errors].index)
    result["question"] = questions
    result["reverse_question"] = reverse_questions
    return result


async def augment_articles_qa(df, max_request=25):
    plan_llm = get_llm()
    questions = []
    unanswerable = []
    answers = []
    errors = []
    limiter = Semaphore(value=max_request)
    flist = [augment_qa_complete(i, plan_llm, limiter) for i in df.context.tolist()]
    for i, augmentation in enumerate(await tqao.tqdm_asyncio.gather(*flist)):
        output = augmentation
        if not isinstance(output, AugmentedQA):
            output = AugmentedQA()
        if output.answer == "FAILED" or output.answerable == "FAILED" or output.unanswerable == "FAILED":
            errors.append(i)
        else:
            questions.append(output.answerable)
            unanswerable.append(output.unanswerable)
            answers.append(output.answer)
    result = df.copy()
    if len(errors) > 0:
        result = result.drop(index=result.iloc[errors].index)
    result["answerable"] = questions
    result["unanswerable"] = unanswerable
    result["answer"] = answers
    return result


def get_data_sets_df(sample_df, test_instances=1000, context_col="body", resp_col="summary", answerable_col="question",
                     unanswerable_col="reverse_question", final_question_col="question",
                     impossible_resp=config.SUMMARY_IMPOSSIBLE_RESP):
    sample_df = sample_df.sample(frac=1, random_state=93).reset_index(drop=True)
    clean_regex = re.compile(r"\*[\s\n]*(?=\*)")
    sample_df[resp_col] = sample_df[resp_col].apply(lambda s: clean_regex.sub(" ", s).strip())
    sample_df[resp_col] = sample_df[resp_col].str.strip()

    body = []
    published = []
    summary = []
    question = []
    pos = []
    for i, row in sample_df.iterrows():
        body.extend([row[context_col], row[context_col]])
        published.extend([row["published"], row["published"]])
        summary.extend([row[resp_col], impossible_resp])
        question.extend([row[answerable_col], row[unanswerable_col]])
        pos.extend([True, False])
    result_df = pd.DataFrame({
        context_col: body,
        "published": published,
        final_question_col: question,
        resp_col: summary,
        "pos": pos
    })

    train_df, test_df = train_test_split(result_df, test_size=test_instances, stratify=result_df.pos)
    return train_df[[context_col, "published", final_question_col, resp_col]], test_df[[context_col, "published",
                                                                                        final_question_col, resp_col]]


def get_full_data(client: bq.Client):
    query = "SELECT id, published, title, body, summary_type, summary FROM Articles.ScrapedArticles " \
            "WHERE summary_type = 'BULLETS'"
    with closing(bqapi.Connection(client=client)) as connection:
        with closing(connection.cursor()) as cursor:
            cursor.execute(query)
            results = [list(r) for r in cursor.fetchall()]
    return pd.DataFrame(results, columns=["id", "published", "title", "body", "summary_type", "summary"])


def create_body_chunks(row, chunks):
    published_date: datetime.datetime = row["published"]
    body_chunks = [f"Published: {published_date.strftime('%Y-%m-%d')}\n{c}" for c in chunks]
    return body_chunks


def get_random_chunks(body_chunks, amount, idx):
    result = []
    for _ in range(amount):
        alt_idx = np.random.choice(len(body_chunks))
        while alt_idx == idx:
            alt_idx = np.random.choice(len(body_chunks))

        alternative_article = body_chunks[alt_idx]
        alt_choice = np.random.choice(alternative_article)
        result.append(alt_choice)
    return result


def inject_noise(df: pd.DataFrame, splitter,
                 target_chunks=7, pure_noise=0.1,
                 context_col="body", result_col="summary",
                 impossible_resp=config.SUMMARY_IMPOSSIBLE_RESP,
                 chunk_processor=create_body_chunks):
    df = df.reset_index(drop=True)
    clean_newline_regex = re.compile(r"\n+")
    chunks = df[context_col].progress_apply(splitter.split_text)
    chunks: List[List[str]] = [[clean_newline_regex.sub("\n", c).strip() for c in chunk] for chunk in chunks]
    all_body_chunks = []
    for idx, row in df.iterrows():
        body_chunks = chunk_processor(row, chunks[idx])
        all_body_chunks.append(body_chunks)

    # Original articles but with shuffled chunks
    og_df = df.copy()
    og_bodies = []
    for b in all_body_chunks:
        shuffled = list(b)[:target_chunks]
        np.random.shuffle(shuffled)
        og_bodies.append("\n\n".join(shuffled))
    og_df[context_col] = og_bodies

    # Original articles + random other articles
    augmented_bodies = []
    aug_df = df.copy()
    for idx, row in aug_df.iterrows():
        num_chunks = len(all_body_chunks[idx])
        num_noise = target_chunks - num_chunks
        if num_noise <= 0:
            augmented_bodies.append("")
        else:
            noise_chunks = get_random_chunks(all_body_chunks, num_noise, idx)
            final_chunks = all_body_chunks[idx] + noise_chunks
            np.random.shuffle(final_chunks)
            augmented_bodies.append("\n\n".join(final_chunks))
    aug_df[context_col] = augmented_bodies
    aug_df = aug_df.loc[aug_df[context_col] != ""]

    # Pure Noise Chunks
    noise_df = df.sample(frac=pure_noise)
    noise_df[result_col] = impossible_resp
    noise_bodies = []
    for idx, row in noise_df.iterrows():
        chunk_amt = np.random.randint(1, target_chunks + 1)
        rand_chunks = get_random_chunks(all_body_chunks, chunk_amt, idx)
        noise_bodies.append("\n\n".join(rand_chunks))
    noise_df[context_col] = noise_bodies

    return pd.concat([og_df, aug_df, noise_df], ignore_index=True)


def fix_summary_tagline(df):
    break_regex = re.compile(r"\n\*")
    sums = df.summary.apply(break_regex.split)
    fixed_sum = []
    for s in sums:
        taglines = s[0].split("\n")
        if len(taglines) > 1:
            tagline = taglines[-1]
        else:
            tagline = taglines[0]
        key_points = [tagline] + [f"* {p}" for p in s[1:]]
        fixed_sum.append("\n".join(key_points))
    df = df.copy()
    df["summary"] = fixed_sum
    return df
