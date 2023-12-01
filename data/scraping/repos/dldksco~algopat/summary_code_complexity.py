from langchain.chat_models import ChatOpenAI
from myclass.problem import ProblemSummary
from myclass.problem import ProblemData
from utils.utils import count_token, preprocessing_code
from chain.usercode.summary_code_complexity import summary_code_complexity_chain
from chain.usercode.summary_code_complexity_long import summary_code_complexity_long_chain
from langchain.text_splitter import TokenTextSplitter
from logging import getLogger

# logger 설정 
logger = getLogger()

async def summary_code_complexity(chat_llm : ChatOpenAI, data : ProblemData, problem_summary : ProblemSummary):
    problem_info = await build_problem_info(problem_summary, data)
    preprocessed_code = await preprocessing_code(data.code, data.language)
    token_length = await count_token(preprocessed_code)
    if token_length < 2000:
        return await summary_code_complexity_short(chat_llm, preprocessed_code, problem_info)
    else :
        return await summary_code_complexity_long(chat_llm, preprocessed_code, problem_info)
    
async def summary_code_complexity_short(chat_llm : ChatOpenAI, code : str, problem_info : ProblemSummary):
    chain = await summary_code_complexity_chain(chat_llm)
    summary_code_complexity_result = await chain.arun(problem_info = problem_info, user_code = code)
    return summary_code_complexity_result

async def summary_code_complexity_long(chat_llm : ChatOpenAI, code : str, problem_info : ProblemSummary):
    chain = await summary_code_complexity_long_chain(chat_llm)
    text_splitter = TokenTextSplitter(chunk_size=600, chunk_overlap=0, encoding_name="cl100k_base")
    codes = text_splitter.split_text(code)
    
    first_code=""
    second_code=""
    existing_result=""
    codes_len = len(codes)
    for i in range(codes_len):
        logger.info(f"iter_count = {i + 1}/{codes_len}, token: {await count_token(codes[i])}")
        if (i > 0):
            first_code = codes[i - 1]
        second_code = codes[i]
        existing_result = await chain.arun(
            problem_info = problem_info,
            existing_result = existing_result,
            first_code = first_code,
            second_code = second_code
        )
    result = existing_result
    return result
    
async def build_problem_info(problem_summary : ProblemSummary, data : ProblemData):
    problem_info = f"\
        problem_description = {problem_summary.gpt_problem_summary_description}\n\
        problem_input = {problem_summary.gpt_problem_summary_input}\n\
        problem_output = {problem_summary.gpt_problem_summary_output}\n\
        problem_constraints = {problem_summary.gpt_problem_summary_constraints}\n\
        problem_expected_time_complexity = {problem_summary.gpt_time_complexity}\n\
        problem_expected_time_complexity_reason = {problem_summary.gpt_time_complexity_reason}\n\
        problem_expected_space_complexity = {problem_summary.gpt_space_complexity}\n\
        problem_expected_space_complexity_reason = {problem_summary.gpt_space_complexity_reason}\n\
        problem_algorithm_type = {problem_summary.problem_algorithm_type}\n\
        problem_time_limit = {problem_summary.problem_time_limit}\n\
        user_runtime = {data.runtime}ms\n\
        problem_space_limit = {problem_summary.problem_space_limit}\n\
        user_memory = {data.memory}kb"
    return problem_info