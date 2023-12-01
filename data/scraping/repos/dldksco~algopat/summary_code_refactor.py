from langchain.chat_models import ChatOpenAI
from myclass.problem import ProblemSummary
from myclass.problem import ProblemData
from utils.utils import count_token, preprocessing_code
from chain.usercode.summary_code_refactor import summary_code_refactor_chain
from chain.usercode.summary_code_refactor_long import summary_code_refactor_long_chain
from langchain.text_splitter import TokenTextSplitter
from logging import getLogger

# logger 설정 
logger = getLogger()

async def summary_code_refactor(chat_llm : ChatOpenAI, data : ProblemData, problem_summary : ProblemSummary):
    problem_info = await build_problem_inout_info(problem_summary, data)
    preprocessed_code = await preprocessing_code(data.code, data.language)
    token_length = await count_token(preprocessed_code)
    if token_length < 2000:
        return await summary_code_refactor_short(chat_llm, preprocessed_code, problem_info)
    else :
        return await summary_code_refactor_long(chat_llm, preprocessed_code, problem_info)
    
    
async def summary_code_refactor_short(chat_llm : ChatOpenAI, code : str, problem_info : ProblemSummary):
    chain = await summary_code_refactor_chain(chat_llm)
    summary_code_refactor_result = await chain.arun(problem_info = problem_info, user_code = code)
    return summary_code_refactor_result

async def summary_code_refactor_long(chat_llm : ChatOpenAI, code : str, problem_info : ProblemSummary):
    chain = await summary_code_refactor_long_chain(chat_llm)
    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0, encoding_name="cl100k_base")
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
    
async def build_problem_inout_info(problem_summary : ProblemSummary, data : ProblemData):
    problem_info = f"\
        problem_input = {problem_summary.gpt_problem_summary_input}\n\
        problem_output = {problem_summary.gpt_problem_summary_output}\n\
        problem_constraints = {problem_summary.gpt_problem_summary_constraints}\n"
    return problem_info