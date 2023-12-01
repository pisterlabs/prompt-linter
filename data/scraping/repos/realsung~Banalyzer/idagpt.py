import idaapi 
import idautils
import idc
import requests
import openai
import time
import sys

openai.api_key = "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"
OPENAI_API_KEY = "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"

# 취약한 함수들의 리스트
VULNERABLE_FUNCTIONS = [
    'strcpy', 'strncpy', 'strcat', 'strncat', 'sprintf', 'vsprintf',
    'gets', 'memcpy', 'memmove', 'fread', 'free', 'system', 'malloc',
    'write', 'read', 'snprintf', 'printf', 'scanf', 'puts'
]

def decompile_function(func_ea):
    """주어진 주소의 함수를 디컴파일합니다."""
    decompiled_func = idaapi.decompile(func_ea)
    if decompiled_func is None or not decompiled_func:
        return None  # 디컴파일 실패
    return str(decompiled_func)


def find_main_in_text_section():
    """'.text' 섹션에서 main 함수를 찾습니다."""
    text_segm = idaapi.get_segm_by_name(".text")
    if not text_segm:
        return idaapi.BADADDR

    for ea in idautils.Functions(text_segm.start_ea, text_segm.end_ea):
        func_name = idc.get_func_name(ea)
        if func_name in ["main", "_main"]:
            return ea

    return idaapi.BADADDR


def first_start():
    """스크립트의 첫 번째 실행 단계를 처리합니다."""
    main_addr = find_main_in_text_section()
    if main_addr == idaapi.BADADDR:
        print("Main function not found.")
        return None

    print("Main Address: ", hex(main_addr))
    return main_addr


def send_code_to_gpt3(decompiled_codes):
    """
    디컴파일된 코드들을 GPT-4에게 전송하고 분석 결과를 받습니다.

    :param decompiled_codes: 분석할 디컴파일된 코드의 딕셔너리.
    :return: GPT-4의 분석 결과 또는 오류 발생 시 None.
    """

    # GPT-3에 보낼 프롬프트를 준비합니다.
    prompt = """
    너가 해야할 일들은 아래와 같아.
    1. 전반적인 바이너리의 동작과정, 기능에 대해서 서술해줘.
    2.코드들을 따라 취약점을 분석해줘 취약점이 발생할 경우 해당 코드를 출력하면서 상세히 취약점이 터지는 이유와 함께 원리또한 설명해줘 마지막으로 한글로 답변해줘:\n\n
    """
    for function_name, code in decompiled_codes.items():
        prompt += f"Function {function_name}:\n{code}\n\n"

    try:
        response = openai.ChatCompletion.create(
            #model="gpt-3.5-turbo-16k-0613",
            model="gpt-4-0613",
            messages=[
                {
                    "role": "system",
                    "content": "You are a skilled security analyst tasked with finding vulnerabilities in the provided code snippets."
                },
                {"role": "user", "content": prompt}
            ]
        )
        
        # Ensure the 'choices' list is not empty and retrieve the message content from the last choice.
        if response.choices and len(response.choices) > 0 and 'message' in response.choices[-1]:
            return response.choices[-1]['message']['content']
        else:
            print("No response choices were returned.")
            return None

    except openai.error.OpenAIError as e:
        print(f"An error occurred: {str(e)}")
        return None


def analyze_func_calls_unique(func_ea, decompiled_codes, visited_funcs=None, call_chain=None, max_depth=10):
    """함수 호출 구조를 분석하고 디컴파일된 코드를 저장합니다."""
    if visited_funcs is None:
        visited_funcs = set()
    if call_chain is None:
        call_chain = []

    func_name = idc.get_func_name(func_ea)
    new_call_chain = call_chain + [f"{func_name}({hex(func_ea)})"]


    if func_ea in visited_funcs:
        return
    visited_funcs.add(func_ea)

    if len(new_call_chain) > 1:
        print(" -> ".join(new_call_chain))

    # 함수를 디컴파일하고 결과를 저장합니다.
    decompiled_code = decompile_function(func_ea)
    if decompiled_code:
        decompiled_codes[func_name] = decompiled_code

    if len(new_call_chain) < max_depth:
        curr_addr = func_ea
        func_end = idc.find_func_end(func_ea)
        while curr_addr < func_end:
            if idc.print_insn_mnem(curr_addr) == "call":
                called_addr = idc.get_operand_value(curr_addr, 0)
                called_name = idc.get_func_name(called_addr)
                if called_name and not called_name.startswith('.'):
                    analyze_func_calls_unique(called_addr, decompiled_codes, visited_funcs, new_call_chain, max_depth)

            curr_addr = idc.next_head(curr_addr, func_end)



def main():
    """메인 실행 함수."""
    main_addr = first_start()
    if main_addr is not None:
        decompiled_codes = {}  # 디컴파일된 코드를 저장할 딕셔너리
        analyze_func_calls_unique(main_addr, decompiled_codes)

        # 일정 시간 기다린 후 GPT-4로 코드를 전송합니다.
        print("Waiting for 3 seconds before sending code to GPT-4...")
        time.sleep(10)

        analysis_result = send_code_to_gpt3(decompiled_codes)
        if analysis_result:
            print(f"Analysis result from GPT-4:\n{analysis_result}")

    

if __name__ == "__main__":
    main()
