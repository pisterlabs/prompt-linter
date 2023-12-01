import os

import openai
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import streamlit as st

from configs import OPENAI_GPT3, CLAUDE_INSTANT, CLAUDE_2, REVIEW_NUM_CAP, OPENAI_CAP

# --- OpenAI Stream Completion ---
openai.api_key  = st.secrets["OpenAI_API_KEY"]

def gpt_stream_completion(prompt, model=OPENAI_GPT3):
    system_prompt, user_prompt = prompt[0], prompt[1]
    messages = [
        {"role": "system", "content": system_prompt}, 
        {"role": "user", "content": user_prompt}, 
        ]
    
    stream = openai.ChatCompletion.create(
        model=model, 
        messages=messages, 
        temperature=0,
        stream=True)
    
    completing_content = ""
    for chunk in stream:
        chunk_content = chunk["choices"][0].get("delta", {}).get("content")
        if chunk_content: 
            completing_content += chunk_content
            st.markdown(completing_content)
    
    return

# --- Anthropic Completion ---
anthropic = Anthropic(
    api_key=st.secrets["Anthropic_API_KEY"],
)

def claude_stream_completion(prompt, model=CLAUDE_INSTANT): 
    messages = f"{HUMAN_PROMPT} {prompt[2]} {AI_PROMPT}"

    stream = anthropic.completions.create(
        model=model,
        max_tokens_to_sample=90000,
        prompt=messages,
        temperature=0,
        stream=True,
    )

    completing_content = ""
    for chunk in stream:
        chunk_content = chunk.completion
        if chunk_content: 
            completing_content += chunk_content
            st.markdown(completing_content)

    return

# --- Prompt Generation ---
def generate_prompt(prod_info, num_of_reviews, review_texts, user_position, analysis_focus, input_question):
    common_prompt_part1 = f"""
    你是一名资深的电商评价分析师。
    你的任务是根据用户提供的电商客户评价列表分析这款 {prod_info} 产品在淘宝平台上的最近{num_of_reviews}条产品评价。\n
    分析过程中的注意事项如下：
    """

    common_prompt_part2 = """
    2. 你的分析结果应该包括：
      a. 主要发现，作为总结段落简洁明了地列出主要发现；
      b. 具体分析，对每个主要发现进行详细地深入探讨；
      c. 优化建议，在分析结束后，请基于你的发现的给出一些改进建议。\n
    仅在必要时，对主要发现给出具体的客户评价内容作为证据，并注明评价日期。\n
    3. 请用 markdown 语法输出你的分析结果。
    """
    
    focus_to_prompt = {
        "⚙️ 产品功能": "产品的功能特性有关的评价内容，如：产品的主要功能，功能的实用性以及客户对功能的反应。",
        "💎 产品质量": "产品的质量问题有关的评价内容，如：产品的耐用性，质量稳定性，以及客户对产品质量的反馈。",
        "🎨 产品外观": "产品的外观设计有关的评价内容，如：产品的外观、颜色，形状，尺寸，以及客户对外观设计的反馈。",
        "🖐️ 使用体验": "产品使用体验有关的评价内容，如：产品的使用便利性，舒适性以及客户在使用过程中遇到的问题。",
        "💰 价格合理性": "产品价格有关的评价内容。如：产品的价格是否合理，价格与产品的价值是否匹配，以及客户对产品价格的反馈。",
        "💳 客户服务与下单体验": "客户服务有关的评价内容，如：在线客服人员的响应速度，服务质量，客服专业性，下单流程的便利性以及客户的其他反馈。",
        "📦 包装与物流": "产品包装和物流问题有关的评价内容，如：产品的包装是否完好、包装的外观设计、物流速度等，以及客户对包装和物流的反馈。",
    }

    position_to_prompt = {
        "👨🏻‍💻 电商运营": "电商运营经理的角度分析，注意任何可能影响产品销量和客户满意度的因素，如产品受欢迎程度、销售策略、下单体验、定价等，以及客户的反馈和建议。",
        "🤵🏻‍♂️ 电商客服": "电商客户服务经理的角度分析，关注如：在线客服人员的响应速度，服务质量，客服专业性，下单流程的便利性以及客户的其他反馈。不要对任何与客服或下单体验无关的评价内容进行分析总结，如产品功能、使用体验等内容不要总结。",
        "👩🏻‍🔬 产品研发": "产品研发经理的角度分析，关注客户对产品功能和设计上优缺点的反馈，如产品功能、用户体验、产品优化需求等，并给出完整的分析总结。不要对任何与产品和产品使用体验无关的评价内容进行分析总结，如客服态度等内容不要总结。",
        "👩🏻‍🔧 生产/质量控制": "生产和质量控制部门经理的角度分析，重点关注客户对产品质量的反馈，如产品质量问题、产品瑕疵等。",
        "✈️ 物流/供应链": "物流和供应链部门经理的角度分析，关注客户对产品包装和物流的反馈，如包装物流问题、物流速度和物流体验等。不要对任何与物流或包装无关的评价内容进行分析总结。",
    }

    if input_question: 
        # generate prompt according to the specific question user has asked
        system_prompt = f"""
        你是一名资深的电商评价分析师。
        你的任务是根据用户提供的电商客户评价列表分析这款 {prod_info} 产品在淘宝平台上的最近{num_of_reviews}条产品评价。\n
        并通过客户的评价内容，回答以下问题：{input_question}
        """
    else: 
        # generate prompt according to the user position and area of interests (focus analysis)
        # if both user position and focus analysis are selected by the user, generate prompt based on focus analysis
        if analysis_focus != "暂不选择":
            system_prompt = common_prompt_part1 \
                + """1. 本次分析不应该面面俱到，而要侧重点明确：a. 首先筛选出客户评价中与""" + focus_to_prompt[analysis_focus] \
                + "b. 然后再对筛选出来的相关内容进行分析总结。\n" + common_prompt_part2
        elif user_position != "暂不选择":
            system_prompt = common_prompt_part1 + "1. 本次分析不应该面面俱到，请你仅站在" + position_to_prompt[user_position] + common_prompt_part2
        else:
            system_prompt = common_prompt_part1 + "1. 请你从不同角度全面地对客户评价进行分类总结和分析，如产品的主要优点和缺点、产品的功能、外观、使用体验、定价、包装、客服服务、产品的质量以及任何其他客户反映的问题。" + common_prompt_part2

    user_prompt = f"\n评价列表：\n```{review_texts}```"
    complete_prompt = system_prompt + user_prompt

    return [system_prompt, user_prompt, complete_prompt]