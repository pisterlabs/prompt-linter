import json
import time
from llm import OpenAifunction, OpenaiChatMessage, get_response_openai_nonstream


async def generate_frqs(text):
    prompt = f"""
You are an educational expert who is tasked with writing open-ended, free-response questions (FRQs) that are geared towards assessing how well students have assimilated the CCSS.ELA-Literacy.W.4 common core standard. The standard is: 

"Draw evidence from literary or informational texts to support analysis, reflection, and research."

Given a text, you produce exactly 10 high-quality FRQs. A good quality FRQ has the following characteristics:

- Clarity: The question must be easily understood. Ambiguity interferes with accurate assessment.
- Alignment with Standard: Ensure the question necessitates drawing evidence from the text for analysis, reflection, or research.
- Age-Appropriateness: Tailor the complexity of the question to the cognitive level of the target age group.
- Analytical Depth: The question should prompt thoughtful analysis, not just a regurgitation of facts.
- Open-Endedness: Questions should allow for more than one valid answer to encourage independent thinking.
- Textual Scope: Make sure the question covers a significant portion of the text, ensuring comprehensive analysis.
- Language Complexity: Keep sentence structures and vocabulary in sync with the students' language skills.
- Bias-Free: Eliminate any cultural,social, or gender biases that may affect a student's ability to respond objectively.
- Action Verbs: Use specific action verbs that align with the cognitive domain you aim to test (analyze, compare, assess).
- Feasibility of Answer: Ensure that the text provides adequate information to answer the question sufficiently.

When you receive a text, you write 10 FRQs that are geared towards assessing this standard. You then use the function `add_frqs` to save your questions.
"""

    messages_for_openai = [
        OpenaiChatMessage(role="system", content=prompt),
        OpenaiChatMessage(
            role="user",
            content=f"""
TEXT: {text}
""",
        ),
    ]
    frq_names = [
        "frq_1",
        "frq_2",
        "frq_3",
        "frq_4",
        "frq_5",
        "frq_6",
        "frq_7",
        "frq_8",
        "frq_9",
        "frq_10",
    ]
    add_frqs_openai_function: OpenAifunction = {
        "name": "add_frqs",
        "description": "Add FRQs for the given text.",
        "parameters": {
            "type": "object",
            "properties": {
                "frq_1": {
                    "type": "string",
                    "description": "Your first FRQ.",
                },
                "frq_2": {
                    "type": "string",
                    "description": "Your second FRQ.",
                },
                "frq_3": {
                    "type": "string",
                    "description": "Your third FRQ.",
                },
                "frq_4": {
                    "type": "string",
                    "description": "Your fourth FRQ.",
                },
                "frq_5": {
                    "type": "string",
                    "description": "Your fifth FRQ.",
                },
                "frq_6": {
                    "type": "string",
                    "description": "Your sixth FRQ.",
                },
                "frq_7": {
                    "type": "string",
                    "description": "Your seventh FRQ.",
                },
                "frq_8": {
                    "type": "string",
                    "description": "Your eighth FRQ.",
                },
                "frq_9": {
                    "type": "string",
                    "description": "Your ninth FRQ.",
                },
                "frq_10": {
                    "type": "string",
                    "description": "Your tenth FRQ.",
                },
            },
            "required": [
                "frq_1", "frq_2", "frq_3", "frq_4", "frq_5", "frq_6", "frq_7", "frq_8", "frq_9", "frq_10", ] 
            },
    }

    start_time = time.time()
    arguments = await get_response_openai_nonstream(
        messages_for_openai,
        functions=[add_frqs_openai_function],
        function_name="add_frqs",
    )
    print(f"OpenAI response time: {time.time() - start_time}")
    frqs = [arguments[frq_name] for frq_name in frq_names]
    return frqs


async def assess_frq(frq, text):
    prompt = f"""
You are an educational expert who is tasked with assessing the quality of free-response questions (FRQs) that are geared towards assessing how well students have assimilated the CCSS.ELA-Literacy.W.4 common core standard. The standard is:

"Draw evidence from literary or informational texts to support analysis, reflection, and research."

Given a text and an FRQ, you assess the quality of the FRQ along the following criteria:

- Clarity: Is the question easily understood? Ambiguity interferes with accurate assessment (1 = Not Clear, 5 = Very Clear).
- Alignment with Standard: Does the question necessitate drawing evidence from the text for analysis, reflection, or research? (1 = Not Aligned, 5 = Very Aligned).
- Age-Appropriateness: Is the complexity of the question tailored to the cognitive level of the target age group? Does it broach topics that can easily be understood and talked about by fourth graders? (1 = Not Appropriate, 5 = Very Appropriate).
- Analytical Depth: Does the question prompt thoughtful analysis, not just a regurgitation of facts? (1 = Not Deep, 5 = Very Deep).
- Open-Endedness: Does the question allow for more than one valid answer to encourage independent thinking? (1 = Not Open-Ended, 5 = Very Open-Ended). 
- Textual Scope: Does the question cover a significant portion of the text, ensuring comprehensive analysis? (1 = Not Comprehensive, 5 = Very Comprehensive).
- Language Complexity: Are the sentence structures and vocabulary in sync with the students' language skills? (1 = Not Complex, 5 = Very Complex).
- Bias-free: Is the answer free from any cultural, social, or gender biases that may affect a student's ability to respond objectively? (1 = Not Bias-Free, 5 = Very Bias-Free).
- Action Verbs: Are specific action verbs used that align with the cognitive domain you aim to test (analyze, compare, assess)? (1 = Not Appropriate, 5 = Very Appropriate).
- Feasibility of Answer: Does the text provide adequate information to answer the question sufficiently? (1 = Not Feasible, 5 = Very Feasible).


When you receive a text and an FRQ, you assess the quality of the FRQ along the above criteria. For each criterium, you write a brief reasoning (no more than two sentences) about your thoughts, containing at least one positive AND one negative aspect.. Then you give a numerical score for that criterium. Finally, you use the function `add_assessment` to save your evaluation of the FRQ.
"""

    messages_for_openai = [
        OpenaiChatMessage(role="system", content=prompt),
        OpenaiChatMessage(
            role="user",
            content=f"""
TEXT: {text}

====================

FRQ: {frq}
""",
        ),
    ]

    add_assessment_openai_function: OpenAifunction = {
        "name": "add_assessment",
        "description": "Add an assessment for the given text and FRQ.",
        "parameters": {
            "type": "object",
            "properties": {
                "clarity_reasoning": {
                    "type": "string",
                    "description": "Your reasoning for the clarity score. Includes AT LEAST one positive AND one negative aspect.",
                },
                "clarity_score": {
                    "type": "number",
                    "description": "Your clarity score.",
                },
                "alignment_reasoning": {
                    "type": "string",
                    "description": "Your reasoning for the alignment score. Includes AT LEAST one positive AND one negative aspect.",
                },
                "alignment_score": {
                    "type": "number",
                    "description": "Your alignment score.",
                },
                "age_appropriateness_reasoning": {
                    "type": "string",
                    "description": "Your reasoning for the age-appropriateness score. Includes AT LEAST one positive AND one negative aspect.",
                },
                "age_appropriateness_score": {
                    "type": "number",
                    "description": "Your age-appropriateness score.",
                },
                "analytical_depth_reasoning": {
                    "type": "string",
                    "description": "Your reasoning for the analytical depth score. Includes AT LEAST one positive AND one negative aspect.",
                },
                "analytical_depth_score": {
                    "type": "number",
                    "description": "Your analytical depth score.",
                },
                "open_endedness_reasoning": {
                    "type": "string",
                    "description": "Your reasoning for the open-endedness score. Includes AT LEAST one positive AND one negative aspect.",
                },
                "open_endedness_score": {
                    "type": "number",
                    "description": "Your open-endedness score.",
                },
                "textual_scope_reasoning": {
                    "type": "string",
                    "description": "Your reasoning for the textual scope score. Includes AT LEAST one positive AND one negative aspect.",
                },
                "textual_scope_score": {
                    "type": "number",
                    "description": "Your textual scope score.",
                },
                "language_complexity_reasoning": {
                    "type": "string",
                    "description": "Your reasoning for the language complexity score. Includes AT LEAST one positive AND one negative aspect.",
                },
                "language_complexity_score": {
                    "type": "number",
                    "description": "Your language complexity score.",
                },
                "bias_free_reasoning": {
                    "type": "string",
                    "description": "Your reasoning for the bias-free score. Includes AT LEAST one positive AND one negative aspect.",
                },
                "bias_free_score": {
                    "type": "number",
                    "description": "Your bias-free score.",
                },
                "action_verbs_reasoning": {
                    "type": "string",
                    "description": "Your reasoning for the action verbs score. Includes AT LEAST one positive AND one negative aspect.",
                },
                "action_verbs_score": {
                    "type": "number",
                    "description": "Your action verbs score.",
                },
                "feasibility_of_answer_reasoning": {
                    "type": "string",
                    "description": "Your reasoning for the feasibility of answer score. Includes AT LEAST one positive AND one negative aspect.",
                },
                "feasibility_of_answer_score": {
                    "type": "number",
                    "description": "Your feasibility of answer score.",
                },
            },
            "required": [
                "clarity_reasoning",
                "clarity_score",
                "alignment_reasoning",
                "alignment_score",
                "age_appropriateness_reasoning",
                "age_appropriateness_score",
                "analytical_depth_reasoning",
                "analytical_depth_score",
                "open_endedness_reasoning",
                "open_endedness_score",
                "textual_scope_reasoning",
                "textual_scope_score",
                "language_complexity_reasoning",
                "language_complexity_score",
                "bias_free_reasoning",
                "bias_free_score",
                "action_verbs_reasoning",
                "action_verbs_score",
                "feasibility_of_answer_reasoning",
                "feasibility_of_answer_score",
            ],
        },
    }

    print("Sending to OpenAI")
    start_time = time.time()
    arguments = await get_response_openai_nonstream(
        messages_for_openai,
        functions=[add_assessment_openai_function],
        function_name="add_assessment",
    )
    print(f"OpenAI response time: {time.time() - start_time}")
    # arguments = json.loads(response["arguments"])
    arguments["frq"] = frq
    
    
    return arguments


def select_best_frq(frq_rankings: list[dict]):
    for frq in frq_rankings:
        frq["score"] = (
            frq["feasibility_of_answer_score"] * 2
            + frq["alignment_score"] * 2
            + frq["clarity_score"]
            + frq["age_appropriateness_score"]
            + frq["analytical_depth_score"]
            + frq["open_endedness_score"]
            + frq["textual_scope_score"]
            + frq["language_complexity_score"]
            + frq["action_verbs_score"]
        ) / 11
    # Remove any frq for which the bias_free_score is less than 5
    frq_rankings = [
        frq_ranking
        for frq_ranking in frq_rankings
        if frq_ranking["bias_free_score"] >= 5
    ]



    frq_rankings = sorted(
        frq_rankings,
        key=lambda frq_ranking: frq_ranking["score"],
        reverse=True,
    )

    return frq_rankings[0]

if __name__ == "__main__":
    import asyncio
    import sys

    text = "A baseball uniform is a type of uniform worn by baseball players, and by some non-playing personnel, such as field managers and coaches. It is worn to indicate the person's role in the game and\u2014through the use of logos, colors, and numbers\u2014to identify the teams and their players, managers, and coaches.Traditionally, home uniforms display the team name on the front, while away uniforms display the team's home location. In modern times, however, exceptions to this pattern have become common, with teams using their team name on both uniforms. Most teams also have one or more alternate uniforms, usually consisting of the primary or secondary team color on the vest instead of the usual white or gray. In the past few decades throwback uniforms have become popular.The New York Knickerbockers were the first baseball team to use uniforms, taking the field on April 4, 1849, in pants made of blue wool, white flannel shirts (jerseys) and straw hats. Caps and other types of headgear have been a part of baseball uniforms from the beginning. Baseball teams often wore full-brimmed straw hats or no cap at all since there was no official rule regarding headgear. Under the 1882 uniform rules, players on the same team wore uniforms of different colors and patterns that indicated which position they played. This rule was soon abandoned as impractical.In the late 1880s, Detroit and Washington of the National League and Brooklyn of the American Association were the first to wear striped uniforms. By the end of the 19th century, teams began the practice of having two different uniforms, one for when they played at home in their own baseball stadium and a different one for when they played away (on the road) at the other team's ballpark. It became common to wear white pants with a white color vest at home and gray pants with a gray or solid (dark) colored vest when away. By 1900, both home and away uniforms were standard across the major leagues.In June 2021, MLB announced a long-term deal with cryptocurrency exchange FTX, which includes the FTX logo appearing on umpire uniforms during all games. FTX is MLB's first-ever umpire uniform patch partner. On November 11, 2022, FTX filed for Chapter 11 bankruptcy protection. MLB removed the FTX patches from umpires' uniforms before the 2023 season."

    async def main():
        print(f"Generating FRQs.")

        frqs = await generate_frqs(text)

        print(f"FRQs: {frqs}")

        print(f"Assessing FRQs.")

        # Rank them in parallel

        start_time = time.time()
        frq_rankings = await asyncio.gather(*[assess_frq(frq, text) for frq in frqs])
        print(f"Assessment time: {time.time() - start_time}")
        with open("frq_rankings.json", "w") as f:
            json.dump(frq_rankings, f, indent=4)

        best_frq = select_best_frq(frq_rankings)
        print(f"Best FRQ: {best_frq}")
    asyncio.run(main())
