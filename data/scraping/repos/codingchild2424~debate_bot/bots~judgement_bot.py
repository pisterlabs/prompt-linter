from modules.gpt_modules import gpt_call
from langchain.prompts import PromptTemplate

def debate_judgement(
        user_debate_history, 
        bot_debate_history
        ):
    
    if len(user_debate_history.split()) < 100:
        bot_response = "Under the 100 words, evaluation is not possible."
    else:
        judgement_prompt_preset = "\n".join([
            "!!Instruction!",
            "You are now the judge of this debate. Evaluate the debate according to the rules below.",
            "Rule 1. Summarize the debate as a whole and each said.",
            "Rule 2. Explain what was persuasive and what made the differnce between winning and losing.",
        ])

        judgement_prompt = "\n".join([
                judgement_prompt_preset,
                "User: " + user_debate_history,
                "Judgement must be logical with paragraphs.",
                "Do not show Rule",
                "Write judgement below.",
                "Judgement: "
                ])

        bot_response = gpt_call(judgement_prompt)
    
    return bot_response