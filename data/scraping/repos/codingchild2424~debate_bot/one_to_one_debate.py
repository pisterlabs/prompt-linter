import re
import random
from langchain.prompts import PromptTemplate
from modules.gpt_modules import gpt_call

def erase_start_word_and_after(text, start_word):
    pattern = re.compile(re.escape(start_word) + '.*')
    return re.sub(pattern, '', text)

def one_to_one_debator(prompt, history, debate_subject, bot_role, history_num):
    # Debate Rule 설명하기
    if history_num == 0:
        print("history_num", history_num)

        user_role = ""
        bot_response = ""

        debate_role = [
            "pro side", 
            "con side"
        ]

        # user role random으로 정하기
        user_debate_role = random.choice(debate_role)
        # user role이 아닌 것이 bot의 role임
        bot_debate_role_list = [role for role in debate_role if role != user_debate_role]

        print("user_debate_role", user_debate_role)
        print("bot_debate_role_list", bot_debate_role_list)

        debate_preset = "\n".join([
            "Debate Rules: ",
            "1) This debate will be divided into two teams, pro and con, with two debates on each team.",
            "2) The order of speaking is: first debater for the pro side, first debater for the con side, second debater for the pro side, second debater for the con side.",
            "3) Answer logically with an introduction, body, and conclusion.\n", #add this one.
            "User debate role: " + user_debate_role,
            "Bot debate roles: " + ", ".join(bot_debate_role_list) + "\n",
            "Debate subject: " + debate_subject
        ])

        # User가 첫번째 차례라면, User에게 먼저 prompt를 받아야 함
        if user_debate_role == debate_role[0]:
            #print("user_debate_role", user_debate_role)
            bot_preset = "\n".join([
                debate_preset + "\n",
                "It's your turn! Write your opinion!"
            ])
            bot_response = bot_preset
            print("bot_response", bot_response)
            #return bot_response
        
        # User가 두번째 차례라면, Bot이 1번째 차례에 대한 response를 만들고, 사용자의 답변을 받아야 함
        elif user_debate_role == debate_role[1]:

            bot_preset = "\n".join([
                debate_preset,
            ])

            first_prompt_template = PromptTemplate(
                input_variables=["prompt"],
                template="\n".join([
                    bot_preset, #persona
                    "{prompt}",
                    "Only say " + debate_role[0] + "\'s opinion after \':\'. Do not write " + debate_role[1] + "\'s " + "opinions, " + debate_role[2] + "\'s " + "opinions and " + debate_role[3] + "\'s " + "opinions.",
                    debate_role[0] + ": "
                    ])
            )
            first_bot_prompt = first_prompt_template.format(
                prompt=""
            )
            first_response = gpt_call(first_bot_prompt)

            # preprocess
            # if first_response contain the first debater for the con side's opinion, remove it.
            first_response = erase_start_word_and_after(first_response, debate_role[1])
            first_response = erase_start_word_and_after(first_response, debate_role[2])
            first_response = erase_start_word_and_after(first_response, debate_role[3])

            #first_response = re.sub(debate_role[1] + ":.*", "", first_response)

            bot_response = "\n".join([
                bot_preset + "\n",
                "-----------------------------------------------------------------",
                "[First debater for the pro side]: " + "\n" + first_response + "\n",
                "-----------------------------------------------------------------",
                "It's your turn! Write your opinion!"
            ])

        # User가 세번째 차례라면, Bot이 1, 2번째 차례에 대한 response를 만들고, 사용자의 답변을 받아야 함
        elif user_debate_role == debate_role[2]:

            bot_preset = "\n".join([
                debate_preset,
            ])
            # first
            first_prompt_template = PromptTemplate(
                input_variables=["prompt"],
                template="\n".join([
                    bot_preset, #persona
                    "{prompt}",
                    debate_role[0] + ": ",
                    ])
            )
            first_bot_prompt = first_prompt_template.format(
                prompt=""
            )
            first_response = gpt_call(first_bot_prompt)

            # second
            second_prompt_template = PromptTemplate(
                input_variables=["first_prompt"],
                template="\n".join([
                    bot_preset, #persona
                    "Only say " + debate_role[1] + "\'s opinion after \':\'. Do not write " + debate_role[0] + "\'s " + "opinions, " + debate_role[2] + "\'s " + "opinions and " + debate_role[3] + "\'s " + "opinions.",
                    debate_role[0] + ": " + "{first_prompt}",
                    debate_role[1] + ": "
                    ])
            )
            second_bot_prompt = second_prompt_template.format(
                first_prompt=first_response
            )
            second_response = gpt_call(second_bot_prompt)

            # preprocess
            # if first_response contain the first debater for the con side's opinion, remove it.
            first_response = erase_start_word_and_after(first_response, debate_role[1])
            first_response = erase_start_word_and_after(first_response, debate_role[2])
            first_response = erase_start_word_and_after(first_response, debate_role[3])
            # if second_response contain the first debater for the con side's opinion, remove it.
            #second_response = re.sub(debate_role[2] + ":.*", "", second_response)
            second_response = erase_start_word_and_after(second_response, debate_role[2])
            second_response = erase_start_word_and_after(second_response, debate_role[3])

            bot_response = "\n".join([
                bot_preset + "\n",
                "-----------------------------------------------------------------",
                "[First debater for the pro side]: " + "\n" + first_response + "\n",
                "-----------------------------------------------------------------",
                "[First debater for the con side]: " + "\n" + second_response + "\n",
                "-----------------------------------------------------------------",
                "It's your turn! Write your opinion!"
            ])


        elif user_debate_role == debate_role[3]:

            bot_preset = "\n".join([
                debate_preset,
            ])

            # first
            first_prompt_template = PromptTemplate(
                input_variables=["prompt"],
                template="\n".join([
                    bot_preset, #persona
                    "{prompt}",
                    debate_role[0] + ": ",
                    ])
            )
            first_bot_prompt = first_prompt_template.format(
                prompt=""
            )
            first_response = gpt_call(first_bot_prompt)

            # second
            second_prompt_template = PromptTemplate(
                input_variables=["first_prompt"],
                template="\n".join([
                    bot_preset, #persona
                    "Only say " + debate_role[1] + "'s opinion after \':\'. Do not write " + debate_role[0] + "\'s " + "opinions, " + debate_role[2] + "\'s " + "opinions and " + debate_role[3] + "\'s " + "opinions.",
                    debate_role[0] + ": " + "{first_prompt}",
                    debate_role[1] + ": "
                    ])
            )
            second_bot_prompt = second_prompt_template.format(
                first_prompt=first_response
            )
            second_response = gpt_call(second_bot_prompt)

            # third
            third_prompt_template = PromptTemplate(
                input_variables=["first_prompt", "second_prompt"],
                template="\n".join([
                    bot_preset, #persona
                    "Only say " + debate_role[2] + "\'s opinion after \':\'. Do not write " + debate_role[0] + "\'s " + "opinions, " + debate_role[1] + "\'s " + "opinions and " + debate_role[3] + "\'s " + "opinions.",
                    debate_role[0] + ": " + "{first_prompt}",
                    debate_role[1] + ": " + "{second_prompt}",
                    debate_role[2] + ": "
                    ])
            )
            third_bot_prompt = third_prompt_template.format(
                first_prompt=first_response,
                second_prompt=second_response
            )
            third_response = gpt_call(third_bot_prompt)

            # preprocess
            # if first_response contain the first debater for the con side's opinion, remove it.
            first_response = erase_start_word_and_after(first_response, debate_role[1])
            first_response = erase_start_word_and_after(first_response, debate_role[2])
            first_response = erase_start_word_and_after(first_response, debate_role[3])
            # if second_response contain the first debater for the con side's opinion, remove it.
            #second_response = re.sub(debate_role[2] + ":.*", "", second_response)
            second_response = erase_start_word_and_after(second_response, debate_role[2])
            second_response = erase_start_word_and_after(second_response, debate_role[3])
            # if third_response contain the first debater for the con side's opinion, remove it.
            thir_response = erase_start_word_and_after(thir_response, debate_role[3])
            #third_response = re.sub(debate_role[3] + ":.*", "", third_response)

            bot_response = "\n".join([
                bot_preset + "\n",
                "-----------------------------------------------------------------",
                "[First debater for the pro side]: " + "\n" + first_response + "\n",
                "-----------------------------------------------------------------",
                "[First debater for the con side]: " + "\n" + second_response + "\n",
                "-----------------------------------------------------------------",
                "[Second debater for the pro side]: " + "\n" + third_response + "\n",
                "-----------------------------------------------------------------",
                "It's your turn! Write your opinion!"
            ])
        else:
            pass

    # Answer and Ask Judgement.
    if history_num == 1:

        debate_role = [
            "first debater for the pro side", 
            "first debater for the con side", 
            "second debater for the pro side",
            "second debater for the con side"
        ]

        print("history1: ", history)

        # user가 가장 첫번째로 답변했다면, 봇이 2, 3, 4 답변을 하고, 평가할지를 물어보면 됨.
        if "User debate role: first debater for the pro side" in history:

            # second
            second_prompt_template = PromptTemplate(
                input_variables=["prompt"],
                template="\n".join([
                    history,
                    "User: {prompt}",
                    debate_role[2] + ": "
                    ])
            )
            second_bot_prompt = second_prompt_template.format(
                prompt=prompt
            )
            second_response = gpt_call(second_bot_prompt)


            # third
            third_prompt_template = PromptTemplate(
                input_variables=["prompt"],
                template="\n".join([
                    history,
                    "User: {prompt}",
                    debate_role[2] + ": "
                    ])
            )
            third_bot_prompt = third_prompt_template.format(
                prompt=prompt
            )
            third_response = gpt_call(third_bot_prompt)

            # fourth
            fourth_prompt_template = PromptTemplate(
                input_variables=["prompt"],
                template="\n".join([
                    history,
                    "User: {prompt}",
                    debate_role[3] + ": "
                    ])
            )
            fourth_bot_prompt = fourth_prompt_template.format(
                prompt=prompt
            )
            fourth_response = gpt_call(fourth_bot_prompt)

            ask_judgement = "Do you want to be the judge of this debate? (If you want, enter any words.)"
            bot_response = "\n".join([
                "[first debater for the con side]: " + "\n" +  second_response + "\n",
                "-----------------------------------------------------------------",
                "[second debater for the pro sid]: " + "\n" +  third_response + "\n",
                "-----------------------------------------------------------------",
                "[second debater for the con side]: " + "\n" +  fourth_response + "\n",
                "-----------------------------------------------------------------",
                ask_judgement
            ])

        # user가 두번째로 답변했다면, 봇이 3, 4 번째 답변을 하고, 평가할지를 물어보면 됨.
        elif "User debate role: first debater for the con side" in history:

            # third
            third_prompt_template = PromptTemplate(
                input_variables=["prompt"],
                template="\n".join([
                    history,
                    "User: {prompt}",
                    debate_role[2] + ": "
                    ])
            )
            third_bot_prompt = third_prompt_template.format(
                prompt=prompt
            )
            third_response = gpt_call(third_bot_prompt)

            # fourth
            fourth_prompt_template = PromptTemplate(
                input_variables=["prompt"],
                template="\n".join([
                    history,
                    "User: {prompt}",
                    debate_role[2] + ": " + third_response,
                    debate_role[3] + ": "
                    ])
            )
            fourth_bot_prompt = fourth_prompt_template.format(
                prompt=prompt
            )
            fourth_response = gpt_call(fourth_bot_prompt)

            # ask_judgement
            ask_judgement = "Do you want to be the judge of this debate? (If you want, enter any words.)"
            bot_response = "\n".join([
                "[second debater for the pro sid]: " + "\n" +  third_response + "\n",
                "-----------------------------------------------------------------",
                "[second debater for the con side]: " + "\n" +  fourth_response + "\n",
                "-----------------------------------------------------------------",
                ask_judgement
            ])

        # user가 세번째로 답변했다면, 봇이 4 번째 답변을 하고, 평가할지를 물어보면 됨.
        elif "User debate role: second debater for the pro side" in history:

            fourth_prompt_template = PromptTemplate(
                input_variables=["prompt"],
                template="\n".join([
                    history,
                    "User: {prompt}",
                    debate_role[3] + ": "
                    ])
            )
            fourth_bot_prompt = fourth_prompt_template.format(
                prompt=prompt
            )
            fourth_response = gpt_call(fourth_bot_prompt)



            ask_judgement = "Do you want to be the judge of this debate? (If you want, enter any words.)"
            bot_response = "\n".join([
                "[second debater for the con side]: " + "\n" + fourth_response + "\n",
                "-----------------------------------------------------------------",
                ask_judgement
            ])

        # user가 네번째로 답변했다면, 바로 평가할지를 물어보면 됨.
        elif "User debate role: second debater for the con side" in history:
            ask_judgement = "Do you want to be the judge of this debate? (If you want, enter any words.)"
            bot_response = ask_judgement
        else:
            pass

    # Judgement.
    if history_num == 2:
        judgement_word_list = "\n".join([
            "!!Instruction!",
            "You are now the judge of this debate. Evaluate the debate according to the rules below.",
            "Rule 1. Decide between the pro and con teams.",
            "Rule 2. Summarize the debate as a whole and what each debater said.",
            "Rule 3. For each debater, explain what was persuasive and what made the differnce between winning and losing.",
        ])

        judgement_prompt_template = PromptTemplate(
            input_variables=["prompt"],
            template="\n".join([
                history,
                "{prompt}",
                judgement_word_list,
                "Judgement: "
                ])
        )
        judgement_bot_prompt = judgement_prompt_template.format(
                prompt=""
        )
        judgement_response = gpt_call(judgement_bot_prompt)

        bot_response = "\n".join([
                "[Judgement]: " + "\n" + judgement_response + "\n",
            ])
        
    return bot_response