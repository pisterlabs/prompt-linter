import os
import openai
import re

def evaluate_conversation(messages):
    messages.append({"role": "system", "content":
                                 """
                                 Evaluate the user's logical thinking skills rigorously on a scale of 0 to 10 for each of the following categories. 
                                 Assign a 0 if there is insufficient information for evaluation or if the discussion is too brief to assess properly. 
                                 Score strictly, considering each point as high achievement.
                                 1. Clarity: Is the user expressing their ideas and opinions clearly and coherently? Is the language used easily understood by everyone involved?
                                 2. Relevance: Are the topics and arguments raised pertinent to the main subject or theme? Does the user stay focused on the issue at hand?
                                 3. Depth: Does the user engage in thoughtful analysis and exploration of the subject matter? Are the arguments supported by evidence or logical reasoning?
                                 4. Open-mindedness: Is the user willing to consider alternative viewpoints and acknowledge the merits of different perspectives?
                                 5. Respectful tone: Does the user treat the assistant with respect and avoid personal attacks or derogatory language?
                                 6. Active listening: Is the user actively listening and responding to assistant's ideas and concerns? Does the user ask clarifying questions or build upon each other's points?
                                 7. Constructive criticism: When the assistant disagrees, Does the user provide constructive feedback and avoid resorting to ad hominem attacks?
                                 8. Balance: Is there a fair distribution of speaking time among the participants? Are all voices and opinions given equal opportunity to be heard?
                                 9. Engagement: Is the user genuinely interested and invested in the conversation? Does the user contribute meaningfully to the discussion?
                                 10. Progress: Does the discussion lead to new insights, understanding, or solutions? Is there a sense of movement or development in the conversation?
                                 Return the score of each of these 10 points in this format.
                                    i.e)
                                    1. 2
                                    2. 3
                                    3. 2
                                    4. 3
                                    5. 4
                                    6. 3
                                    7. 4
                                    8. 2
                                    9. 4
                                    10. 3
                                 """
                                 })
    openai.api_key = os.environ["OPENAI_API_KEY"]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.1,
        max_tokens=256
    )
    text = response.choices[-1].message.content

    # print(text)
    scores = []
    for criterion in range(1, 11):
        score_match = re.search(rf"{criterion}\.\s(\d+)", text)
        if score_match:
            score = int(score_match.group(1))
            scores.append(score)
        else:
            scores.append(0)

    return scores
