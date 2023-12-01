import os
import json
from openai import OpenAI
from openai._exceptions import OpenAIError, AuthenticationError, RateLimitError

def get_satisfaction_scores(topQuestionAnswers):
    aggregated_data = []
    
    client = OpenAI(api_key=os.environ.get("OPEN_AI_KEY"))

    for qna in topQuestionAnswers:
        question = qna['question']
        answers = qna['answers']
        times_asked = qna['times_asked']
        try: 
            response = client.chat.completions.create(
                        model="gpt-3.5-turbo-0613",
                        messages=[{
                            'role': 'system',
                            'content': """You will be provided with a question and array of answers. Please rate the answer on a scale of 1 to 5, with 1 being the lowest and 5 being the highest. The basis of your rating should be how well the answer addresses the question. Output should only be in valid JSON format. AS:
                            { question: "", 
                              answers: [{
                                    answer: "",
                                    satisfaction_score:  int
                                }],
                            },
                            """
                            },
                            {
                            'role': 'user',
                            'content': f"Q: {question}\nA: {answers}\nSatisfaction score: Int \n Justification: String"
                            }
                        ],
                        temperature=0.8,
                        max_tokens=1000,
                    
                    )
            # convert response to json
            formatted_response = json.loads(response.choices[0].message.content)

            # add times_asked to formatted_response
            formatted_response['times_asked'] = times_asked

            aggregated_data.append(formatted_response)
          # handle multiple exceptions at once
        except (OpenAIError, AuthenticationError, RateLimitError) as e:
            print(e)
            return {
                'status_code': 500,
                'error': 'Something went wrong. Please try again later.'
            }
        except Exception as e:
            print(e)
            formatted_response = {
                'question': question,
                'answers': answers,
                'times_asked': times_asked,
            }         
            aggregated_data.append(formatted_response)
    return aggregated_data

def get_top_negative_score_questions(qna_list, limit=10):
    """Return the top 10 negative score questions."""
    negative_score_questions = []

    for question in qna_list:
        score = 0

        if question.get("unanswered") or not question.get("answer"):
            score -= 2.5

        if isinstance(question.get('answer_vote'), int) and question.get('answer_vote') < 0:
            score -= 1

        question["score"] = score

        if score < 0:
            negative_score_questions.append(question)

    negative_score_questions.sort(key=lambda row: (row['score'], row['created_at']))

    return negative_score_questions[:limit]
