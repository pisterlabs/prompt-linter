from .filter_questions import get_questions_from_filter, get_regex_list, filter_response, get_page_filter_blacklist, filter_page_by_any, get_system_prompt
from threading import Thread
import openai
import re
import backoff

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)

def get_question(text, questions, category):
    question_text = ""
    for i in range(len(questions)):
        question_text += f"\nQuestion {i+1}: {questions[i]}"

    category_prompt1, category_prompt2, category_prompt3 = get_system_prompt(category)

    return f"""
    {category_prompt1} Your responses should always be in a consistent format and follow a specific structure. It is important to note that you should only provide answers that can be found within the given context and never use a seperate source of data.\n\n
    {category_prompt2} Your responses should be clear, concise, and specific to the context given.\n\n
    {category.capitalize()} Document: "{text}"\n\n
    {question_text}\n\n
    Answer all of the questions in the following format, where X is the question number, include the # separator:
    \n\n
    # Answer X: [GPT will generate the answer here]
    \n\n
    Context X: [{category_prompt3}]
    """.lstrip()

def get_predictions(text_prompt):
    prediction = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages = [
            {"role": "system", "content": "You are a question and answering service named GPT."},
            {"role": "user", "content": text_prompt}
        ]
    )
    return prediction


def prediction_thread(text, category, filter, response_dict, custom_questions=None, price_calculation=False):
    if not custom_questions:
        questions = get_questions_from_filter(category, filter)
    else:
        questions = custom_questions
    response_dict[filter.capitalize()] = dict()
    if price_calculation:
        response_dict[filter.capitalize()] = 0
    for page_no in range(len(text)):
        page = re.sub(r'[^\w\s]+$', '', text[page_no]) + "."
        # Skip if the page doesn't contain any of the keywords
        if (not filter_page_by_any(page, get_regex_list(category, filter)) and category not in ["depreciation"]) or filter_page_by_any(page, get_page_filter_blacklist()):
            continue
        elif not price_calculation:
            text_prompt = get_question(page, questions, category)
            #token_amount = int(len(text_prompt) / 4) + 800
            prediction = get_predictions(text_prompt)
            prediction = prediction["choices"][0]["message"]["content"].lstrip("\n")
            filtered = filter_response(prediction, response_dict, filter)
            if filtered is True:
                continue
            else:
                prediction = filtered
            sections = prediction.split("# ")

            for i in range(len(sections)):
                try:
                    question_no = int(re.search(r'\d+:', sections[i]).group()[:-1])
                    sections[i] = f"\nQuestion {question_no}: "+ questions[question_no-1]  + "\n"+ sections[i]
                except:
                    continue

            prediction = "\n".join(sections)
            response_dict[filter.capitalize()][f"Page {page_no + 1}"] = prediction
            if category == "custom_question" and response_dict[filter.capitalize()] == dict():
                response_dict[filter.capitalize()] = {"N/A": "No answer found"}
        else:
            token_amount = len(get_question(page, questions)) / 4
            response_dict[filter.capitalize()] += token_amount


def ennumerate_custom_questions(questions):
    question_str = ""
    for i in range(len(questions[0])):
        question_str += f"{i+1}. {questions[0][i]}\n"
    question_str = question_str[:-1]
    return question_str


def predict_text(text, category, filters, price_calculation=False):
    response = dict()
    thread_list = list()
    for filter in filters:
        if type(filter) == dict:
            thread = Thread(target=prediction_thread, args=(text, category, f"Custom Questions:\n{ennumerate_custom_questions(list(filter.values()))}", response), kwargs={
                            "custom_questions": list(filter.values()), "price_calculation": price_calculation})
        else:
            thread = Thread(target=prediction_thread, args=(
                text, category, filter, response), kwargs={"price_calculation": price_calculation})
        thread.start()
        thread_list.append(thread)
    for thread in thread_list:
        thread.join()
    return response
