from firebase_admin import credentials, firestore
import firebase_admin
from langchain.chat_models import ChatOpenAI
import openai
from retry import retry
import json
from tqdm import tqdm
import time

MODEL = 'gpt-4'
ANSWER_PROMPT_PREFIX = '''
You are a Chartered Financial Analyst (CFA) Level 1 expert who generates step by step instructions to solve CFA practice problems.
If the question references an Exhibit, assume the data provided under 'data' is the Exhibit.
When given a question, generate a step by step solution to the problem in the following format:
QUESTION: json structured representation of the question
ANSWER: json structured representation of the answer

Please use latex when necessary (do not forget!). For example, if the answer is 1/2, you should write \\\\frac{1}{2}.

EXAMPLE 1:
QUESTION: {'options': {'A': '1.5%', 'C': '3%', 'B': '2.5%'}, 'question': 'In a typical year, 5% of all CEOs are fired for “performance” reasons. Assume that CEO performance is judged according to stock performance and that 50% of stocks have above-average returns or “good” performance. Empirically, 30% of all CEOs who were fired had “good” performance. Using Bayes’ formula, what is the probability that a CEO will be fired given “good” performance?', 'data': ''}
ANSWER:{
  "answer": "C",
  "solution_steps": {
    "step 1": "Identify the given probabilities: $P(F) = 0.05$, $P(G) = 0.5$, $P(G|F) = 0.3$",
    "step 2": "Apply the Bayes' formula: $P(F | G) = \\\\frac{{P(G | F) \\\\cdot P(F)}}{{P(G)}}$",
    "step 3": "Substitute the known values into the formula: $P(F | G) = \\\\frac{{(0.3 \\\\cdot 0.05)}}{0.5}$",
    "step 4": "Calculate the result: $P(F | G) = 0.03$ or $3\\\\%$"
  }
}

EXAMPLE 2:
QUESTION:{'options': {'A': 'f(x)', 'C': 'h(x)', 'B': 'g(x)'}, 'question': 'The conditions for a probability function are satisfied by:', 'data': {'table': {'rows': [{'f(x) = P(X = x)': '-0.25', 'h(x) = P(X = x)': '0.20', 'X = x': '1', 'g(x) = P(X = x)': '0.20'}, {'f(x) = P(X = x)': '0.25', 'h(x) = P(X = x)': '0.25', 'X = x': '2', 'g(x) = P(X = x)': '0.25'}, {'f(x) = P(X = x)': '0.50', 'h(x) = P(X = x)': '0.30', 'X = x': '3', 'g(x) = P(X = x)': '0.50'}, {'f(x) = P(X = x)': '0.25', 'h(x) = P(X = x)': '0.35', 'X = x': '4', 'g(x) = P(X = x)': '0.05'}], 'headers': ['X = x', 'f(x) = P(X = x)', 'g(x) = P(X = x)', 'h(x) = P(X = x)']}}}
ANSWER:{
    "answer": "B",
    "solution_steps": {
        "step 1": "Identify the given probabilities for each function: $f(x), g(x), h(x)$.",
        "step 2": "Check each function to ensure all probabilities are in the range $[0,1]$.",
        "step 3": "Calculate the sum of probabilities for each function: $f(x) = 0.75$, $g(x) = 1.00$, $h(x) = 1.10$.",
        "step 4": "Determine that $g(x)$ is the valid probability function as the sum is $1$ and all probabilities are in the range $[0,1]$."
    }
}

EXAMPLE 3:
QUESTION: {'options': {'A': 't-test', 'C': 'Paired comparisons test', 'B': 'F-test'}, 'question': 'Which of the following should be used to test the difference between the variances of two normally distributed populations?', 'data': ''}
ANSWER:{
    "answer": "B",
    "solution_steps": {
        "step 1": "Understand that the question is asking about the test used to compare variances between two normally distributed populations.",
        "step 2": "Identify that the F-test is used to compare the variances of two populations.",
        "step 3": "Recognize that a t-test and paired comparisons tests are typically used for comparing means, not variances."
    }
}'''

ANSWER_PROMPT = '''
QUESTION: {question}
ANSWER:'''

MC_PROMPT_PREFIX = '''
You're job is to identify whether or not submitted questions are multiple choice questions or not.
Questions will be provided in json format and your response should be true or false.

EXAMPLE 1:
QUESTION: {'options': {'A': '1.5%', 'C': '3%', 'B': '2.5%'}, 'question': 'In a typical year, 5% of all CEOs are fired for “performance” reasons. Assume that CEO performance is judged according to stock performance and that 50% of stocks have above-average returns or “good” performance. Empirically, 30% of all CEOs who were fired had “good” performance. Using Bayes’ formula, what is the probability that a CEO will be fired given “good” performance?', 'data': ''}
ANSWER: True
EXAMPLE 2:
QUESTION: {'options': {'A': 'Describe the distinct possible outcomes for terminal put value. (Think of the put’s maximum and minimum values and its minimum price increments.)', 'C': 'Letting Y stand for terminal put value, express in standard notation the probability that terminal put value is less than or equal to $24. No calculations or formulas are necessary.', 'B': 'Is terminal put value, at a time before maturity, a discrete or continuous random variable?'}, 'question': 'A European put option on stock conveys the right to sell the stock at a prespecified price, called the exercise price, at the maturity date of the option. The value of this put at maturity is (exercise price – stock price) or $0, whichever is greater. Suppose the exercise price is $100 and the underlying stock trades in increments of $0.01. At any time before maturity, the terminal value of the put is a random variable.', 'data': ''}
ANSWER: False
EXAMPLE 3:
QUESTION:{'options': {'A': 'f(x)', 'C': 'h(x)', 'B': 'g(x)'}, 'question': 'The conditions for a probability function are satisfied by:', 'data': {'table': {'rows': [{'f(x) = P(X = x)': '-0.25', 'h(x) = P(X = x)': '0.20', 'X = x': '1', 'g(x) = P(X = x)': '0.20'}, {'f(x) = P(X = x)': '0.25', 'h(x) = P(X = x)': '0.25', 'X = x': '2', 'g(x) = P(X = x)': '0.25'}, {'f(x) = P(X = x)': '0.50', 'h(x) = P(X = x)': '0.30', 'X = x': '3', 'g(x) = P(X = x)': '0.50'}, {'f(x) = P(X = x)': '0.25', 'h(x) = P(X = x)': '0.35', 'X = x': '4', 'g(x) = P(X = x)': '0.05'}], 'headers': ['X = x', 'f(x) = P(X = x)', 'g(x) = P(X = x)', 'h(x) = P(X = x)']}}}
ANSWER: True
EXAMPLE 4:
QUESTION: {'options': {'A': 'H0: μ = 10 versus Ha: μ ≠ 10, with a calculated t-statistic of 2.05 and critical t-values of ±1.984.', 'C': 'H0: μ = 10 versus Ha: μ ≠ 10, with a calculated t-statistic of 2.', 'B': 'H0: μ ≤ 10 versus Ha: μ > 10, with a calculated t-statistic of 2.35 and a critical t-value of +1.679'}, 'question': 'For each of the following hypothesis tests concerning the population mean, μ, state the conclusion regarding the test of the hypotheses.', 'data': ''}
ANSWER: False'''

MC_PROMPT = '''
QUESTION: {question}
ANSWER:'''

@retry(tries=5, delay=2, backoff=2)
def call_openai(prefix: str, prompt: str, input: str, temperature: float = 0.0) -> str:
    try:
        prompt = prefix + prompt.format(question=input)
        # print(prompt)
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        # response = response['choices'][0]['message']['content'].strip()
        response = json.loads(response['choices'][0]['message']['content'].strip())
    except openai.APIError as e:
        print(e)
        raise # This will trigger the retry
    except json.JSONDecodeError as e:
        print(response['choices'][0]['message']['content'].strip())
        print(e)
        raise
    except Exception as e:
        print(e)
        raise
    return response


# Set up Firebase
cred = credentials.Certificate('cfa-creds.json')
firebase_admin.initialize_app(cred)
db = firestore.client()
col_ref = db.collection('questions')
# Stream the documents in the collection
docs = [question.to_dict() for question in col_ref.stream()]
print(docs[1])

llm = ChatOpenAI(temperature=0, model=MODEL)


for doc in tqdm(docs):
    time.sleep(1.0)
    # if doc['parsed_question']['id'] != "74fd8267-342d-461f-b6b6-c617223beddf":
    #     continue
    doc_id = doc['parsed_question']['id']
    mc_question = doc['mc_question']
    input_question = {key: value for key, value in doc["parsed_question"].items() if key in ["question", "options", "data"]}
    # This all returns the most relevamnt docs from the ElasticSearch index
    # They can then be fed into a custom prompt to answer the question
    # print("Question: ", input_question)

    # Check if the question is a MC question and that an answer doesn't already exist
    if mc_question == "True":
        ans = call_openai(prefix=ANSWER_PROMPT_PREFIX, prompt=ANSWER_PROMPT, input = json.dumps(input_question))
        # print("Answer: ", {"answer": ans})
        doc_ref = db.collection('questions').document(doc_id)
        doc_ref.update({"answer": ans})
    # ans = call_openai(prefix=MC_PROMPT_PREFIX, prompt=MC_PROMPT, input = json.dumps(input_question))
    # print("Answer: ", {"mc_question": ans})

    ## update forebase with the answer
    # doc_ref = db.collection('questions').document(doc_id)
    # doc_ref.update({"mc_question": ans})