import openai
import random
from datasets import load_dataset


API_KEY = "random_key"
openai.api_key = API_KEY

system_msgs = {
    'easy': "You are a primary school teacher who designs easy reading comprehension questions \
             for primary school students.",
    'medium': "You are a middle school teacher who designs medium level reading comprehension questions \
               for middle school students.",
    'hard': "You are a university professor who designs difficult reading comprehension questions \
             for collage students."
}


def response_parser(response, num_qns):
    qa_pairs = []

    cleaned_response = [line.strip() for line in response.split('\n') if len(line.strip()) > 0]
    if len(cleaned_response) != 2 * num_qns:
        return None

    for i in range(0, 2 * num_qns, 2):
        q = cleaned_response[i]
        a = cleaned_response[i + 1]
        if ":" in q:
            q = q.split(":")[1]
        elif str(i / 2 + 1) + "." in q:
            q = q.split(str(i / 2 + 1) + ".")[1]

        if ":" in a:
            a = a.split(":")[1]
        elif str(i / 2 + 1) + "." in q:
            a = a.split(str(i / 2 + 1) + ".")[1]

        qa_pairs.append({
            "question": q,
            "answer": a
        })
    return qa_pairs


def generate_questions(context, num_qns=5, difficulty_level='easy'):
    system_msg = system_msgs[difficulty_level]
    user_msg = "Please read the following context and generate {cnt} different question-answer pairs in a list. \
                Each element of your list should contain one question-answer pair with clear separation. \
                Context: {context}".format(cnt=num_qns, context=context)

    response = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k",
                                            messages=[{"role": "system", "content": system_msg},
                                                      {"role": "user", "content": user_msg}])
    print(response['choices'][0]['message']['content'])
    qa_pairs = response_parser(response['choices'][0]['message']['content'], num_qns)
    return qa_pairs


if __name__ == '__main__':
    train_dataset, val_dataset, test_dataset = load_dataset("kmfoda/booksum",
                                                            split=["train", "validation", "test"])
    sample = random.choice(train_dataset)
    qa_pairs = generate_questions(sample["chapter"])
    print(qa_pairs)

