
import openai

openai.api_key = ''


def load_qa_pairs(file_path):
    qa_pairs = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            question, answer = line.strip().split(',')
            qa_pairs.append({'question': question, 'answer': answer})
    return qa_pairs

qa_pairs = load_qa_pairs('fitness_qa.txt')


def find_answer_in_dataset(user_input, qa_pairs):
    for qa in qa_pairs:
        if user_input.lower() in qa['question'].lower():
            return qa['answer']
    return None  


def get_gpt3_response(user_input):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=user_input,
        max_tokens=50  
    )
    return response.choices[0].text


def chatbot_response(user_input):
    
    dataset_answer = find_answer_in_dataset(user_input, qa_pairs)
   
    if dataset_answer:
        return dataset_answer 
    else:
        
        return get_gpt3_response(user_input)

user_input = input("You: ")
response = chatbot_response(user_input)
print("FitnessBot:", response)
