from tifascore import get_question_and_answers, filter_question_and_answers, UnifiedQAModel, tifa_score_benchmark, tifa_score_single,  VQAModel
import openai

import os
import re

if __name__ == "__main__":
    
    # prepare the models
    openai.api_key = "" # your api key
    unifiedqa_model = UnifiedQAModel("allenai/unifiedqa-v2-t5-large-1363200")
    vqa_model = VQAModel("vilt")
    
    
    # Generate questions with GPT-3.5-turbo
    # Filter questions with UnifiedQA

    questions = []
    filtered_questions = []
    for text in ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']:
        gpt3_questions = get_question_and_answers(text)
        questions.append(gpt3_questions)
        filtered_questions = filter_question_and_answers(unifiedqa_model, gpt3_questions)
        filtered_questions.append(filtered_questions)


    
    # See the questions
    # print(gpt3_questions)
    print(filtered_questions)

    # get TIFA score  
    score = 0.0
    img_dir = "../imagen-pytorch/fake_images_text/"
    count = 0.0
    for filename in os.listdir(img_dir):
        f = os.path.join(img_dir, filename)
        if os.path.isfile(f):
            count += 1
            label = int(re.findall(r'\d+', filename)[0])
            result = tifa_score_single(vqa_model, filtered_questions[label], f)
            score += result['tifa_score']
    print(score/count)
           

