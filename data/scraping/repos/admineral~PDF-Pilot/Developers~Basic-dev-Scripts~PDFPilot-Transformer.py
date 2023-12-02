import os
import fitz
import json
import requests
from transformers import pipeline
import openai
import re

class HandoutAssistant:
    def __init__(self):
        self.nlp = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

    def pdf_to_text(self, pdf_path):
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text")
        return text

    def load_handout_questions(self, json_file_path):
        with open(json_file_path, "r") as f:
            questions_data = json.load(f)

        for idx, question_data in enumerate(questions_data):
            question_data["id"] = idx + 1

        return questions_data

    def get_relevant_segments(self, questions_data, user_question):
        relevant_segments = []

        for question_data in questions_data:
            context = question_data["segmentText"]
            response = self.nlp(question=user_question, context=context)

            if response["score"] > 0.5:
                relevant_segments.append({
                    "id": question_data["id"],
                    "segment_text": context,
                    "score": response["score"]
                })

        relevant_segments.sort(key=lambda x: x["score"], reverse=True)
        return relevant_segments[:10]

    def save_relevant_segments(self, relevant_segments, output_path):
        with open(output_path, "w") as f:
            json.dump(relevant_segments, f, indent=2)

    def generate_prompt(self, question, relevant_segments):
        prompt = f"""
You are an AI Q&A bot. You will be given a question and a list of relevant text segments with their IDs. Please provide an accurate and concise answer based on the information provided, or indicate if you cannot answer the question with the given information. Also, please include the ID of the segment that helped you the most in your answer by writing <ID: > followed by the ID number.

Question: {question}

Relevant Segments:"""

        for segment in relevant_segments:
            prompt += f'\n{segment["id"]}. "{segment["segment_text"]}"'

        return prompt

    def get_openai_answer_and_id(self, prompt, question):
        openai.api_key = "YOUR-OPEN-AI-API-KEY-HERE" #YOUR API KEY HERE 
        print("Sending prompt to OpenAI API...")
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=0.5,
            max_tokens=2000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        # Save the question and response to a file
        result = {
            'question': question,
            'response': response
        }
        with open('openai_question_and_response.json', 'w') as outfile:
            json.dump(result, outfile, indent=4)

        answer_text = response.choices[0].text.strip()
        lines = response.choices[0].text.strip().split('\n')
        answer = lines[0].strip()
        
        try:
            segment_id = int(re.search(r'<ID: (\d+)>', answer).group(1))
            answer = re.sub(r'<ID: \d+>', '', answer).strip()
        except AttributeError:
            segment_id = None

        return answer, segment_id

def highlight_text(input_pdf, output_pdf, text_to_highlight):
    print("Highlighting text:", text_to_highlight)

    # Split the text into phrases
    phrases = text_to_highlight.split('\n')

    with fitz.open(input_pdf) as doc:
        for page in doc:
            for phrase in phrases:
                areas = page.search_for(phrase)
                if areas:
                    print(f"Text found on page {page.number}: {phrase}")
                    for area in areas:
                        highlight = page.add_highlight_annot(area)
                        highlight.update()
                else:
                    print(f"Text not found on page {page.number}: {phrase}")

        doc.save(output_pdf)
    

if __name__ == "__main__":
    assistant = HandoutAssistant()

    pdf_path = '/Users/handout.pdf'
    json_output_path = '/Users/Logs/handout_json_output.json'
    relevant_segments_output_path = '/Users/Logs/most_relevant_segments.json'
    qa_output_path = '/Users//Logs/questions_and_answers.json'
    highlighted_pdf_path = '/Users/Logs/highlighted.pdf'

    text = assistant.pdf_to_text(pdf_path)

    url = "https://api.ai21.com/studio/v1/segmentation"
    payload = {
        "sourceType": "TEXT",
        "source": text
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": "Bearer YOUR-A21-API-KEY-HERE" #Add your key HERE
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        print("Segmentation API call was successful")
        json_response = response.json()
        segmented_text = json_response.get("segments")

        print("Saving JSON output to a new file...")
        with open(json_output_path, 'w') as json_output_file:
            json.dump(segmented_text, json_output_file, indent=2)

        print(f"JSON output saved to {json_output_path}")

    else:
        print(f"An error occurred: {response.status_code}")

    questions_data = assistant.load_handout_questions(json_output_path)

    question_and_answers = []

    while True:
        user_question = input("\nUser: ").strip()
        if user_question.lower() in ["quit", "exit", "bye"]:
            print("Goodbye!")
            break

        if not user_question:
            print("Please enter a valid question.")
            continue

        relevant_segments = assistant.get_relevant_segments(questions_data, user_question)

        if relevant_segments:
            assistant.save_relevant_segments(relevant_segments, relevant_segments_output_path.format(user_question))

            prompt = assistant.generate_prompt(user_question, relevant_segments)

            openai_answer, segment_id = assistant.get_openai_answer_and_id(prompt, user_question)

            print(f"OpenAI Answer: {openai_answer}")

            question_and_answers.append({"question": user_question, "answer": openai_answer, "segment_id": segment_id})

            with open(qa_output_path, "w") as qa_file:
                json.dump(question_and_answers, qa_file, indent=2)
            
            print("Segment ID:", segment_id)
            print("Relevant Segments:", relevant_segments)

            segment_to_highlight = None
            for segment in relevant_segments:
                if segment["id"] == segment_id:
                    segment_to_highlight = segment["segment_text"]
                    break

            if segment_to_highlight:
                highlight_text(pdf_path, highlighted_pdf_path, segment_to_highlight)
            else:
                print("Error: Could not find segment with the specified ID")


        else:
            print("Sorry, I couldn't find an answer to your question in the handout.")

 

