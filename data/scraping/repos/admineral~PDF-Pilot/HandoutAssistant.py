import os
import fitz
import json
import requests
import openai
import re
import pathlib
import langchain
import faiss
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings


#ADD API-KEYs PLEASE !!!
os.environ["OPENAI_API_KEY"] = "YOUR-OPENAI-API-KEY-HERE"
os.environ["AI21_API_KEY"] = "YOUR-AI21-Studio-API-KEY-HERE"


class PDFHandler:
    @staticmethod
    def pdf_to_text(pdf_path):
        doc = fitz.open(pdf_path)
        text = ""
        page_texts = []
        for page in doc:
            page_text = page.get_text("text")
            text += page_text
            page_texts.append({"text": page_text, "page_number": page.number})
        return text, page_texts

    @staticmethod
    def highlight_text(input_pdf, output_pdf, text_to_highlight):
        phrases = text_to_highlight.split('\n')
        with fitz.open(input_pdf) as doc:
            for page in doc:
                for phrase in phrases:
                    areas = page.search_for(phrase)
                    if areas:
                        for area in areas:
                            highlight = page.add_highlight_annot(area)
                            highlight.update()
            doc.save(output_pdf)

class AI21Segmentation:
    @staticmethod
    def segment_text(text):
        url = "https://api.ai21.com/studio/v1/segmentation"
        payload = {
            "sourceType": "TEXT",
            "source": text
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {os.environ['AI21_API_KEY']}"
        }
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            json_response = response.json()
            return json_response.get("segments")
        else:
            print(f"An error occurred: {response.status_code}")
            return None

class OpenAIAPI:
    def __init__(self):
        openai.api_key = os.environ["OPENAI_API_KEY"]

    def get_answer_and_id(self, prompt):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=0.5,
            max_tokens=2000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        answer_text = response.choices[0].text.strip()
        lines = response.choices[0].text.strip().split('\n')
        answer = lines[0].strip()
        answer = answer.replace("Answer:", "").strip()
        try:
            segment_id = int(re.search(r'<ID: (\d+)>', answer).group(1))
            answer = re.sub(r'<ID: \d+>', '', answer).strip()
        except AttributeError:
            segment_id = None
        return answer, segment_id

class HandoutAssistant:
    def __init__(self):
        self.openai_api = OpenAIAPI()
        self.current_pdf_path = None
        self.questions_data = None
        self.embedder = OpenAIEmbeddings()


    def process_pdf(self, pdf_path):
        text, page_texts = PDFHandler.pdf_to_text(pdf_path)
        #print(f"page_texts: {page_texts}")  # Add this line
        segmented_text = AI21Segmentation.segment_text(text)
        questions_data = self.assign_page_numbers_and_ids_to_segments(segmented_text, page_texts)
        return questions_data

    def assign_page_numbers_and_ids_to_segments(self, segmented_text, page_texts):
        for idx, segment in enumerate(segmented_text):
            segment_text = segment["segmentText"]
            segment["id"] = idx + 1
            max_overlap = 0
            max_overlap_page_number = None
            #print(f"Segment text: {segment_text}")  # Add this line
            for page_text in page_texts:
                overlap = len(set(segment_text.split()).intersection(set(page_text["text"].split())))
                if overlap > max_overlap:
                    max_overlap = overlap
                    max_overlap_page_number = page_text["page_number"]
            segment["page_number"] = max_overlap_page_number + 1
            #print(f"Element ID: {segment['id']}, Page Number: {segment['page_number']}, Max Overlap Page Text: {page_texts[max_overlap_page_number]['text']}")
            print(f"Element ID: {segment['id']}, Page Number: {segment['page_number']}")
        return segmented_text

    def build_faiss_index(self, questions_data):
        # Convert questions_data to a list of Documents
        documents = [Document(page_content=q_data["segmentText"], metadata={"id": q_data["id"], "page_number": q_data["page_number"]}) for q_data in questions_data]


        # Create the FAISS index (vector store) using the langchain.FAISS.from_documents() method
        vector_store = langchain.FAISS.from_documents(documents, self.embedder)

        return vector_store


    def get_relevant_segments(self, questions_data, user_question, faiss_index):
        retriever = faiss_index.as_retriever()
        retriever.search_kwargs = {"k": 2}

        docs = retriever.get_relevant_documents(user_question)

        relevant_segments = []
        for doc in docs:
            segment_id = doc.metadata["id"]
            segment = next((segment for segment in questions_data if segment["id"] == segment_id), None)
            if segment:
                relevant_segments.append({
                    "id": segment["id"],
                    "segment_text": segment["segmentText"],
                    "score": doc.metadata.get("score", None),

                    "page_number": segment["page_number"]
                })

        relevant_segments.sort(key=lambda x: x["score"] if x["score"] is not None else float('-inf'), reverse=True)

        return relevant_segments




    def generate_prompt(self, question, relevant_segments):


        prompt = f"""
You are an AI Q&A bot. You will be given a question and a list of relevant text segments with their IDs. Please provide an accurate and concise answer based on the information provided, or indicate if you cannot answer the question with the given information. Also, please include the ID of the segment that helped you the most in your answer by writing <ID: > followed by the ID number.

Question: {question}

Relevant Segments:"""
        print("\n\n")
        for segment in relevant_segments:
            prompt += f'\n{segment["id"]}. "{segment["segment_text"]}"'

            print(f"Relevant Element ID: {segment['id']}")  # Add this line to print the relevant element IDs
        return prompt

    def process_pdf_and_get_answer(self, pdf_path, question):
        if pdf_path != self.current_pdf_path:
            self.current_pdf_path = pdf_path
            self.questions_data = self.process_pdf(pdf_path)

            # Build the FAISS index (vector store)
            self.faiss_index = self.build_faiss_index(self.questions_data)

        # Use the retriever to search for the most relevant segments
        relevant_segments = self.get_relevant_segments(self.questions_data, question, self.faiss_index)

        if not relevant_segments:
            return "I couldn't find enough relevant information to answer your question.", None, None, None

            

        prompt = self.generate_prompt(question, relevant_segments)
        answer, segment_id = self.openai_api.get_answer_and_id(prompt)

        if segment_id is not None:
            segment_data = next((seg for seg in relevant_segments if seg["id"] == segment_id), None)
            segment_text = segment_data["segment_text"] if segment_data else None
            page_number = next((segment["page_number"] for segment in self.questions_data if segment["id"] == segment_id), None)
        else:
            page_number = None
            segment_text = None


        return answer, segment_id, segment_text, page_number






def main():
    pdf_path = "/Users/handout.pdf"
    output_pdf = "/Users/output.pdf"
    question = "How are the findings of the post-project evaluation documented?"

    handout_assistant = HandoutAssistant()
    answer, segment_id, segment_text, page_number = handout_assistant.process_pdf_and_get_answer(pdf_path, question)


    print("\n\n----------------------")
    print(f"Answer:\n\n{answer}")
    print("----------------------\n")

    if page_number is not None:
        print("\n-------------------------")
        print(f"Relevant Page_Number: {page_number}")
        print("-------------------------\n")

        print("\n---------------")
        print(f"Relevant ID: {segment_id}")
        print("---------------\n")

        print("\n============================================================")
        print(f"Relevant Text-Segment: \n\n{segment_text}")
        print("============================================================\n\n")

        PDFHandler.highlight_text(pdf_path, output_pdf, segment_text)
        print(f"Highlighted PDF saved to: {output_pdf}")
    else:
        print("No relevant segment found to highlight in the PDF.\n")

if __name__ == "__main__":
    main()

