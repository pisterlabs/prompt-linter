import argparse
import sys
import os
import openai
import logging
# Set the logging level for all loggers to ERROR before importing the classes
logging.getLogger().setLevel(logging.ERROR)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
#from src.models.bert_embed import BertEmbed
from api_secrets import API_KEY
#from dtu_api import API_KEY
#from src.models.semantic_search_function import get_similar_paragraphs
from src.models.semantic_search_two_towers import get_similar_paragraphs
from src.models.bert_rerank import BertRerank
from src.models.reranker_function import rerank_paragraphs
import numpy as np
from src.features.two_towers_fine_tune_multiplenegatives import *

openai.api_key = API_KEY



def process_question_with_gpt4(question, list_of_paragraphs):
    formatted_paragraphs = "\n\n".join([f"Tekststykke {i+1}: {p}" for i, p in enumerate(list_of_paragraphs)])



    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Du er en chatbot der besvarer et bruger spørgsmål efter bedste evne baseret på information i en række givne tekststykker. Baser alene dit svar på informationen i disse refer til dem i din argumentation. Gennemgå tekststykkerne og noter dig om nogle af disse indeholder information, der på nogen måde kan bruges til at besvare spørgsmålet. Du kan eventuelt fortolke spørgsmålet en anelse, så det passer bedre til den information du er blevet givet. Såfremt den givne information slet ikke er relevant i forhold til spørgsmålet, så må du opsummere den givne information og forklare hvorfor dette ikke er tilstrækkeligt relevant til at begynde at besvare spørgsmålet."},
            {"role": "user", "content": f"Bruger spørgsmål: {question}\n\n{formatted_paragraphs}"}
        ],
        temperature=0.5,
        max_tokens=500,
    )

    answer = response['choices'][0]['message']['content']
    return answer

def main(question):
    print(20*"-")
    print("Loading models...")
    # Load the fine-tuned re-ranker model
    fine_tuned_model_path = 'models/fine_tuned_model/check_highbatch2_model.pth'

    # Load the fine-tuned re-ranker model
    reranker = BertRerank(model_path=fine_tuned_model_path)
    
    print(20*"-")
    print("Running semantic search...")

    # Perform semantic search to get the top 1000 paragraphs
    similar_paragraphs, similar_file_indices = get_similar_paragraphs(question, k=100)

    print(20*"-")
    # Rerank the paragraphs using the rerank_paragraphs function
    reranked_paragraphs = rerank_paragraphs(question, similar_file_indices, similar_paragraphs, reranker) 
    all_info_paragraphs = []
    final_text = []
    seen = set()  # Create an empty set to keep track of seen elements
    for context_paragraph, score, filename, paragraph_index in reranked_paragraphs:
        if context_paragraph not in seen:  # Check if the element has not been seen before
            all_info_paragraphs.append((context_paragraph, score, filename, paragraph_index))
            final_text.append(context_paragraph)
            seen.add(context_paragraph)  # Add the element to the seen setl, then we remove duplicates


    print(f"Top 8 reranked afsnit:\n")
    print(20*"-")
# Print the top 10 reranked paragraphs
    for idx, (context_paragraph, score, filename, paragraph_index) in enumerate(all_info_paragraphs[:8]):
        print(f"Paragraph {idx + 1}: {context_paragraph}\nRelevance Score: {score}\n")


    print(20*"-")
    print("Generating answer with GPT-4...")
    answer = process_question_with_gpt4(question, final_text[:8])

    print(f"Spørgsmål: {question}\n")
    print(f"Svar: {answer}\n")

if __name__ == "__main__":
    """UNCOMMENT THIS TO RUN THE SCRIPT FROM THE COMMAND LINE"""
    parser = argparse.ArgumentParser(description="Rerank top 1000 paragraphs for a given question.")
    parser.add_argument("question", type=str, help="Input question for semantic search and reranking")
    args = parser.parse_args()
    main(args.question)

    #main("Producerer Danmark flere vindmøller i dag end for 10 år siden?")


