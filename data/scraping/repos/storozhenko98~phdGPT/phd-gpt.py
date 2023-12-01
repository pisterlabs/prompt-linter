from PyPDF2 import PdfReader
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")
directory = './articles'


def procure_text(article_name):
    reader = PdfReader(article_name)
    pages = ""
    for page in reader.pages:
        pages += page.extract_text()
    return pages


def lentext(article):
    return len(article.split())

reg_prompt = """You are phdGPT and your job is to summarize articles for me. When summarizing, focus on the most important points in the article. Be sure to return at least 500 words. If the article is long, return an appropriately lengthy summary. Additionally, ensure that you cover the articles systematically since they are mostly from analytic philosophy. Be sure to explain any and all complex terms in the article. This is for a comprehensive understanding as part of an important PhD project. Make sure you are technical and thorough. Complexities are important."""

chunk_prompt = """You are phdGPT, and you're tasked with summarizing a section of a longer article. Please concentrate on the key points and concepts in this section. Clarify any complicated terms or ideas. As each section is part of a more extensive analysis in analytic philosophy, it is crucial to maintain a systematic approach in summarizing. This summary will contribute to a collective summary of the entire article. Ensure that this section is comprehensive, technical, explains all relevant aspects, and is at least 500 words long. Make sure you are technical and thorough. Complexities are important."""

final_summary_prompt = """You are phdGPT, and your task is to create a cohesive and comprehensive summary from the summaries of different sections of a lengthy article. Please focus on synthesizing the main ideas, arguments, and concepts from each section to form a coherent overview. Explain any complex terms and ensure that the final summary is systematic and detailed, as it is part of an important PhD project in analytic philosophy. The summary has to be comprehensive and at least 500 words long. It has to capture all important and technically relevant aspects of the article. Make sure you are technical and thorough. Complexities are important."""

def summarize_article(article, prompt):
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Here is the article: " + article},
        ]
    )
    return completion.choices[0].message.content


def summarize_long(article, article_name):
    words = article.split()
    chunks = [' '.join(words[i:i + 4000]) for i in range(0, len(words), 4000)]
    summaries = []
    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i + 1} of {article_name}")
        summary = summarize_article(chunk, chunk_prompt)
        summaries.append(summary)
        #write intermediate summaries to /intermediaries 
        with open(f"./intermediaries/{article_name}_summary_{i + 1}.txt", "w") as f:
            f.write(summary)
    # perform length check on summaries together
    all_summaries = ' '.join(summaries)
    if len(all_summaries.split()) > 5000:
        print("Summaries are long: will chunk again")
        summary_words = all_summaries.split()
        summary_chunks = [' '.join(summary_words[i:i + 5000]) for i in range(0, len(summary_words), 5000)]
        summary_summaries = []
        for j, sum_chunk in enumerate(summary_chunks):
            print(f"Summarizing chunk {j + 1} of summary of {article_name}")
            summary_of_summary = summarize_article(sum_chunk, chunk_prompt)
            summary_summaries.append(summary_of_summary)
        summaries = summary_summaries

    final_summary = summarize_article(' '.join(summaries), final_summary_prompt)
    with open(f"./summaries/{article_name}_final_summary.txt", "w") as f:
        f.write(final_summary)
    print(f"Finished summarizing {article_name}")
    return final_summary


def main():
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            print("Summarizing " + filename)
            article = procure_text(f"{directory}/{filename}")
            if lentext(article) > 4000:
                print("Article is long: will chunk")
                summarize_long(article, filename[:-4])
            else:
                summary = summarize_article(article, reg_prompt)
                with open(f"./summaries/{filename[:-4]}_summary.txt", "w") as f:
                    f.write(summary)
                print(f"Finished summarizing {filename}")


if __name__ == '__main__':
    main()