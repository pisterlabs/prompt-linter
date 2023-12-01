import os
# import json
import openai
# from pdflatex import PDFLaTeX

openai.api_key = os.getenv("OPENAI_API_KEY")

def chat(prompt):
    if len(prompt) == 0:
        return ""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            max_tokens=1000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        # debug only
        # json_pretty = json.dumps(response, indent=2)
        # print(json_pretty)
        # print("Question:" + prompt + "\n\n")
        # print("chatGPT answer\n")

        result = response["choices"][0]["message"]["content"]
        return result

    except Exception as e:
        print(e)


def build_article_prompt(word_list):
    concated_word_list = ",".join(word_list)
    print(concated_word_list)
    return concated_word_list


def write_to_latex(word_list, article1, article2):
    f = open("biblical_article_template.tex", "r")
    latex_content = f.read()

    # latex_content = latex_content.replace("@TITLE@", title)
    latex_content = latex_content.replace("@WORDLIST@", word_list)
    latex_content = latex_content.replace("@BIBLICAL_ARTICLE1@", article1)
    latex_content = latex_content.replace("@BIBLICAL_ARTICLE2@", article2)

    print(latex_content)

    fw = open("biblical_article_rendered.tex", "w")
    fw.write(latex_content)
    fw.close()
    f.close()


def generate_article(word_list):
    # print(my_word_list)
    word_list_str = ",".join(my_word_list)

    word_list_question = "Please generate a Latex formatted table for these words: " + word_list_str + ", including three columns, 'word', 'meaning', 'word root and meaning' in latex format with width ratio 0.3, 0.4 and 0.3 respectively. table's row and cell should have separator lines. Please only return the Latex content, nothing else."
    word_list_table = chat(word_list_question)

    biblical_article_prompt = "now you are biblical article writer for Sunday message, please use all these words: " + word_list_str + ", write one article within 500 words. This article should include all these words and quote bible NIV verses as more as possible. Make sure the content is safe for youth. Please include a title at the beginning, bold it, no other content except the title and content. Please list all the bible verses at the end of the article with bible version.  Article should NOT include any URL."
    biblical_article_1 = chat(biblical_article_prompt)
    biblical_article_2 = chat(biblical_article_prompt)

    # bold words
    for word in my_word_list:
        biblical_article_1 = biblical_article_1.replace(word, "\\textbf{" + word + "}")
        biblical_article_2 = biblical_article_2.replace(word, "\\textbf{" + word + "}")

        biblical_article_1 = biblical_article_1.replace(word.lower(), "\\textbf{" + word.lower() + "}")
        biblical_article_2 = biblical_article_2.replace(word.lower(), "\\textbf{" + word.lower() + "}")

    write_to_latex(word_list_table, biblical_article_1, biblical_article_2)


if __name__ == '__main__':

    # sample keywords from one bible study fellowship focusing on how to establish healthy and biblical couple relationship
    # replace them with your word list to generate new genre articles.
    # rerun the scripts with same keywords also can generate variations
    my_word_list = ["relation", "wife","husband", "family", "respect", "convict", "convince", "control", "coerce", "expectation", "love"]
    generate_article(my_word_list)

