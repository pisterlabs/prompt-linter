
import cohere

co = cohere.Client("9Io2cczfvEbcCBmO4JUnybY7gZbIDDln8QC2xA60")
def summmarize(text): #input text of the webpage to put into api (as a string)
    response = co.generate(
        model = "xlarge",
        prompt = "Summarize this text: " + text,
        max_tokens = 50, #length of the summary ~2-3 tokens per word, number of words?
        temperature = 0.6,
        k = 0,
        p = 1,
        num_generations = 1 
    )
    # print(response.generations[0].text)
    summary = response.generations[0].text
    return summary #output summary of the webpage as text 

def qna(text, question): #input webpage as text and question as string
    response = co.generate(
        model = "xlarge",
        prompt = "Answer this question based on the text: " + question + "\n" + " Text: " + text,
        max_tokens = 50,
        temperature = 0.6,
        k = 0,
        p = 1,
        num_generations = 1
    )

    answer = response.generations[0].text
    print(answer)
    return answer #output answer as text

if __name__ == "__main__":
    test_text = "The platypus is cute."
    test_question = "What animal is cute?"
    # summarize(test_text)
    qna(test_text, test_question)