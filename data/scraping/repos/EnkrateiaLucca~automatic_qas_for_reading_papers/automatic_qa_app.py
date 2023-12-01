# Load a pdf with langchain PyPDFLoader
from langchain.document_loaders import PyPDFLoader
import openai
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def load_paper(file_path="./paper.pdf"):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs


def create_qa(context, num=5):
    # Defining the context for creating the Q&As
    # Prompt to create the questions
    q_a_prompt = f"Create a set of {num} questions with answers based solely on this text from a paper:\n\n{context}\n\n. Separate each block composed of a question and an answer with 3 dashes '---' like this Q: <question>\n A:<answer> --- Q: <question>\n A:<answer> etc.... Let's think step by step. Q:"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful research and\
            programming assistant"},
                  {"role": "user", "content": q_a_prompt}]
    )
    
    return response["choices"][0]["message"]["content"]

def run_qa_session(docs, num, context_window):
    overall_scores = []
    qa_dict = {}
    for i, page_num in enumerate(range(0, len(docs), context_window)):
        # Concatenating the pages (set by the context_window) to give as context for the Q&A
        context = "".join([page.page_content for page in docs[page_num:page_num+context_window]])
        q_a = create_qa(context, num)
        # Create a list of questions and answers from the output string by leveraging the '---' separator
        q_a_list = q_a.split('---')
        scores_list = []
        for qa in q_a_list:
            question = qa.split("A:")[0].replace("Q:", "")
            answer = qa.split("A:")[1].replace("Q:", "")
            user_answer = input(question)
            qa_dict[f"Round {i}"] = {"question": question, "answer": answer, "user_answer": user_answer}
            print("CORRECT ANSWER: ", answer)
            print("***")
            score_feedback = evaluate_answer(question, answer, user_answer)
            score = score_feedback.split("SCORE:")[1].split("FEEDBACK:")[0]
            try:
                feedback = score_feedback.split("FEEDBACK:")[1]
            except:
                feedback = "Error getting feedback"
            # write a check to make sure the output can be turned into an integer
            try:
                score = int(score)
            except:
                print("The score could not be converted to an integer. Please try again.")
                print("The output score was: ", score)
            if type(score)==int:
                scores_list.append(score)
            print("SCORE:", score)
            print("***")
            print("FEEDBACK:", feedback)
            print("*********")
        round_score = sum(scores_list)/len(scores_list)
        print("ROUND SCORE:", round_score)
        overall_scores.append(round_score)
        continue_input = input("Press enter to continue to the next round or press 'q' to quit.")
        if continue_input == "q":
            break
        
        
    
    return overall_scores, qa_dict


def evaluate_answer(question, true_answer, user_answer):
    # Evaluate the answer
    evaluate_prompt = f"Given this question: {question} for which the correct answer is this: {true_answer}, give a score from 0 to 100 to the following answer given by the user: {user_answer}. The output should be formmated as follows: SCORE: <score number as an integer (e.g 45, 90, etc...)> \n: FEEDBACK: <A one sentence feedback justifying the score.>"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = [{"role": "system", "content": "You are a helpful research and\
            programming assistant"},
                    {"role": "user", "content": evaluate_prompt}]
    )

    return response["choices"][0]["message"]["content"]




def plot_scores(overall_scores):
    plt.plot(overall_scores)
    plt.xlabel("Round")
    plt.ylabel("Score")
    plt.title("Q&A Session Scores")
    plt.show()


def main():
    docs = load_paper(file_path)
    num = 1
    context_window = 3
    overall_scores, qa_dict = run_qa_session(docs, num, context_window)
    plot_scores(overall_scores)
    print("The Q&A data: ", qa_dict)

if __name__ == "__main__":
    file_path = "./paper.pdf"
    main()