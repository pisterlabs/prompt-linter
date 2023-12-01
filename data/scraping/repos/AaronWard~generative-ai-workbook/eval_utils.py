"""
This contains helper functions used for
evaluating the autogen output vs the expected
answer.

"""
from langchain.llms import OpenAI
from langchain.evaluation.qa import QAEvalChain
from evaluate import load

def run_evaluation_chain(qa_pairs, predictions):
    eval_chain = QAEvalChain.from_llm(llm = OpenAI(temperature=0))
    return eval_chain.evaluate(
        qa_pairs, predictions, question_key="question", prediction_key="response"
    )

def modify_qa_pairs(qa_pairs, predictions):
    # Some data munging to get the examples in the right format
    modified_qa_pairs = [x.copy() for x in qa_pairs]
    for i, eg in enumerate(modified_qa_pairs):
        eg["id"] = str(i)
        eg["answers"] = {"text": [eg["answer"]], "answer_start": [0]}
        predictions[i]["id"] = str(i)
        predictions[i]["prediction_text"] = predictions[i]["response"]

    for p in predictions:
        del p["response"]

    for eg in modified_qa_pairs:
        del eg["question"]
        del eg["answer"]

    return modified_qa_pairs

def get_precision_score(graded_outputs):
    correct = 0
    for graded_output in graded_outputs:
        if graded_output["results"].strip() == "CORRECT":
            correct += 1


    return correct/len(graded_outputs)


def get_squad_score(modified_qa_pairs, predictions):
    """
    https://rajpurkar.github.io/SQuAD-explorer/

    Exact Match: For each question-answer pair, if the tokens of the model's prediction 
        exactly match the tokens of the true answer, exact_match is 100; otherwise, exact_match is 0. 
        One can imagine that each token matching is a rare occurrence for a stochastic system. 
        This metric should be taken with a grain of salt for our use case. 
    F1 Score: This is a well-known metric that cares equally about the precision and 
        recall of the system. Precision is the ratio of shared tokens to the total number of tokens 
        in the prediction. Recall is the ratio of shared tokens to the 
        total number of tokens in the ground truth.
    """
    squad_metric = load("squad")

    return squad_metric.compute(
        references=[modified_qa_pairs[1]],
        predictions=[predictions[1]],
    )