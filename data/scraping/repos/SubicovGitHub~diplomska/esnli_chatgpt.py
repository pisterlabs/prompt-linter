import pandas as pd
from datasets import load_dataset
import openai
import time


def process_response(response):
    response_split = response.split(':')
    r = ''.join([c for c in response_split[0].lower() if c.isalpha() or c == ' '])
    response_first_part_split = r.split()

    predicted_label = 6
    for word in response_first_part_split:
        if word == "entailment":
            predicted_label = 0
            break
        elif word == "contradiction":
            predicted_label = 2
            break
        elif word == "neutral":
            predicted_label = 1
            break

    cfe_1 = response_split[-2].split('\"')[1]
    cfe_2 = response_split[-1].split('\"')[1]

    return predicted_label, cfe_1, cfe_2


if __name__ == "__main__":
    openai.api_key = ""

    test_data = load_dataset("esnli", split="test")
    x_test_premise = test_data[0:99]["premise"]
    x_test_hypothesis = test_data[0:99]["hypothesis"]
    y_test = test_data[0:99]["label"]

    data_counterfactual = pd.DataFrame()
    data_counterfactual["Premise"] = ""
    data_counterfactual["Hypothesis"] = ""
    data_counterfactual["Label"] = 0
    data_counterfactual["Predicted"] = 0
    data_counterfactual["Counterfactual_1"] = ""
    data_counterfactual["Counterfactual_2"] = ""
    data_counterfactual["Response"] = ""

    wrong_class = 0
    f = open("res.txt", "w")
    for i, premise in enumerate(x_test_premise):
        hypothesis = x_test_hypothesis[i]
        send_content = "Can you solve a natural language inference problem and tell me " \
                       "whether the relation between following premise and hypothesis is " \
                       "that of entailment, contradiction or neutral (there is no relation). " \
                       "\npremise: \"" + premise + "\" \nhypothesis: \"" + hypothesis + "\"\n" \
                       "Can you also modify the hypothesis in a minimal way so that " \
                       "the new relationship would be that of the other two relations? " \
                       "Please reply with modified hypotheses only."
        answer = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": send_content}]
        )
        print(i)
        time.sleep(20)      # API rate limit

        result = answer.choices[0].message.content
        try:
            y_pred, counterfactual_1, counterfactual_2 = process_response(result)
            print(f"---->{y_pred}")
        except:
            print("error")
            counterfactual_1 = ""
            counterfactual_2 = ""
            y_pred = 8
        print(result)
        print("------------------------------\n\n")

        if y_pred != y_test[i]:
            wrong_class += 1

        inst = {
            "Premise": premise,
            "Hypothesis": hypothesis,
            "Label": y_test[i],
            "Predicted": y_pred,
            "Counterfactual_1": counterfactual_1,
            "Counterfactual_2": counterfactual_2,
            "Response": result
        }
        data_counterfactual.loc[len(data_counterfactual)] = inst
        f.write(result + "\n|\n")

    f.close()
    data_counterfactual.to_csv("esnli_chatgpt.csv", index=False, sep='|')

    ca = (len(y_test) - wrong_class) / len(y_test)
    print(f"ca: {ca}*")
