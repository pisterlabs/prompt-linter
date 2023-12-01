import json
import os
import pdb
import re
import time
import traceback
import numpy as np
import openai
from absl import app, flags, logging
from matplotlib.font_manager import json_load


openai.api_key = os.getenv("OPENAI_API_KEY_3")

openai.organization = os.getenv("OPENAI_API_ORGANIZATION_3")


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "prompt_file", default=None, help="Prompt file to use for the problem"
)

flags.DEFINE_string("output_file", default=None, help="Output file to write to")

flags.DEFINE_string("input_delim", default=",", help="Input digits delimiter")

flags.DEFINE_string("output_delim", default=",", help="Output digits delimiter")

flags.DEFINE_string("exp_file", default=None, help="json file from previous exps")

flags.DEFINE_string(
    "output_form", default="reversed", help="Whether to order the digits"
)  # options=["ordered", "reversed", "plain"]

flags.DEFINE_string(
    "range", default="3,8", help="Range of number of digits to evaluate (min, max)"
)

flags.DEFINE_integer("seed", default=0, help="random seed")

flags.DEFINE_integer("max_tokens", default=400, help="LM max generation length")

flags.DEFINE_string("exp_folder", "results", help="Experiment folder")

flags.DEFINE_string("engine", "text-davinci-002", help="GPT engines")


def digit_to_str(digit: int, delim: str = ",") -> str:
    return delim.join(list(str(digit)))


def random_with_n_digits(rng, n: int) -> int:
    range_start = 10 ** (n - 1)
    range_end = (10**n) - 1
    return rng.integers(range_start, range_end)


def eval_one_length(outputs, answers, preds):
    total_count = 0.0
    total_correct = 0.0
    for i, (output, answer) in enumerate(zip(outputs, answers)):

        if FLAGS.output_delim == ",":
            digit_regex = r"([0-9]+(,[0-9]+)+)"
        elif FLAGS.output_delim == " ":
            digit_regex = r"([0-9]+(\s[0-9]+)+)"
        elif FLAGS.output_delim == "":
            digit_regex = r"([0-9]+)"
        else:
            raise ValueError(f"Unknown delimiter: {FLAGS.output_delim}")

        try:
            if FLAGS.output_form == "ordered":
                pred = re.search(
                    r"correct order ?i?s? " + digit_regex, output
                ).groups()[0]
            elif FLAGS.output_form == "ordered_v2":
                pred = re.search(
                    r"answer ?i?s? " + digit_regex,
                    output,
                ).groups()[0]
            elif FLAGS.output_form == "reversed":
                pred = re.search(r"reverse order is " + digit_regex, output).groups()[0]
                pred = pred[::-1]
            elif FLAGS.output_form == "plain":
                # assert FLAGS.output_delim == ""
                output = output.split("\n")[0]
                # print("here the output is: ", output)
                pred = re.search(digit_regex, output).groups()[0]
                # print("here the pred is: ", pred)
            elif FLAGS.output_form == "zeroshot_v2":
                # assert FLAGS.output_delim == ""
                # print("here the output is: ", output)
                output = output.lower().split("answer")[1]
                pred = re.search(digit_regex, output).groups()[0]
                # print("here the pred is: ", pred)
            else:
                raise ValueError("Unknown ordering")

            pred = (
                pred.replace(" ", "")
                .replace(FLAGS.output_delim, "")
                .replace(".", "")
                .replace(",", "")
                .lstrip("0")
            )
            preds.append(pred)
            total_correct += int(pred) == int(answer.replace(" ", ""))
            total_count += 1
        except:
            logging.info(f"Parse error:  {output}")
            preds.append("Parse Error")
            total_count += 1
    accuracy = total_correct / total_count
    return accuracy


def main(_):
    rng = np.random.default_rng(FLAGS.seed)

    logging.info(f"Using delimiter:{FLAGS.input_delim}:{len(FLAGS.input_delim)}")

    with open(FLAGS.prompt_file) as handle:
        template = handle.read()

    if FLAGS.exp_file is None:
        logging.info(f"Will use open ai to get the outputs")
        exp_data = {}
        rstart, rend = FLAGS.range.split(",")
        for n in range(int(rstart), int(rend)):
            inputs = []
            answers = []
            outputs = []
            preds = []
            exp_data[n] = {
                "inputs": inputs,
                "answers": answers,
                "outputs": outputs,
                "preds": preds,
            }
            for _ in range(5):
                current_inputs = []
                for i in range(20):
                    x1 = random_with_n_digits(rng, n)
                    x2 = random_with_n_digits(rng, n)
                    y = x1 + x2
                    y_str = digit_to_str(y, delim=" ")
                    x1_str = digit_to_str(x1, delim=FLAGS.input_delim)
                    x2_str = digit_to_str(x2, delim=FLAGS.input_delim)
                    current_input = template.format(x1=x1_str, x2=x2_str)
                    current_inputs.append(current_input)
                    inputs.append(current_input)
                    answers.append(y_str)

                try:
                    response = openai.Completion.create(
                        engine=FLAGS.engine,
                        prompt=current_inputs,
                        temperature=0,
                        max_tokens=FLAGS.max_tokens,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0,
                    )

                    current_outputs = response["choices"]
                    current_outputs = [
                        current_outputs[i]["text"] for i in range(len(current_outputs))
                    ]
                    outputs.extend(current_outputs)
                except Exception as e:
                    logging.warn("Error:", e)

            time.sleep(60)

            exp_data[n]["accuracy"] = eval_one_length(outputs, answers, preds)
    else:
        logging.info(f"Loading from exp_file: {FLAGS.exp_file}")
        with open(os.path.join(FLAGS.exp_folder, FLAGS.exp_file)) as handle:
            exp_data = json.load(handle)

        for n, v in exp_data.items():
            preds = []
            exp_data[n]["preds"] = preds
            exp_data[n]["accuracy"] = eval_one_length(
                v["outputs"], v["answers"], v["preds"]
            )

    output_file = os.path.join(FLAGS.exp_folder, FLAGS.output_file)
    # if os.path.exists(output_file):
    #     with open(output_file) as handle:
    #         exp_data_file = json.load(handle)
    #         exp_data_file["7"] = exp_data.get(7, None) or exp_data["7"]

    #     with open(output_file, "w") as handle:
    #         json.dump(exp_data_file, handle)

    # # else:
    with open(output_file, "w") as handle:
        json.dump(exp_data, handle)


if __name__ == "__main__":
    app.run(main)
