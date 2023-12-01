import argparse
import json
import os.path
from pathlib import Path
from typing import List, Dict

from gptinference import utils
from tqdm import tqdm

from gptinference.base_prompt import Prompt
from gptinference.openai_wrapper import OpenAIWrapper
from gptinference.utils import read_jsonl_or_json, write_json

from demography_prediction import filter_demographs

OUTPUT_MAP = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

class PersonaCreator(Prompt):
    def __init__(self, engine: str, openai_wrapper: OpenAIWrapper):
        super().__init__()
        self.openai_wrapper = openai_wrapper
        self.engine = engine

    def make_query(self, implicit_persona: List[str], explicit_persona: List[str],
                   topic: str, question: List[str], choices: List[str]) -> str:
        if not question or not choices:
            return ""

        # implicit prompt
        if implicit_persona:
            implicit_persona_list = []
            for persona in implicit_persona:
                # implicit_persona_list.append(persona["declarative_opinion"])
                implicit_persona_list.append(persona)
            implicit_persona_list = [f"{i+1}. {persona}\n" for i, persona in enumerate(implicit_persona_list)]
            implicit_persona_str = "".join(implicit_persona_list)

        # explicit prompt
        if explicit_persona:
            explicit_persona_list = []
            for persona in explicit_persona:
                for key, value in persona.items():
                    explicit_persona_list.append(f"{key}: {value}")
            explicit_persona_str = "\n".join(explicit_persona_list)


        # question prompt
        choices = [f"{OUTPUT_MAP[i]}.{choice}\n" for i, choice in enumerate(choices)]
        choice = "".join(choices)

        # implicit + explicit
        if explicit_persona and implicit_persona:
            prompt = \
f"""A person can be described as follows:

{explicit_persona_str}

The person has the following opinions on {topic}.

Opinions:
{implicit_persona_str}
Based on the above list of opinions and the demographic information, which answer choice will this person select for the question:

Question: {question}

Answer choices:
{choice}
Answer:
"""
        # implicit prompt
        elif implicit_persona:
            prompt = \
f"""A person has the following opinions on {topic}.

Opinions:
{implicit_persona_str}
Based on the above list of opinions, which answer choice will this person select for the question:

Question: {question}

Answer choices:
{choice}
Answer:
"""
            # explicit prompt
        elif explicit_persona:
            prompt = \
f"""A person can be described as follows:

{explicit_persona_str}

Based on the demographic information, which answer choice will this person select for the question:

Question: {question}

Answer choices:
{choice}
Answer:
"""         # zero shot no-persona
        else:
            prompt = \
f"""Question: {question}

Answer choices:
{choice}
Answer:
"""


        return prompt

    def __call__(self, implicit_persona: List[str], explicit_persona: List[str],
                 topic:str, question: str, choices: List[str]) -> str:
        generation_query = self.make_query(
            implicit_persona=implicit_persona,
            explicit_persona=explicit_persona,
            topic=topic,
            question=question,
            choices=choices,
        )

        generated_sent = self.openai_wrapper.call(
            prompt=generation_query,
            engine=self.engine,
            max_tokens=1,
            stop_token="###",
            temperature=0.0
        )
        return generated_sent.strip()  # gpt3 turbo adds newline in the beginning so strip it.


def early_stopping(topicwise_ctr, topic, max_topics, max_users):
    cond_users  = max_users > 0 and topicwise_ctr.get(topic, 0) >  max_users
    if cond_users:
        return True

    cond_topics = max_topics > 0 and len(topicwise_ctr) > max_topics
    if cond_topics:
        return True

    return False

class PersonalizedQA:
    """
    save personalized opinion to output json.
    """
    def __init__(self, args):
        self.num_implicit = args.num_implicit
        self.implicit_sampling = args.implicit_sampling
        self.explicit_definition = args.explicit_definition

    def mk_implicit_persona(self, option, persona_qa, random_qa):
        user_profile = []
        if option == -1:
            return None
        elif self.implicit_sampling == "random":
            user_profile = [x["declarative_opinion"] for x in random_qa]
        elif self.implicit_sampling == "topk":
            # "topk_opinions": {
            #     "Gun owners who have children in their home should keep their shooting skills up-to-date, but it is not essential.": 0.8729113543065781,
            #     "I worry a little about being the victim of a violent crime.": 0.7769040591312351
            #  }
            user_profile = [x for x, score in persona_qa["topk_opinions"].items()]
        return utils.take(num=self.num_implicit, arr=user_profile) if option == 0 or option == 2 else None

    def personalized_qa(self, persona: PersonaCreator, user_responses_jsons: Dict, option: int,
                        max_users:int, max_ques: int, max_topics: int):
        """ get answer choice based on implicit/explicit/implicit+explicit persona.
        """
        model_generated = []
        #############################################################
        # original: 15 topics x ~3000 users x ~75 ques = 3,375,000 questions
        # current:  15 topics x   100 users x ~55 ques = 85,000 questions
        # option1:  15 topics x    50 users x ~25 ques = 18,750 questions  (num implicit points = 8)
        # option2:  15 topics x    25 users x ~50 ques = 18,750 questions  (num implicit points = 8)
        # option3:  15 topics x    35 users x ~30 ques = 15,750 questions  (num implicit points = 8)

        #############################################################
        #  100         "topic": "Views on gender",
        #  100         "topic": "Trust in science",
        #  100         "topic": "Race",
        #  100         "topic": "Privacy & Surveilance",
        #  100         "topic": "Political views",
        #  100         "topic": "Misinformation",
        #  100         "topic": "Guns",
        #  100         "topic": "Global attitudes",
        #  100         "topic": "Gender & Leadership",
        #  100         "topic": "Family & Relationships",
        #  100         "topic": "Economic inequality",
        #  100         "topic": "Community types & sexual harassment",
        #  100         "topic": "Biomedical & food issues",
        #  100         "topic": "Automation and driverless vehicles",
        #  100         "topic": "America in 2050",
        #############################################################

        topicwise_ctr = dict()
        for user_response_json in tqdm(user_responses_jsons, desc="users..."):

            user_id = user_response_json["user_id"]
            topic = user_response_json["topic"]
            if topic not in topicwise_ctr:
                topicwise_ctr[topic] = 0
            topicwise_ctr[topic] += 1

            if early_stopping(topicwise_ctr=topicwise_ctr, max_topics=max_topics, max_users=max_users, topic=topic):
                continue

            if option == -1:
                # implicit_persona = None
                explicit_persona = None
            else:
                # user_profile = user_response_json["implicit_persona"]
                # if self.num_implicit > len(user_profile):
                #     num_implicit = len(user_profile)
                # else:
                #     num_implicit = self.num_implicit
                # # implicit_persona = user_profile[:num_implicit] if option == 0 or option == 2 else None
                explicit_persona = self.mk_explicit_persona(all_traits=user_response_json["explicit_persona"]) if option == 1 or option == 2 else None

            generated_output = []
            all_user_implicit_ques = user_response_json["implicit_questions"]
            limited_user_implicit_ques = utils.take(arr=all_user_implicit_ques, num=max_ques)
            for persona_qa in limited_user_implicit_ques:
                    user_choice= "UNKNOWN"
                    try:
                        user_choice = persona_qa["answer"]
                        choice_idx = persona_qa["choices"].index(user_choice)
                        user_choice = OUTPUT_MAP[choice_idx]
                        model_choice = persona(
                            implicit_persona=self.mk_implicit_persona(option=option, persona_qa=persona_qa, random_qa=user_response_json["implicit_persona"]),
                            explicit_persona=explicit_persona,
                            topic=topic,
                            question=persona_qa["question"],
                            choices=persona_qa["choices"],
                        )
                        generated_output.append({
                            "model_choice": model_choice,
                            "user_choice": user_choice,
                            "qid": persona_qa["qid"],
                        })
                    except Exception as exc:
                        generated_output.append({
                            "model_choice": "UNKNOWN",
                            "user_choice": user_choice,
                            "qid": persona_qa["qid"],
                        })
                        print(f"Exception: {exc}")

            model_generated.append({"user_id": user_id, "topic": topic, "generated_output": generated_output})
        return model_generated

    def mk_explicit_persona(self, all_traits):
        # ideo_demo, ideo, demo
        return filter_demographs(all_demo_ideo=all_traits, filter_on=self.explicit_definition)


def calculate_accuracy(model_generation_path):
    model_generation = read_jsonl_or_json(model_generation_path)
    print("================ model_generation_path: {} ================".format(model_generation_path))
    accuracy_list = []
    user_accuracy_list = []
    for model_output in model_generation:
        user_id = model_output["user_id"]
        generated_output = model_output["generated_output"]

        correct, incorrect = 0, 0
        for response in generated_output:
            model_choice = response['model_choice']
            user_choice = response['user_choice']

            if user_choice == model_choice:
                correct += 1
            else:
                incorrect += 1
        accuracy_per_user = correct / len(generated_output)
        user_accuracy_list.append({user_id: accuracy_per_user})
        accuracy_list.append(accuracy_per_user)
    final_accuracy = sum(accuracy_list) / len(accuracy_list)
    return {"accuracy": final_accuracy, "user-accuracy": user_accuracy_list}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", type=str, default="data/opinionqa/sampled_user_responses_decl.json", help="json path")
    parser.add_argument("--out_dir", type=str, default="data/model-output/", help="json path")
    parser.add_argument("--cache_path", type=str, default="data/cache/gpt_cache.jsonl", help="json path")
    parser.add_argument("--num_implicit", type=int, default=2, help="number of implicit persona to use")
    parser.add_argument("--max_users", type=int, default=35, help="max num users to do inference on.")
    parser.add_argument("--max_ques", type=int, default=30, help="max ques to test inference on.")
    parser.add_argument("--max_topics", type=int, default=-1, help="max topics to test inference on (currently, ~15).")
    parser.add_argument("--max_retries", type=int, default=2, help="max number of openai retries when a call fails.")
    parser.add_argument("--implicit_sampling", type=str, default="random", help="random or topk past opinions")
    parser.add_argument("--explicit_definition", type=str, default="ideo_demo", help="ideo_demo, ideo, demo")
    parser.add_argument("--option", type=int, default=0, choices=[-1, 0, 1, 2], help="-1: no-persona, 0: implicit, 1: explicit, 2: both")
    args = parser.parse_args()
    os.environ["OPENAI_MAX_TRIES_INT"] = str(args.max_retries)

    if args.option == 0:
        dir_name = f"implicit_{args.num_implicit}pts"
    if args.option == 1:
        dir_name = "explicit"
    if args.option == 2:
        dir_name = f"imp-{args.num_implicit}pts_exp"
    if args.option == -1:
        dir_name = f"no-persona"

    dir_name += f"-t{args.max_topics}-u{args.max_users}-q{args.max_ques}"
    if args.explicit_definition != "ideo_demo":
        dir_name += f"-explicit-means-{args.explicit_definition}"

    print(f"\nStart experiment: {dir_name} ...")

    persona = PersonaCreator(engine="text-davinci-003", openai_wrapper=OpenAIWrapper(cache_path=args.cache_path))
    output = PersonalizedQA(args).personalized_qa(
        persona=persona,
        user_responses_jsons=read_jsonl_or_json(args.in_path),
        option=args.option,
        max_users=args.max_users,
        max_ques=args.max_ques,
        max_topics=args.max_topics
    )

    output_dir = os.path.join(args.out_dir, dir_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = os.path.join(output_dir, "model_generation.json")
    write_json(outpath=output_file, json_data=output)

    metrics = calculate_accuracy(output_file)
    metrics_file = os.path.join(output_dir, "model_accuracy.json")
    write_json(outpath=metrics_file, json_data=metrics)