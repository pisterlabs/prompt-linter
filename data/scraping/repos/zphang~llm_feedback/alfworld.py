import tqdm.auto as tqdm
from typing import List, Dict, Optional
import pandas as pd

from langchain.schema import (
    SystemMessage, HumanMessage, AIMessage
)

from .base import BaseTask
from ...utils.models import get_chat_model

MAX_NUM_STEPS = 50


class AlfworldTask(BaseTask):
    """Alfworld task.

    Requires installation of https://github.com/alfworld/alfworld.
    """

    def get_dataset(self, phase: str):
        """We return an environment object instead of a dataset.

        Call iterate_envs to iterate over the env obj.
        """
        import alfworld.agents.environment
        minimal_config = {
            "env": {
                "domain_randomization": False,
                "goal_desc_human_anns_prob": 0.0,
                "regen_game_files": False,
                "task_types": [1, 2, 3, 4, 5, 6],
                "expert_type": "handcoded",
            },
            "logic": {
                "domain": '$ALFWORLD_DATA/logic/alfred.pddl',
                "grammar": '$ALFWORLD_DATA/logic/alfred.twl2',
            },
            "dataset": {
                "data_path": '$ALFWORLD_DATA/json_2.1.1/train',
                "eval_id_data_path": '$ALFWORLD_DATA/json_2.1.1/valid_seen',
                "eval_ood_data_path": '$ALFWORLD_DATA/json_2.1.1/valid_unseen',
                "num_eval_games": -1,
            },
            "general": {
                "training_method": 'dagger',
            },
            "dagger": {
                "training": {
                    "max_nb_steps_per_episode": 50,
                }
            }
        }
        if phase == "validation":
            train_eval = "eval_out_of_distribution"
        else:
            raise KeyError(phase)
        env = alfworld.agents.environment.AlfredTWEnv(minimal_config, train_eval=train_eval)
        env = env.init_env(batch_size=1)
        return AlfworldFakeDataset(env)

    def get_chain(self, generation_llm: str, feedback_llm: str, refinement_llm: str,
                  chain_name: Optional[str] = None):
        # We'll have a specific function for this.
        return {
            "initial": AlfworldAgentChain(model=get_chat_model(generation_llm)),
            "feedback": get_chat_model(feedback_llm),
            "refinement": AlfworldAgentChain(model=get_chat_model(refinement_llm)),
        }

    def process_all(self, chain, dataset, max_num_examples: int):
        all_outputs = []
        for i, env_info in zip(tqdm.trange(max_num_examples), dataset.iterate_envs()):
            all_outputs.append(self.process_env(chain=chain, env_info=env_info))
        return all_outputs

    def process_env(self, chain, env_info):
        initial_out = self.tackle_env(agent_chain=chain["initial"], env_info=env_info)
        messages = construct_messages(initial_out["history"])
        if initial_out["reward"] == 0:
            messages[-1].content += "\nYou did not succeed in completing the task. Based on the above, what advice would you give to the next person who tries this task?"
        else:
            messages[-1].content += "\nYou were successful at the task. Based on the above, what advice would you give to the next person who tries this task?"
        feedback = chain["feedback"].predict_messages(messages).content
        refinement_out = self.tackle_env(agent_chain=chain["refinement"], env_info=env_info)
        return {
            "initial": initial_out,
            "feedback": feedback,
            "refinement": refinement_out,
        }

    @classmethod
    def tackle_env(cls, agent_chain: "AlfworldAgentChain", env_info):
        history = [{"obs": env_info["obs"]}]
        commands = env_info["commands"]
        env = env_info["env"]
        reward = 0

        for _ in range(MAX_NUM_STEPS):
            raw_act_idx_plus_one = agent_chain.process(history=history, commands=commands)
            act_idx_plus_one = get_num(raw_act_idx_plus_one)
            if act_idx_plus_one > len(commands):
                print(f"WARNING: act_idx_plus_one={act_idx_plus_one} but len(commands)={len(commands)}")
                act_idx_plus_one = 1
            act_idx = act_idx_plus_one - 1
            act = commands[act_idx]

            obs, reward, done, info = env.step([act])
            obs = process_ob(obs[0])
            done = done[0]
            if done:
                print("DONE")
                break
            commands = info["admissible_commands"][0]
            history.append({"obs": obs, "act_idx_plus_one": act_idx_plus_one, "act": act})
        return {
            "history": history,
            "reward": reward[0],
        }

    def evaluate(self, phase: str, outputs: List[Dict]):
        # This is a terrible evaluation metric, but it's just an example.
        # In practice we need to parse the output and get the answer.
        scores = {"initial_score": [], "refined_score": []}
        for row in outputs:
            scores["initial_score"].append(row["initial"]["reward"])
            scores["refined_score"].append(row["refinement"]["reward"])
        return {
            "initial_score": float(pd.Series(scores["initial_score"]).mean()),
            "refined_score": float(pd.Series(scores["refined_score"]).mean()),
        }


class AlfworldFakeDataset:
    def __init__(self, env):
        self.env = env

    def __len__(self):
        return len(self.env.gamefiles)

    def iterate_envs(self):
        for _ in range(len(self)):
            ob, info = self.env.reset()
            yield {
                "obs": ob[0],
                "commands": info["admissible_commands"][0],
                "env": self.env,
            }


class AlfworldAgentChain:
    def __init__(self, model):
        self.model = model

    def query_model(self, messages):
        return self.model.predict_messages(messages, stop=["\n"]).content

    def process(self, history, commands, feedback=None):
        return self.query_model(construct_messages(
            history=history, commands=commands, feedback=feedback,
        ))


def construct_messages(history, commands=None, feedback=None):
    messages = [
        SystemMessage(
            content="You are an assistant playing a text-based game. You can only respond by returning the number corresponding to an allowed action."),
    ]
    for hist_row in history:
        if "act" not in hist_row:
            # First observation
            if feedback:
                messages.append(HumanMessage(content="{obs}".format(**hist_row)))
            else:
                messages.append(HumanMessage(content="{obs}\n(NOTE: The following feedback was provided on a previous attempt.\n\n{feedback}\n\nPlease take the above into account.)".format(
                    obs=hist_row["obs"],
                    feedback=feedback,
                )))
        else:
            messages.append(AIMessage(content="{act_idx_plus_one}\n".format(**hist_row)))
            messages.append(
                HumanMessage(content="You chose: {act_idx_plus_one} - {act}\n{obs}".format(**hist_row)))
    if commands:
        assert isinstance(messages[-1], HumanMessage)
        messages[-1].content += "\n[Choose one action]\n" + "\n".join(
            f"{i}: {x}" for i, x in enumerate(commands, start=1)) + f"\nChoice ({1}-{len(commands)}):"
    return messages


def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]
    return ob


def get_num(string):
    string = string.split()
    ls = []
    for s in string:
        if s.isdigit():
            ls.append(s)
        else:
            break
    if ls:
        return int("".join(ls))
    else:
        return 1
