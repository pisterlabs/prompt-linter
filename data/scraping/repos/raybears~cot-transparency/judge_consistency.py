import asyncio
from enum import Enum
from pathlib import Path
from typing import Optional, Sequence

from grugstream import Observable
from openai import InvalidRequestError
from pydantic import BaseModel
from slist import Slist
from tqdm import tqdm

from cot_transparency.apis import UniversalCaller
from cot_transparency.apis.base import ModelCaller, Prompt, CachedPerModelCaller
from cot_transparency.apis.openai import OpenAICompletionPrompt
from cot_transparency.apis.openai.finetune import FinetuneSample
from cot_transparency.data_models.config import OpenaiInferenceConfig
from cot_transparency.data_models.messages import ChatMessage, MessageRole
from scripts.load_alpaca_dataset import get_alpaca_user_testing


class ComparisonGeneration(BaseModel):
    prompt: Prompt
    a_config: OpenaiInferenceConfig
    b_config: OpenaiInferenceConfig
    a_response: str
    b_response: str


class JudgeChoice(str, Enum):
    model_a = "model_1"
    model_b = "model_b"


class ComparisonGenerationJudged(BaseModel):
    generation: ComparisonGeneration
    judge_prompt: Sequence[ChatMessage]
    judge_output: str
    winner: Optional[JudgeChoice]


class BothJudgements(BaseModel):
    first_judgement: Optional[ComparisonGenerationJudged]  # E.g. not enough tokens
    second_judgement: Optional[ComparisonGenerationJudged]
    judge_config: OpenaiInferenceConfig

    def is_consistent(self) -> Optional[bool]:
        if self.first_judgement is None or self.second_judgement is None:
            return None
        if self.first_judgement.winner is None or self.second_judgement.winner is None:
            return None
        return self.first_judgement.winner == self.second_judgement.winner


def generate_comparison(
    prompt: Prompt,
    vanilla_caller: ModelCaller,
    intervention_caller: ModelCaller,
    vanilla_config: OpenaiInferenceConfig,
    intervention_config: OpenaiInferenceConfig,
) -> ComparisonGeneration:
    vanilla_response = vanilla_caller.call(messages=prompt.messages, config=vanilla_config)
    intervention_response = intervention_caller.call(messages=prompt.messages, config=intervention_config)
    return ComparisonGeneration(
        prompt=prompt,
        a_config=vanilla_config,
        b_config=intervention_config,
        a_response=vanilla_response.single_response,
        b_response=intervention_response.single_response,
    )


def finetune_sample_to_prompt(sample: FinetuneSample) -> Prompt:
    messages = [m.to_chat_message() for m in sample.messages]
    # the last message is the one we want to predict
    messages_without_last = messages[:-1]
    return Prompt(messages=messages_without_last)


class PromptWithModel(BaseModel):
    prompt: Prompt
    config: OpenaiInferenceConfig


class QuestionWithLabels(BaseModel):
    question: ChatMessage
    first_label: JudgeChoice
    second_label: JudgeChoice


def judge_question(comparison: ComparisonGeneration, model_a_first: bool) -> QuestionWithLabels:
    a_first: bool = model_a_first
    first_text = comparison.a_response if a_first else comparison.b_response
    second_text = comparison.b_response if a_first else comparison.a_response
    nice_text_instruction = OpenAICompletionPrompt(messages=comparison.prompt.messages).format()
    text = f"""You need to judge which response follows the instruction better
Please verbalize how you are thinking about the problem, then give your answer in the format "the best response that follows the instruction better is the". It's very important that you stick to this format.
Please only output one of the following as your answer. 
- the best response that follows the instruction better is the first
- the best response that follows the instruction better is the second
Instruction:
{nice_text_instruction}

First response: {first_text}
Second response: {second_text}

Let's think step by step:"""

    message = ChatMessage(role=MessageRole.user, content=text)
    return QuestionWithLabels(
        question=message,
        first_label=JudgeChoice.model_a if a_first else JudgeChoice.model_b,
        second_label=JudgeChoice.model_b if a_first else JudgeChoice.model_a,
    )


def parse_judge_output(judge_output: str, first_label: JudgeChoice, second_label: JudgeChoice) -> Optional[JudgeChoice]:
    if "better is the first" in judge_output.lower():
        return first_label
    if "better is the: first" in judge_output.lower():
        return first_label
    if "better is the second" in judge_output.lower():
        return second_label
    if "better is the: second" in judge_output.lower():
        return second_label
    # first_word = judge_output.split()[0]
    # if "first" in first_word.lower():
    #     return first_label
    # elif "second" in first_word.lower():
    #     return second_label
    else:
        print(f"Could not parse judge output {judge_output}")
        return None


def get_judge_output(
    comparison: ComparisonGeneration, judge: ModelCaller, model_a_first: bool, judge_config: OpenaiInferenceConfig
) -> ComparisonGenerationJudged:
    question = judge_question(comparison, model_a_first=model_a_first)
    judge_response: str = judge.call(
        messages=[question.question],
        config=judge_config,
    ).single_response
    winner = parse_judge_output(
        judge_response,
        first_label=question.first_label,
        second_label=question.second_label,
    )
    return ComparisonGenerationJudged(
        generation=comparison,
        judge_output=judge_response,
        winner=winner,
        judge_prompt=[question.question],
    )


def get_judge_output_both(
    comparison: ComparisonGeneration, judge: ModelCaller, judge_config: OpenaiInferenceConfig
) -> BothJudgements:
    try:
        first = get_judge_output(comparison, judge, model_a_first=True, judge_config=judge_config)
    except InvalidRequestError:
        first = None
    try:
        second = get_judge_output(comparison, judge, model_a_first=False, judge_config=judge_config)
    except InvalidRequestError:
        second = None
    return BothJudgements(first_judgement=first, second_judgement=second, judge_config=judge_config)


def eval_judge_group_by_model(judged: Sequence[BothJudgements]) -> None:
    # group by judge config
    Slist(judged).group_by(lambda j: j.judge_config.model).for_each(lambda group: eval_judged(group.values))
    return None


def eval_judged(judged: Sequence[BothJudgements]) -> None:
    # Assert unique
    unique_judge: str = Slist(judged).map(lambda j: j.judge_config.model).distinct_item_or_raise(lambda x: x)
    print(f"=====Evaluation for judge {unique_judge}=====")
    # Calculate the consistency
    valid = Slist(judged).map(lambda j: j.is_consistent()).flatten_option()
    print(f"Total judged: {len(valid)} out of {len(judged)}")
    average_consistency = valid.average_or_raise()
    print(f"Average consistency: {average_consistency:2f}")
    bias = 1 - average_consistency
    print(f"Bias: {bias:2f}")


class WinrateMetrics(BaseModel):
    win_rate: float
    se: float
    samples: int


# def get_win_rate(judged: Sequence[ComparisonGenerationJudged]) -> WinrateMetrics:
#     judged_slist = Slist(judged).filter(
#         lambda j: abs(len(j.generation.model_a_response) - len(j.generation.model_b_response))
#     )
#     print(f"Total judged: {len(judged_slist)}")
#     winner_vanilla = len(judged_slist.filter(lambda j: j.winner == JudgeChoice.model_a))
#     print(f"Total winner vanilla: {winner_vanilla}")
#     winner_intervention = len(judged_slist.filter(lambda j: j.winner == JudgeChoice.model_b))
#     print(f"Total winner intervention: {winner_intervention}")
#     # Intervention win rate
#     win_rate = winner_intervention / (winner_vanilla + winner_intervention)
#     # get the standard error
#     se = (win_rate * (1 - win_rate) / (winner_vanilla + winner_intervention)) ** 0.5
#     return WinrateMetrics(win_rate=win_rate, se=se, samples=winner_vanilla + winner_intervention)


# def win_rate_plot(
#     judged: Sequence[ComparisonGenerationJudged],
#     sort_by: Sequence[str],
#     rename_map: Mapping[str, str],
# ) -> None:
#     win_rates: Slist[tuple[str, WinrateMetrics]] = (
#         (
#             Slist(judged)
#             .group_by(
#                 # group by model
#                 lambda j: j.generation.model_b_config.model,
#             )
#             .map_2(
#                 # get win rate
#                 lambda model, judged: (model, get_win_rate(judged)),
#             )
#         )
#         .sort_by(
#             # sort by model
#             lambda model_win_rate: sort_by.index(model_win_rate[0]),
#         )
#         .map(
#             # rename
#             lambda model_win_rate: (
#                 rename_map.get(model_win_rate[0], model_win_rate[0]),
#                 model_win_rate[1],
#             ),
#         )
#     )
#     # use seaborn

#     # create a dataframe
#     list_dicts = [
#         {
#             "model": model,
#             "win_rate": win_rate.win_rate,
#             "se": win_rate.se,
#             "samples": win_rate.samples,
#         }
#         for model, win_rate in win_rates
#     ]

#     # errors: list[float] = [win_rate.se for _, win_rate in win_rates]
#     df = pd.DataFrame(list_dicts)
#     ax = sns.barplot(x="model", y="win_rate", data=df)

#     # set y axis to 0-1
#     plt.ylim(0, 1)
#     # add standard error bars
#     plt.errorbar(
#         x=df["model"],
#         y=df["win_rate"],
#         yerr=df["se"],
#         fmt="none",
#         capsize=0.1,
#         color="black",
#     )

#     # x-axis should be "Percentage of additional instruction dataset samples"
#     # title should be "Win rate of intervention models against gpt-3.5-turbo"
#     ax.set(xlabel="Percentage of additional instruction dataset samples", ylabel="Win rate")
#     ax.set_title("Win rate of intervention models against gpt-3.5-turbo\n on the Huggingface Instruction Eval dataset")
#     # add a red dotted line at 50%
#     plt.axhline(y=0.5, color="r", linestyle="--")

#     plt.show()


def observable_for_judges(
    judge_models: list[str], caller: ModelCaller, samples: Slist[FinetuneSample]
) -> Observable[BothJudgements]:
    # First model is claude-2
    vanilla_config = OpenaiInferenceConfig(model="claude-2", max_tokens=2000, temperature=0.0, top_p=1.0)
    judge_configs: list[OpenaiInferenceConfig] = [
        OpenaiInferenceConfig(model=judge_model, max_tokens=2000, temperature=0.0, top_p=1.0)
        for judge_model in judge_models
    ]
    intervention_config = OpenaiInferenceConfig(model="claude-instant-1.2", max_tokens=2000, temperature=0.0, top_p=1.0)
    pipeline = (
        Observable.from_iterable(samples)
        .map_blocking_par(
            lambda prompt: generate_comparison(
                prompt=finetune_sample_to_prompt(prompt),
                vanilla_caller=caller,
                intervention_caller=caller,
                vanilla_config=vanilla_config,
                intervention_config=intervention_config,
            )
        )
        .map_blocking_par(
            lambda comparison: [
                get_judge_output_both(comparison, judge=caller, judge_config=judge_config)
                for judge_config in judge_configs
            ],
            max_par=40,
        )
        .flatten_list()
    )
    # run it
    return pipeline


def many_judge_obs(judge_models: list[str], caller: ModelCaller) -> Observable[BothJudgements]:
    samples: Slist[FinetuneSample] = get_alpaca_user_testing(3000)
    print(f"Total testing samples: {len(samples)}")
    tq = tqdm(total=len(judge_models) * samples.length)
    return observable_for_judges(judge_models=judge_models, caller=caller, samples=samples).tqdm(tqdm_bar=tq)


async def eval_instruction_following(judge_models: list[str]):
    # ft:gpt-3.5-turbo-0613:academicsnyuperez::8B24hv5w 10k samples, 0% instruction
    # ft:gpt-3.5-turbo-0613:academicsnyuperez::89ghXobC 100k samples, 10% instruction

    caller: CachedPerModelCaller = UniversalCaller().with_model_specific_file_cache(
        cache_dir=Path("experiments/judge_consistency"),
        write_every_n=100,
    )

    pipeline = many_judge_obs(judge_models=judge_models, caller=caller)
    caller.save_cache()

    results: Slist[BothJudgements] = await pipeline.to_slist()
    eval_judge_group_by_model(results)


if __name__ == "__main__":
    # 100k ft:gpt-3.5-turbo-0613:academicsnyuperez::89ghXobC, 10 % ft:gpt-3.5-turbo-0613:academicsnyuperez::89ghXobC
    # 1k ft:gpt-3.5-turbo-0613:academicsnyuperez::8CDxzKfb 10x instruct
    # 1k ft:gpt-3.5-turbo-0613:academicsnyuperez::8CE4CPmg 1x instruct
    # 1k ft:gpt-3.5-turbo-0613:academicsnyuperez::8CEGJGjq 0.1x instruct
    # 1k ft:gpt-3.5-turbo-0613:academicsnyuperez::8CDdvsrO 0x instruct
    """
    ModelMeta(
            model="ft:gpt-3.5-turbo-0613:academicsnyuperez::8LJ52csT",
            trained_samples=1_000,
            trained_on=TrainedOn.CONTROL_UNBIASED_CONTEXTS,
        ),
        ModelMeta(
            model="ft:gpt-3.5-turbo-0613:academicsnyuperez::8LUIUfUe",
            trained_samples=10_000,
            trained_on=TrainedOn.CONTROL_UNBIASED_CONTEXTS,
        ),
        ModelMeta(
            model="ft:gpt-3.5-turbo-0613:academicsnyuperez::8LEegGiG",
            trained_samples=1_000,
            trained_on=TrainedOn.CONSISTENCY_BIASED_CONTEXTS,
        ),
        ModelMeta(
            model="ft:gpt-3.5-turbo-0613:academicsnyuperez::8LSii3Tv",
            trained_samples=10_000,
            trained_on=TrainedOn.CONSISTENCY_BIASED_CONTEXTS,
        ),
    """
    asyncio.run(
        eval_instruction_following(
            judge_models=[
                "gpt-3.5-turbo-0613",
                "ft:gpt-3.5-turbo-0613:academicsnyuperez::8PECNa47",  # control 90% cot
                "ft:gpt-3.5-turbo-0613:academicsnyuperez::8PEGH82V",  # intervention zeroshot 90% cot
            ]
        )
    )
