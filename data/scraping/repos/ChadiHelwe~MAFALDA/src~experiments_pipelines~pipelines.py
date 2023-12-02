import json
import logging
import os
from collections import OrderedDict
from typing import Any

from langchain.chains import ConversationChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from tqdm import tqdm

from src.classification_models.quantized_llama_based_models import (
    LLaMABasedQuantizedModel,
)
from src.experiments_pipelines.chatbot import ChatBotLLM
from src.process_docanno_data import process_data
from src.utils import read_jsonl

PROMPT = {
    0: """
    {instruction_begin}

    Text: "{example_input}"

    Determine whether the following sentence contains a fallacy or not:
    
    Sentence: "{sentence_input}" {instruction_end}
    
    Output:
    
    """,
    1: """
    {instruction_begin}

    Text: "{example_input}"

    Based on the above text, identify the fallacy (if any) in the following sentence. If a fallacy is present, specify the type(s) of fallacy without providing explanations. The possible types of fallacy are:
    - appeal to emotion
    - fallacy of logic
    - fallacy of credibility    
    
    Sentence: "{sentence_input}" {instruction_end}
    
    Output:
        
    """,
    2: """
    {instruction_begin}

    Text: "{example_input}"

    Based on the above text, identify the fallacy (if any) in the following sentence. If a fallacy is present, specify the type(s) of fallacy without providing explanations. The possible types of fallacy are:
    - appeal to positive emotion
    - appeal to anger
    - appeal to fear
    - appeal to pity
    - appeal to ridicule
    - appeal to worse problems
    - causal oversimplification
    - circular reasoning
    - equivocation
    - false analogy
    - false causality
    - false dilemma
    - hasty generalization
    - slippery slope
    - straw man
    - fallacy of division
    - ad hominem
    - ad populum
    - appeal to (false) authority
    - appeal to nature
    - appeal to tradition
    - guilt by association
    - tu quoque
    
    Sentence: "{sentence_input}" {instruction_end}
    
    Output:
    """,
}


ALT_PROMPT = {
    0: """
    {instruction_begin}

    Definitions:
    - An argument consists of an assertion called the conclusion and one or more assertions called premises, where the premises are intended to establish the truth of the conclusion. Premises or conclusions can be implicit in an argument.
    - A fallacious argument is an argument where the premises do not entail the conclusion.

    Text: "{example_input}"

    Based on the above text, determine whether the following sentence is part of a fallacious argument or not:
    
    Sentence: "{sentence_input}" {instruction_end}
    
    Output:
    
    """,
    1: """
    {instruction_begin}

    Definitions:
    - An argument consists of an assertion called the conclusion and one or more assertions called premises, where the premises are intended to establish the truth of the conclusion. Premises or conclusions can be implicit in an argument.
    - A fallacious argument is an argument where the premises do not entail the conclusion.

    Text: "{example_input}"

    Based on the above text, determine whether the following sentence is part of a fallacious argument or not. If it is, indicate the type(s) of fallacy without providing explanations. The potential types of fallacy include:
    - appeal to emotion
    - fallacy of logic
    - fallacy of credibility    
    
    Sentence: "{sentence_input}" {instruction_end}
    
    Output:
        
    """,
    2: """
    {instruction_begin}

    Definitions:
    - An argument consists of an assertion called the conclusion and one or more assertions called premises, where the premises are intended to establish the truth of the conclusion. Premises or conclusions can be implicit in an argument.
    - A fallacious argument is an argument where the premises do not entail the conclusion.
    
    Text: "{example_input}"

    Based on the above text, determine whether the following sentence is part of a fallacious argument or not. If it is, indicate the type(s) of fallacy without providing explanations. The potential types of fallacy include:
    - appeal to positive emotion
    - appeal to anger
    - appeal to fear
    - appeal to pity
    - appeal to ridicule
    - appeal to worse problems
    - causal oversimplification
    - circular reasoning
    - equivocation
    - false analogy
    - false causality
    - false dilemma
    - hasty generalization
    - slippery slope
    - straw man
    - fallacy of division
    - ad hominem
    - ad populum
    - appeal to (false) authority
    - appeal to nature
    - appeal to tradition
    - guilt by association
    - tu quoque
    
    Sentence: "{sentence_input}" {instruction_end}
    
    Output:
    """,
}


def zero_or_few_shots_pipeline(
    model: LLaMABasedQuantizedModel,
    dataset_path: str = None,
    prediction_path: str = None,
    level: int = 0,
    alt_prompt: bool = True,
):
    logger = logging.getLogger("MafaldaLogger")

    if alt_prompt:
        prompt = PromptTemplate(
            input_variables=[
                "example_input",
                "sentence_input",
                "instruction_begin",
                "instruction_end",
            ],
            template=ALT_PROMPT[level],
        )
    else:
        prompt = PromptTemplate(
            input_variables=[
                "example_input",
                "sentence_input",
                "instruction_begin",
                "instruction_end",
            ],
            template=PROMPT[level],
        )

    chatbot_model = ChatBotLLM(model=model)
    if model.model_name == "gpt-3.5":
        chatbot_model.max_length = 1024
    chatbot_chain = LLMChain(llm=chatbot_model, prompt=prompt)

    data = read_jsonl(dataset_path)
    processed_data = process_data(data)
    assert len(data) == len(
        processed_data
    ), f"Data length mismatch: {len(data)} != {len(processed_data)}"

    # Check already processed examples
    already_processed = set()
    if os.path.exists(prediction_path):
        with open(prediction_path, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    already_processed.add(entry["text"])
                except json.JSONDecodeError:
                    # Handle improperly formatted last line
                    f.seek(0)
                    all_lines = f.readlines()
                    with open(prediction_path, "w") as fw:
                        fw.writelines(all_lines[:-1])

    with open(prediction_path, "a") as f:
        for example, processed_example in tqdm(
            zip(data, processed_data), total=len(data)
        ):
            if example["text"] in already_processed:
                logger.info(f"Skipping already processed example: {example['text']}")
                continue

            logger.info(example["text"])
            # pred_outputs = '{"prediction": {'
            pred_outputs = OrderedDict()
            for s in processed_example:
                logger.info(s)
                output = chatbot_chain.run(
                    example_input=example["text"],
                    sentence_input=s,
                    instruction_begin=model.instruction_begin,
                    instruction_end=model.instruction_end,
                )
                logger.info(output)
                # pred_outputs += f'"{s}": "{output}",'
                pred_outputs[s] = output
            # pred_outputs = pred_outputs[:-1] + "}}"
            json_line = json.dumps(
                {
                    "text": example["text"],
                    "prediction": pred_outputs,
                }
            )
            f.write(json_line + "\n")
