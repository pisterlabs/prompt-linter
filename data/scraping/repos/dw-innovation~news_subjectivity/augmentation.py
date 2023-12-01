import random
import os
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
from langchain import PromptTemplate, LLMChain, HuggingFaceHub
from langchain.llms import HuggingFacePipeline, OpenAI
from dotenv import load_dotenv
from loguru import logger

logger.add(f"{__name__}.log", rotation="500 MB")


def augment_data(dataset_path,
                 model,
                 style,
                 language,
                 model_name,
                 output_dir):
    with open(Path('templates') / f'{language}.txt') as f:
        template = f.read()

    prompt = PromptTemplate(
        input_variables=["sentence", "style"],
        template=template)

    if model == 'openai':
        load_dotenv()
        os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
        logger.info(os.environ.get("OPENAI_API_KEY"))

        llm = OpenAI(model_name=model_name,
                     openai_api_key=os.environ.get("OPENAI_API_KEY"),
                     temperature=0.9,  # we chose 0.9 as temperature for creative answers
                     request_timeout=250
                     )

    llm_chain = LLMChain(llm=llm, prompt=prompt, output_key='final_output')
    dataset = pd.read_csv(dataset_path, sep='\t')

    num_subj_samples = len(dataset[dataset["label"] == 'SUBJ'])
    num_obj_samples = len(dataset[dataset["label"] == 'OBJ'])

    num_augmented_samples = num_obj_samples - num_subj_samples
    logger.info(f'{num_augmented_samples} samples in {dataset_path} to augment')

    random.seed(0)

    if style == 'normal':
        '''If no style is used, we augment from the subjective samples which are the minority'''
        indices = random.sample(range(0, num_subj_samples - 1), num_augmented_samples)
        samples = dataset[dataset["label"] == 'SUBJ']
    else:
        indices = random.sample(range(0, num_obj_samples - 1), num_augmented_samples)
        samples = dataset[dataset["label"] == 'OBJ']

    augmented_data = []
    for index in indices:
        sentence = samples.iloc[index]["sentence"]

        generated_text = llm_chain.run(sentence=sentence, style=style)
        augmented_data.append({"generated_text": generated_text,
                               "original": sentence})

    logger.info(f"Augmentation is for {language} done. It is saved to {output_dir}")
    augmented_data = pd.DataFrame(augmented_data)
    augmented_data.to_csv(Path(output_dir) / f"{style}.tsv", sep='\t', index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset')
    parser.add_argument('--model')
    parser.add_argument('--model_name', choices=['text-davinci-003', 'gpt-3.5-turbo'])
    parser.add_argument('--style')
    parser.add_argument('--output_dir')
    parser.add_argument('--language')

    args = parser.parse_args()

    augment_data(dataset_path=args.dataset,
                 model=args.model,
                 style=args.style,
                 model_name=args.model_name,
                 language=args.language,
                 output_dir=args.output_dir)
