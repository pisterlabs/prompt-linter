from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from datasets import load_dataset
import typer


def annotate(dataset_name, classes):
    llm = OpenAI(temperature=0)
    prompt = PromptTemplate(
        input_variables=["text", "classes"],
        template="Classify text in one of the following classes: {classes}\n\nText: {text}\nClass:",
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    dataset = load_dataset(dataset_name)

    for example in dataset["train"].select(range(5)):
        text = example["text"]
        label = chain.run({"text": text, "classes": classes})
        print(f"{text}\nClass: {label}\n\n")


if __name__ == "__main__":
    typer.run(annotate)
