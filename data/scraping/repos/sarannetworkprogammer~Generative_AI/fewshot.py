from langchain import FewShotPromptTemplate

from langchain.llms import OpenAI
from langchain.chains import LLMChain



from langchain.prompts import PromptTemplate 

from dotenv import load_dotenv


load_dotenv()


def main():
    examples = [
        {"name":"Spotify","category":"Music"},
        {"name":"Instagram","category":"social"},
        {"name":"TikTok","category":"social"},
        {"name":"Tinder","category":"dating"},

    ]


    app_name_prompt_template = PromptTemplate(
        input_variables = ["name","category"],
        template="""
        Category = {category}
        App name: {name}

        """
    )

    few_shot_template = FewShotPromptTemplate(
        examples = examples,
        example_prompt = app_name_prompt_template,
        prefix ="Create names for apps based on their category",
        suffix = "Category: {category}\nApp name:",
        input_variables =["category"],
        example_separator ="\n"
    )

    llm = OpenAI(verbose=True, temperature =0.1)

    chain = LLMChain(llm=llm, prompt= few_shot_template)

    app_name = chain.run("music")

    print(app_name)

if __name__ == "__main__":
    main()