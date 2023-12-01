import json
import random

from langchain.chains import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser, PydanticOutputParser, RetryWithErrorOutputParser
from pydantic import BaseModel, Field

from cat.mad_hatter.decorators import hook
from cat.log import log


class LLMAnswer(BaseModel):
    support: str = Field(description="the parts of the sentence related to the context, otherwise None")
    other: str = Field(description="the parts of the sentence not related to the context, otherwise None")


@hook
def before_cat_sends_message(message, cat):
    settings = cat.mad_hatter.plugins["stay_on_topic"].load_settings()

    if settings["posterior_check"]:

        context = cat.mad_hatter.execute_hook("agent_prompt_declarative_memories",
                                              cat.working_memory["declarative_memories"])

        # ccat_schema = ResponseSchema(name="support",
        #                              description=
        #                              "here the parts of the sentence related to the context, otherwise None")
        # other_schema = ResponseSchema(name="other",
        #                               description=
        #                               "here the parts of the sentence not related to the context, otherwise None")
        # schema = [ccat_schema, other_schema]
        # output_parser = StructuredOutputParser.from_response_schemas(schema)

        parser = PydanticOutputParser(pydantic_object=LLMAnswer)
        # retry_parser = RetryWithErrorOutputParser.from_llm(
        #     parser=parser, llm=cat.llm
        # )
        # Support: a sentence about {settings["topic_description"]}.
        #         Other: a sentence that is asking something general, different from the previous topics.
        #
        template = f"""
        Given this context --> {{context}}
        Split this input based on topics --> {{text}}

        {{format_instructions}}"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "text"],
            output_parser=parser,
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        prompt_value = prompt.format_prompt(context=context, text=message["content"])

        chain = LLMChain(
            llm=cat._llm,
            prompt=prompt,
            verbose=True
        )

        answer = chain.run({"context": context, "text": message["content"]})
        log(answer, "CRITICAL")
        # answer = answer.replace("null", "None")
        # answer = eval(answer)
        output_dict = parser.parse(answer)
        formatted_answer = output_dict.dict()
        log(formatted_answer, "ERROR")

        # prompt = f"""Rewrite the sentence in a JSON with this format:
        #                     {{
        #                         'support': here the parts of the sentence related to the context, otherwise None
        #                         'other': here the parts of the sentence not related to the context, otherwise None
        #                     }}
        #                 SENTENCE --> {message["content"]}
        #                 CONTEXT --> {context}
        #             """
        #
        # answer = cat.llm(prompt)
        # try:
        #     answer = answer.replace("null", "None")
        #     log(answer, "CRITICAL")
        #     json_answer = json.loads(answer)
        #     log(json_answer, "ERROR")
        #
        #     message["content"] = json_answer["support"]
        # except:
        #
        #
        # if json_answer["support"] is None:
        #     message["content"] = random.choice([
        #         "Sorry, I have no memories about that.",
        #         "I can't help you on this topic.",
        #         "A plugin oblige me to stay on topic.",
        #         "I can't talk about that."
        #     ])

    return message
