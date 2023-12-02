from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field, validator
from chains import llm


# Define your desired data structure.
class RephrasedPost(BaseModel):
    post: str = Field(description="the rewritten post")
    
    # You can add custom validation logic easily with Pydantic.
    @validator('post')
    def starts_with_rewritten(cls, field):
        field = field.strip()
        if field.startswith("Rewritten Post:"):
            return field[16:]
        return field


parser = PydanticOutputParser(pydantic_object=RephrasedPost)

prompt = PromptTemplate(
    input_variables=["post"],
    template="""You are an editor for reddit posts. Your job is to rewrite an individual user's Reddit post to be less inflammatory and toxic while maintaining the original intention and stances in their post. Provide a rewritten version of their post that satisfies these parameters. Do not add any text except for the rewritten post.
{format_instructions}
Original Post: {post}""",
    partial_variables={"format_instructions": parser.get_format_instructions()}
)


openAIChain = LLMChain(llm=llm, prompt=prompt)
runOpenAIChain = lambda x: parser.parse(openAIChain.run(x)).dict()