
# -- imports 
from pydantic import BaseModel, Field

from langchain.prompts import PromptTemplate

from langchain.output_parsers import StructuredOutputParser, ResponseSchema, PydanticOutputParser, RetryWithErrorOutputParser, OutputFixingParser



# ==================================================
# -- Report: Outline
# ==================================================

class Outline(BaseModel):
  section_titles: list[str] = Field(..., description="list of section titles that best represent the logical foundation of a report based on provide source material.")
  section_descriptions: list[str] = Field(..., description="list of section descriptions that best represent the logical foundation of a report based on provide source material.")


def generateOutline_(llm, summary, objective_topic = ''): 

  output_parser = PydanticOutputParser(pydantic_object=Outline)

  format_instructions = output_parser.get_format_instructions()

  prompt = PromptTemplate(
      #template="Provided is an objective topic of a report and detailed summary of source materials. Pretend you are a world class report writer. Generate the section titles and descriptions for each section that represents the foundation of a world class report on the source material. The descriptions should be detailed enough to for a writer to fully understand what information should be included.  \n{format_instructions}\n{objective_topic}\n{summary}",
      template="You are a brilliant writer who is confident and writes in an a logical, engaging tone. Provided is an objective topic of a report and detailed summary of source materials. Generate the section titles and descriptions for each section that represents the foundation of a world class report on the source material. The descriptions should be detailed enough to for a writer to fully understand what information should be included.  \n{format_instructions}\n{objective_topic}\n{summary}",
      input_variables=["objective_topic","summary"],
      partial_variables={"format_instructions": format_instructions}
  )

  model = llm 

  input_ = prompt.format_prompt(objective_topic=objective_topic, summary=summary)
  output = model(input_.to_string())

  output = output_parser.parse(output)

  return output

def generateKeyEventOutline_(llm, raw_text, objective_topic = ''): 

  output_parser = PydanticOutputParser(pydantic_object=Outline)

  format_instructions = output_parser.get_format_instructions()

  prompt = PromptTemplate(
      template="""You are a brillant editor for a world class news organization. Provided is the combined titles and summaries of various events. Identify each distinct event and generate the section titles and descriptions for each distinct event.
       
        The descriptions should be detailed enough to for a writer to fully understand what information should be included in a short article. ,
        
        \n{format_instructions}\n{objective_topic}\n{raw_text}""",

      input_variables=["objective_topic","raw_text"],
      partial_variables={"format_instructions": format_instructions}
  )

  model = llm 

  input_ = prompt.format_prompt(objective_topic=objective_topic, raw_text=raw_text)
  output = model(input_.to_string())

  output = output_parser.parse(output)

  return output
