
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

def loadOutlinePrompt(summary, context, topic): 
  # -- define prompt 
  output_parser = PydanticOutputParser(pydantic_object=Outline)

  prompt_template = """You are a brilliant writer who is confident and writes in an a logical, engaging tone. Provided is a report's topic objective and detailed summary of source materials. Generate the section titles and descriptions for each section that represents the foundation of a world class report on the source material. The descriptions should be detailed enough to for a writer to fully understand what information should be included.:
  Report Scope: {topic}
  Report Summary: {summary}
  Research Context: {context}
  Report Outline:"""

  # -- define inputs
  inputs = {'topic': topic, 'context': context, 'summary': summary}

  return prompt_template, inputs, output_parser

# ==================================================
# -- Writing tools 
# ==================================================

def loadSummaryPrompt(topic, context, word_count = 300): 
  # -- define prompt 
  prompt_template = """You are a brilliant writer who is confident and writes in an a logical, engaging tone. Provided is the topic scope of a report. Use the context below to write a detailed abstract summary of the report within the target word count:
  Report Topic: {topic}
  Target Word Count: {word_count}
  Context: {context}
  Report Summary:"""

  # -- define inputs
  inputs = {'topic': topic, 'context': context, 'word_count': word_count}

  return prompt_template, inputs, None

def loadExpandPrompt(objective, original_text, context, word_count = 0): 
  word_count = len(original_text.split(' ')) * 1.5 if word_count == 0 else word_count
  word_count = int(word_count)
  
  # -- define prompt 
  prompt_template = """You are a brilliant writer who is confident and writes in an a logical, engaging tone. Your task is to take the provided orginal text and expand it to meet the objective guidelines. Use the context below to rewrite the original text with additional details within the target word count:
  Objective: {objective}
  Target Word Count: {word_count}
  Original Text: {original_text}
  Context: {context}
  Rewritten Text:"""

  # -- define inputs
  inputs = {'objective': objective, 'context': context, 'word_count': word_count, 'original_text': original_text}

  return prompt_template, inputs, None

def loadOutlinePlanPrompt(objective, original_text, context, word_count = 0): 
  word_count = len(original_text.split(' ')) * 1.5 if word_count == 0 else word_count
  word_count = int(word_count)

  # -- define prompt 
  prompt_template = """You write concise plans for writing individual report sections other writers. Provided is an original description of a section of a report. Using only the source material Context provided, generate a concise step-by-step plan someone can follow to competently write this particular section of the report. Do not exceed target word count:
  Objective: {objective}
  Target Word Count: {word_count}
  Original Text: {original_text}
  Context: {context}
  Rewritten Text:"""

  # -- define inputs
  inputs = {'objective': objective, 'context': context, 'word_count': word_count, 'original_text': original_text}

  return prompt_template, inputs, None

def loadWriterPrompt(objective, context, word_count = 500): 
  
  # -- define prompt 
  prompt_template = """You are a brilliant writer who is confident and writes in an a logical, engaging tone. Your task is to write a section of a report to meet the outline Objective. Using only the Context below, write the section's text with supporting details and keep it below the target word count. Additionally, do not include any information that is not relavent to the Objective of the section since the section is part of a larger report.:
  Objective: {objective}
  Target Word Count: {word_count}
  Context: {context}
  Section Text:"""

  # -- define inputs
  inputs = {'objective': objective, 'context': context, 'word_count': word_count}

  return prompt_template, inputs, None


def loadEditorPlanPrompt(inputs): 
  word_count = 0 if 'word_count' not in inputs.keys() else inputs['word_count']
  word_count = len(inputs['original_text'].split(' ')) * 1.5 if word_count == 0 else word_count
  word_count = int(word_count)

  # -- define prompt 
  prompt_template = """You are a world class report editor who writes concise plans and notes for editing text for other writers. Provided is the original text and an editing object you must follow. 
  
  Using only the Original Text and the source material Context provided, generate an incredibly specific step-by-step plan on how to precisely edit the Original Text to achieve the Editor Objective. Do not repeat yourself and do not exceed target word count:

  Editor Objective: {objective}
  Target Word Count: {word_count}
  Original Text: {original_text}
  Context: {context}
  Editor Plans:"""

  # -- define inputs
  inputs = {'objective': inputs['objective'], 'context': inputs['context'], 'word_count': word_count, 'original_text': inputs['original_text']}

  return prompt_template, inputs, None

def loadRewritePrompt(inputs): 
  word_count = 0 if 'word_count' not in inputs.keys() else inputs['word_count']
  word_count = len(inputs['original_text'].split(' ')) * 1.5 if word_count == 0 else word_count
  word_count = int(word_count)

  # -- define prompt 
  prompt_template = """You are a world class writer who is confident and writes in an a logical, engaging tone. Provided is notes from the editor and the accompanying original text. 
  
  Using only the Original Text and source material Context provided, rewrite the Original Text so that the Editor Notes are adequetly incorporated into the new text. Do not exceed target word count:

  Editor Notes: {objective}
  Target Word Count: {word_count}
  Original Text: {original_text}
  Context: {context}
  Rewritten Text:"""

  # -- define inputs
  inputs = {'objective': inputs['objective'], 'context': inputs['context'], 'word_count': word_count, 'original_text': inputs['original_text']}

  return prompt_template, inputs, None

def loadRewriteOutlinePrompt(inputs): 
  word_count = 0 if 'word_count' not in inputs.keys() else inputs['word_count']
  word_count = len(inputs['original_text'].split(' ')) * 1.5 if word_count == 0 else word_count
  word_count = int(word_count)

  # -- define prompt 
  prompt_template = """You excel at making precise edits to original plans. The plans you really excel at editing are plans to write a particular section of a report.
  
  Provided is the original plan for a particular section of a report denoted as Original Text.
    
  Using only the Original Text and source material Context provided, regenerate a new version of the Original Text which precisely incorporates the Editor Feedback while keeping the original structure of the Original Text intact. The resultant Rewritten Text shoud closely resemble the original plan if needed and be a concise step-by-step plan someone can follow to competently write this particular section of the report. Do not exceed target word count:
  
  Editor Notes: {objective}
  Target Word Count: {word_count}
  Original Text: {original_text}
  Context: {context}
  Rewritten Text:"""

  prompt_template = """You are a precise editor who is able to surgically edit original text to meet a specific objective. 

  Another editor has given you detailed Editor Plans on how to edit the Original Text description of a particular section of the report. 

  The Original Text is designed to be a concise step-by-step plan someone can follow to competently write this particular section of the report. 

  Using only the Original Text and source material Context provided, regenerate a new version of the Original Text step-by-step plan which precisely incorporates the Editor Feedback while keeping the original structure of the Original Text intact.
  
  To be clear, you are not to write the section, only to regenerate the concise step-by-step plan someone can follow to competently write this particular section of the report. Do not exceed target word count:

  Editor Notes: {objective}
  Target Word Count: {word_count}
  Original Text: {original_text}
  Context: {context}
  Rewritten Text:"""

  prompt_template = """You are the orginal author of the Original Text which was intended to be a concise step-by-step plan someone can follow to competently write this particular section of the report. 

  Now, an editor has given you a detailed notes on how to readjust the Original Text plan to make some improvements to the step-by-step plan. 

  Using only the Editor Notes, Original Text, and source material Context provided, precisely follow the Editor Notes to regenerate the original step-by-step plan while keeping the original structure of the Original Text intact. 

  The regenerated step-by-step plan should be a concise step-by-step plan someone can follow to competently write this particular section of the report.

  If the Editor Notes are minor, please do not feel the need to make significant edits to the Original Text.
  
  Editor Notes: {objective}
  Target Word Count: {word_count}
  Original Text: {original_text}
  Context: {context}
  Step-By-Step Plan:"""

  # -- define inputs
  inputs = {'objective': inputs['objective'], 'context': inputs['context'], 'word_count': word_count, 'original_text': inputs['original_text']}

  return prompt_template, inputs, None

def loadReportFlowPrompt(inputs): 
  obective = "remove redundancies across sections, remove unnecessary, ensure the report is logically structured, and ensure the report flows from one section to the next" if 'objective' not in inputs.keys() else inputs['objective']
  
  word_count = 0 if 'word_count' not in inputs.keys() else inputs['word_count']
  word_count = len(inputs['section_text'].split(' ')) * 1.5 if word_count == 0 else word_count
  word_count = int(word_count)

  # -- define prompt 
  prompt_template = """You are a world class report editor who writes concise plans and notes for editing text for other writers. Provided is the First Draft of a report and the text from a particular Section. 
  
  Using only the source material Context provided, generate a specific step-by-step plan on how to edit the particular Section Text to achieve the Editor Objective. Do not repeat yourself and do not exceed target word count:

  Editor Objective: {objective}
  Target Word Count: {word_count}
  First Draft: {report_text}
  Section Text: {section_text}
  Editor Plans:"""

  # -- define inputs
  inputs = {'objective': obective, 'report_text': inputs['report_text'], 'word_count': word_count, 'section_text': inputs['section_text']}

  return prompt_template, inputs, None

def loadOneShotWritePrompt(inputs): 
  word_count = 0 if 'word_count' not in inputs.keys() else inputs['word_count']
  word_count = len(inputs['section_text'].split(' ')) * 1.5 if word_count == 0 else word_count
  word_count = int(word_count)

  # -- define prompt 
  prompt_template = """You are a brilliant writer who is confident and writes in an a logical, engaging tone. 
  
  Using only the Report Outline and the source material Context below, write an expertly delivered report with supporting details and keep it below the target word count. Be sure to use section headers denoted in markdown format.:
  
  Report Topic: {topic}
  Target Word Count: {word_count}
  Report Outline: {outline}
  Context: {context}
  Report Text:"""

  # -- define inputs
  inputs = {'topic': inputs['topic'], 'outline': inputs['outline'], 'word_count': word_count, 'context': inputs['context']}

  return prompt_template, inputs, None

# ==================================================
# -- Editing Tools 
# ==================================================

def loadFindSectionPrompt(sections):
  # -- define prompt 
  schema = {
      "properties": {
         "section": {
              "type": "string",
              "enum": sections,
              "description": "section title referenced in a user query",
          },
      },
      "required": ["section"],
  }


  return schema

def loadDecideOutlinePrompt():
  # -- define prompt 
  schema = {
      "properties": {
         "outline_label": {
              "type": "string",
              "enum": ['outline', 'section_text'],
              "description": "describes whether a user wants to edit the outline description or section text",
          },
      },
      "required": ["outline_label"],
  }


  return schema
   


def loadCleanRequestPrompt(inputs):
  # -- used to reword a request for a edit and make more clear 
  
   # -- define prompt 
  prompt_template = """You are world class at giving directions to others by taking their requests and rewording them to be more clear. 
  
  Using only the provided User Request, generate a New Request that is more clear to the reciever.:

  User Request: {user_request}
  New Request:"""

  # -- define inputs
  inputs = {'user_request': inputs['user_request']}

  return prompt_template, inputs, None


def loadFindEvents(inputs):
  # -- used to reword a request for a edit and make more clear 
  
   # -- define prompt 
  prompt_template = """You are world class at giving directions to others by taking their requests and rewording them to be more clear. 
  
  Using only the provided User Request, generate a New Request that is more clear to the reciever.:

  User Request: {user_request}
  New Request:"""

  # -- define inputs
  inputs = {'user_request': inputs['user_request']}

  return prompt_template, inputs, None


