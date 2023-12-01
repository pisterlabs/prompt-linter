

from langchain import PromptTemplate

class ModelPrompt():
    def __init__(self) -> None:
        pass

    def prepare_prompt(self, input_variables, template):
        return PromptTemplate(template = template, input_variables=input_variables)
    
    def get_audio_prompt(self):
        template = """You're a Professor who helps students on Youtube contents .
 
        {context}
 
        Answer with Accurate data   to the question and the way Professor speaks and only depends on provided Data. 
 
        Question: {question}
        Answer:"""

        prompt = self.prepare_prompt(template=template, input_variables=["context", "question"])
        #TODO - check if we need prompt.format?
        return prompt


    def get_text_description_prompt(self):
        template = """you are a Video content describer and create a details description of a video depends on    
        captions of all images   . All captions are in sequence of the video 
        {image_captions}
        Create a Title :
        Create  a Full Descrption :
        """
        prompt = self.prepare_prompt(input_variables=["image_captions"], template=template)
        return prompt


    def get_video_prompt(self):
        template = """You're a Helper of blind person who needs help to understand a visual of a video .He can understand audio part of the video  .
 
        {context}
 
        Answer with Accurate data   to the question and the way Helper of  poor vision person speaks and only depends on provided Data. 
 
        Question: {question}
        Answer:"""
 
        prompt = self.prepare_prompt(template=template, input_variables=["context", "question"])
        return prompt



