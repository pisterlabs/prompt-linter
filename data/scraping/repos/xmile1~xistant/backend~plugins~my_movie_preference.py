from langchain.tools import BaseTool

class MyMoviePreferencePlugin:
    def __init__(self, model):
        self.model = model
    
    def get_lang_chain_tool(self):
      return [MyPreferenceTool()]
    

class MyPreferenceTool(BaseTool):
  """Tool to get my movie preference"""

  name = "My movie preference"
  description = (
    "This tool is useful for determining the user's movie preferences. It provides insights into the types of movies the user likes"
  )
  def _run(self, query: str) -> str: 
      return "I enjoy a variety of genres, with a particular interest in crime, mystery, and suspense. I seem to enjoy watching limited series, documentaries, and movies that explore true crime stories, investigations, and scandals. I also seem to appreciate character-driven stories that explore human relationships and emotions. My watch history suggests that I have a diverse taste in movies, ranging from thrilling action to dark dramas and even comedies. Overall, My movie preferences indicate that i enjoy thought-provoking and engaging films that keep me on the edge of my seat."
   
  async def _arun(self, query: str) -> str:
      """Use the GoogleAssit tool asynchronously."""
      raise NotImplementedError("Google assist does not support async")
