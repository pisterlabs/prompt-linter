from langchain.schema import SystemMessage, HumanMessage
from Action import Action
import subprocess

class PlaySongFromYoutubeAction(Action):
    def __init__(self, llm):
        self.name="timer"
        self.selection_prompt="Playing music?"
        self.parameter_prompt="Task: Extract the search term to enter into a search engine to find the song. Search term:"
        self.llm = llm
        self.processID = None

    def execute(self, query):
        search_term = self.get_single_paramter(query)
        self.play_youtube(search_term)
        return "Searching and playing music with respect to "+search_term+" seconds."

    def get_single_paramter(self, query):
        prompt = query+" "+self.parameter_prompt
        messages = [
            SystemMessage(content = "You are a helpful assistant who helps to search for music."),
            HumanMessage(content = prompt)
        ]
        response = self.llm(messages)        
        return response.content

    def play_youtube(self, search_term):
        search_term = search_term.replace('"','')
        self.processID = subprocess.Popen(['ytfzf', "-m", "-a", search_term])
        pass