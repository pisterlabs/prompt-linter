import json
#from text.openai_text import OpenAITextGeneration
#from text.create_messages import Messages
#from text.context_window import ContextWindow
#from handlers.tool_handler import ToolHandler
#
#
#class NarroratorTool(ToolHandler):
#    def __init__(self, name="NarratorTool"):
#        super().__init__(name)
#        self.name = name
#        self.text = OpenAITextGeneration()
#        self.messages = Messages()
#        self.context = ContextWindow(window_size=30)
#                
#    def create_message(self, prompt: str) -> str:
#        return self.messages.create_message(role="system", content=prompt).model_dump()
#            
#    def command(self, prompt: str) -> str:
#        return self.text.send_chat_complete([self.create_message(prompt)]).choices[0].message.content
#    
#    
#class Narrator:
#    def __init__(self):
#        self.name = "Narrator"
#        self.tool = NarroratorTool()
#    
#    def send_narration(self, prompt: str) -> str:
#        primer = self.get_primer()
#        primer.append(f"{prompt}\n")
#        primer.append(self.get_context())        
#        return self.tool.command(json.load(prompt))
#        
#    def get_primer(self) -> str:
#        return self.tool.get_message_by_type("Primer", "Narrator")
#                   
            