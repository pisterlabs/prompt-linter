import guidance

class PromptTemplate:
    def __init__(self, default_character: str, default_prompt: str):
        self.default_character = default_character
        self.default_prompt = default_prompt
        self._characters = {}
        self._prompt_template_str = {}
        self._prompt_templates = {}
        
    def get(self, session_id: str, llm) -> guidance.Program:
        if session_id in self._prompt_templates:
            return self._prompt_templates[session_id]
        character = self._characters.setdefault(session_id, self.default_character)
        template = self._prompt_template_str.setdefault(session_id, self.default_prompt)
        self._prompt_templates[session_id] = guidance(template, llm=llm, character=character)
        return self._prompt_templates[session_id]
    
    def set(self, session_id: str, prompt: str):
        self._prompt_template_str[session_id] = prompt
        self._prompt_templates.pop(session_id, None)
        
    def set_character(self, session_id: str, character: str):
        self._characters[session_id] = character
        self._prompt_templates.pop(session_id, None)