from typing import Optional, List
import guidance
import requests
import os

class LocalMemory:
    context: Optional[str] = None
    template: str = """
Summarise the following conversation history, taking the existing context into account:

Context:
{{await 'context'}}

History:
{{await 'history'}}

Summary: {{gen 'summary'}}
"""
    
    def __init__(self, llm=None):
        print("Local Memory")
        self.context = {}
        self.messages = {}
        self.llm = llm
        self.prompt = guidance(self.template, llm=llm)
        
    def refresh_from(self, session_id: str) -> None:
        self.context.setdefault(session_id, "")
        self.messages.setdefault(session_id, [])
    
    def _summarise(self, session_id: str) -> str:
        contextualise = self.messages[session_id][:-10]
        if not contextualise:
            return
        self.messages[session_id] = self.messages[session_id][-10:]
        response = self.prompt(context=self.get_context(session_id), history="\n".join([message["content"] for message in contextualise]))
        return response['summary']
    
    def add_message(self, role: str, content: str, session_id: str) -> None:
        self.messages.setdefault(session_id, []).append({"role": role, "content": content})
        if len(self.messages[session_id]) > 20:
            self._summarise(session_id)
        self.context[session_id] = content
        
    def get_history(self, session_id) -> List[str]:
        return [message["content"] for message in self.messages.setdefault(session_id, [])]

    def get_formatted_history(self, session_id: str) -> str:
        history = self.get_history(session_id)
        return "\n".join(history)
    
    def get_context(self, session_id) -> str:
        return self.context.setdefault(session_id, "")
    
class MotorheadMemory:
    url: str = "http://motorhead:8001"
    timeout = 3000
    memory_key = "history"
    context: Optional[str] = None
    
    def __init__(self, llm=None):
        print("Motorhead memory")
        self.context = {}
        self.messages = {}
    
    def refresh_from(self, session_id: str):
        res = requests.get(
            f"{self.url}/sessions/{session_id}/memory",
            timeout=self.timeout,
            headers={"Content-Type": "application/json"},
        )
        res_data = res.json()
        self.context[session_id] = res_data.get("context", "NONE")
        messages = res_data.get("messages", [])
        messages.reverse()
        self.messages[session_id] = messages # Not strictly thread safe, but not too harmful

    def add_message(self, role: str, content: str, session_id: str):
        requests.post(
            f"{self.url}/sessions/{session_id}/memory",
            timeout=self.timeout,
            json={
                "messages": [
                    {"role": role, "content": f"{content}"},
                ]
            },
            headers={"Content-Type": "application/json"},
        )
        self.messages.setdefault(session_id, []).append({"role": role, "content": content})
        
    def get_history(self, session_id) -> List[str]:
        if session_id not in self.messages:
            self.refresh_from(session_id)
        return [message["content"] for message in self.messages[session_id]]
    
    def get_formatted_history(self, session_id) -> str:
        history = self.get_history(session_id)
        return "\n".join(history)
    
    def get_context(self, session_id) -> str:
        self.refresh_from(session_id)
        return self.context[session_id]
    
if os.environ.get('MEMORY') == 'local':
    Memory = LocalMemory
else:
    Memory = MotorheadMemory