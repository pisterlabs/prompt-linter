import g4f
from langchain.schema import HumanMessage, SystemMessage, AIMessage

class G4FModel:

    def convmessage(self,m):
        if isinstance(m, HumanMessage):
            return { "role": "user", "content": m.content }
        if isinstance(m, SystemMessage):
            return { "role": "system", "content": m.content }
        if isinstance(m, AIMessage):
            return { "role": "assistant", "content": m.content }
        return { "role": "user", "content": m.content }

    def __call__(self, msgs):
        response = g4f.ChatCompletion.create(
                model=g4f.models.default,
                messages=[self.convmessage(m) for m in msgs])
        return AIMessage(content=response)
