import openai
import os
from pydantic import BaseModel, ValidationError, field_validator
from dotenv import load_dotenv

class OpenAIAPIClient:
    def __init__(self):

        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        self.client = openai.OpenAI(api_key=api_key)

    def validate_citation(self, question: str, reasoning: str, citation: str, source_text: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a validator. Determine if the citation is valid for the given reasoning and source text. Explain your reasoning."},
                {"role": "user", "content": f"Question: {question}\nReasoning: {reasoning}\nCitation: {citation}\nSource Text: {source_text}"}
            ]
        )
        return response.choices[0].message.content.strip()

class AIResponse(BaseModel):
    question: str
    reasoning: str
    citation: str
    source_text: str
    validation_response: str = ""

    @field_validator('validation_response')
    @classmethod
    def validate_citation(cls, v, values):
        if 'invalid' in v.lower():
            raise ValidationError(f"Validation failed: {v}")
        return v

    def validate(self, client: OpenAIAPIClient):
        self.validation_response = client.validate_citation(
            self.question, self.reasoning, self.citation, self.source_text
        )

def process_ai_response(client: OpenAIAPIClient, question: str, reasoning: str, citation: str, source_text: str):
    try:
        response = AIResponse(question=question, reasoning=reasoning, citation=citation, source_text=source_text)
        response.validate(client)
        print("Validation successful:", response)
    except ValidationError as e:
        print(str(e))

# Example usage
client = OpenAIAPIClient()
question = "How is climate change impacting polar regions?"
reasoning = "Climate change is leading to rapid melting of polar ice caps"
citation = "Recent studies show a significant increase in polar ice melt over the past decade"
source_text = "The polar regions are experiencing the effects of global warming, with noticeable reductions in ice cover"
process_ai_response(client, question, reasoning, citation, source_text)
