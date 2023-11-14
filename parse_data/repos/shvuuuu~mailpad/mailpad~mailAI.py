from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

class mailpadAI:
    def __init__(self):
        self.openai_llm = None
        self.default_openai_config = {"temperature": 0.9, "model": "text-davinci-003"}

    def openai(self, temperature=None, model=None):
        if temperature is None:
            temperature = self.default_openai_config["temperature"]
        if model is None:
            model = self.default_openai_config["model"]

        self.openai_llm = OpenAI(temperature=temperature, model=model)

    def get_llm_response(self, form_input, email_sender, email_recipient, email_style):
        llm = self.openai_llm
        if llm is None:
            raise ValueError("LLM model not initialized.")

        template = """
        Write an email with {style} style and includes topic: {email_topic}.
        \nSender: {sender}
        Recipient: {recipient}
        \nEmail Text:
        """

        prompt = PromptTemplate(
            input_variables=["style", "email_topic", "sender", "recipient"],
            template=template,
        )

        response = llm(prompt.format(email_topic=form_input, sender=email_sender, recipient=email_recipient, style=email_style))
        return response
    
__all__ = ['mailpadAI']