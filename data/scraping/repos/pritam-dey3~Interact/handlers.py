import openai

from interact.base import Cascade, Handler, Message


class OpenAiLLM(Handler):
    """Handler for generating a response using OpenAI's Language Model."""

    role = "Assistant"

    def __init__(self, role: str = None, model: str = "gpt-3.5-turbo") -> None:
        if role is not None:
            self.role = role
        self.model = model

    async def process(self, msg: Message, csd: Cascade) -> Message:
        """Generate a response using the message passed to this handler. If OpenAI api
        key is not set in the environment, then the api key can be passed as a variable
        in the Cascade.vars dictionary.

        Args:
            msg (Message): user response sent to OpenAI chatGPT.
            csd (Cascade): Casccade that this handler is a part of.

        Returns:
            Message: response from OpenAI chatGPT.
        """
        api_key = csd.vars.get("api_key", None)
        res = await openai.ChatCompletion.acreate(
            model=self.model,
            api_key=api_key,
            messages=[
                {"role": "user", "content": str(msg)},
            ],
        )

        reply = ". ".join(c["message"]["content"] for c in res["choices"])
        return Message(primary=reply, sender=self.role, openai_response=dict(res))


class AssignRole(Handler):
    """Assign a role to the last message sent to this Handler by the current Cascade."""

    def __init__(self, role: str) -> None:
        self.role = role

    async def process(self, msg: Message, csd: Cascade) -> Message:
        return msg
