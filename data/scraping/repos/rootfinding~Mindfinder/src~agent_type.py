"""
## `DialogueAgent` class
The `DialogueAgent` class is a simple wrapper around the `ChatOpenAI` model that stores the message history from the `dialogue_agent`'s point of view by simply concatenating the messages as strings.

It exposes two methods:
- `send()`: applies the chatmodel to the message history and returns the message string
- `receive(name, message)`: adds the `message` spoken by `name` to message history

"""
import regex
from typing import List, Dict, Tuple
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage,
)
from src.pinecone_helpers import query

class DialogueAgent:
    def __init__(
            self,
            name: str,
            system_message: SystemMessage,
            model: ChatOpenAI,
    ) -> None:
        self.name = name
        self.system_message = system_message
        self.model = model
        self.prefix = f"{self.name}: "
        self.reset()

    def reset(self):
        self.message_history = ["Here is the conversation so far."]

    def send(self) -> str:
        """
        Applies the chatmodel to the message history
        and returns the message string
        """
        message = self.model(
            [
                self.system_message,
                HumanMessage(content="\n".join(self.message_history + [self.prefix])),
            ]
        )
        return message.content

    def receive(self, name: str, message: str) -> None:
        """
        Concatenates {message} spoken by {name} into message history
        """
        self.message_history.append(f"{name}: {message}")


"""
## `PineconeDialogueAgent` class
The `PineconeDialogueAgent` class is a simple wrapper around the `ChatOpenAI` model that stores the message history from the `dialogue_agent`'s
    point of view by simply concatenating the messages as strings.
    
It also querys the Pincone API to get old questions that look like the current question and adds them to the message history.

It exposes two methods:
- `send()`: applies the chatmodel to the message history and returns the message string
- `receive(name, message)`: adds the `message` spoken by `name` to message history

"""
class PineconeDialogueAgent:
    def __init__(
            self,
            name: str,
            old_questions_namespace: str,
            old_questions_filter: Dict[str, Dict[str, str]],
            system_message: SystemMessage,
            model: ChatOpenAI,
    ) -> None:
        self.name = name
        self.old_questions_namespace = old_questions_namespace
        self.old_questions_filter = old_questions_filter
        self.system_message = system_message
        self.model = model
        self.prefix = f"{self.name}: "
        self.already_asked = []
        self.reset()

    def reset(self):
        self.message_history = ["Here is the conversation so far."]

    def get_old_questions(self, question: str) -> List[Tuple[str, str]]:
        """
        Querys the Pincone API to get old questions that look like the current question
        """

        results = query(question, self.old_questions_filter, self.old_questions_namespace, 3)

        if len(results['matches']) == 0:
            return []

        old_questions = []

        for result in results['matches']:
            old_questions.append((result["metadata"]["question"], result["metadata"]["answer"]))

        return old_questions


    def send(self) -> str:
        """
        Applies the chatmodel to the message history
        and returns the message string
        """

        # iterate message history backwards until we find the last question from the other agent
        last_question_from_sender = 0

        for j in range(len(self.message_history) - 1, -1, -1):
            if regex.match(r"^[^:]+: .*$", self.message_history[j]) is not None and not self.message_history[j].startswith(self.name):
                if self.message_history[j] in self.already_asked:
                    continue
                last_question_from_sender = j
                self.already_asked.append(self.message_history[j])
                break

        # if we found a question, add the old questions to the message history
        if last_question_from_sender > 0:
            pre_prompt_h = "In the past I have been asked the following questions: "
            pre_prompt = "\n".join([pre_prompt_h] + [f" -- Question {q} - Answer {a} " for q, a in self.get_old_questions(self.message_history[last_question_from_sender])])
            self.message_history.extend([pre_prompt])

        message = self.model([
            self.system_message,
            HumanMessage(content="\n".join(self.message_history + [self.prefix]))
        ])
        return message.content

    def receive(self, name: str, message: str) -> None:
        """
        Concatenates {message} spoken by {name} into message history
        """
        self.message_history.append(f"{name}: {message}")