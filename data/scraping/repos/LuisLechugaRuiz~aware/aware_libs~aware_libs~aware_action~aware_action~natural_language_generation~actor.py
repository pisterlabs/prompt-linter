from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from aware_narratives.prompts.learning import (
    get_actor_prompt,
)

from typing import Dict, List, Union


class ConversationActor(object):
    """Actor of the conversation"""

    def __init__(self, temperature=0.0):
        # Load prompt
        actor_prompt = get_actor_prompt()
        # Create evaluation chain
        self.actuation_chain = LLMChain(
            llm=ChatOpenAI(temperature=temperature, max_tokens=512),
            prompt=actor_prompt,
        )

    def answer_to_user(
        self,
        name,
        self_description,
        user_information,
        user_name,
        expected_role,
        short_term_goals,
        hypotheses,
        example,
        emotions,
        thoughts,
        immediate_goals,
        recommendations,
        chat_history,
        user_msg,
    ) -> Union[str, List[str], Dict[str, str]]:
        answer = self.actuation_chain.predict(
            name=name,
            self_description=self_description,
            user_name=user_name,
            user_information=user_information,
            expected_role=expected_role,
            short_term_goals=short_term_goals,
            emotions=emotions,
            thoughts=thoughts,
            hypotheses=hypotheses,
            immediate_goals=immediate_goals,
            recommendations=recommendations,
            chat_history=chat_history,
            example=example,
            input=user_msg,
        )
        return answer


def main():
    conversation_actor = ConversationActor()
    # TODO: Fill me properly.
    # answer = conversation_actor.answer_to_user(
    # print(f"Actor:\n {answer}")


if __name__ == "__main__":
    main()
