from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from aware_narratives.prompts.learning import (
    DEF_CONVERSATION_EXAMPLE,
    DEF_CONVERSATION_AGENT_EXAMPLE_MAP,
    DEF_PERSON_INFORMATION_EXAMPLE,
    DEF_PERSON_NAME_EXAMPLE,
    get_psychologist_prompt,
)
from typing import Dict, List, Union


class ConversationPsychologist(object):
    """Evaluates the personality of the user and gives recommendations"""

    def __init__(self, temperature=0.0):
        # Load prompt
        psychologist_prompt = get_psychologist_prompt()
        # Load example
        self.example = DEF_CONVERSATION_AGENT_EXAMPLE_MAP["Psychologist"]
        # Create evaluation chain
        self.personality_evaluation_chain = LLMChain(
            llm=ChatOpenAI(temperature=temperature, max_tokens=512),
            prompt=psychologist_prompt,
        )

    def evaluate_personality(
        self, conversation, user_information, person_name
    ) -> Union[str, List[str], Dict[str, str]]:
        max_retries = 2  # TODO: Get from config
        retries = 0
        while retries <= max_retries:
            try:
                evaluation = self.personality_evaluation_chain.predict_and_parse(
                    conversation=conversation,
                    example=self.example,
                    user_information=user_information,
                    person_name=person_name,
                )
                return evaluation
            except Exception as e:
                print(f"Error: {e}")
                retries += 1
        raise RuntimeError(
            f"Failed to get psychologist evaluation after {max_retries + 1} attempts"
        )


def main():
    conversation_psychologist = ConversationPsychologist()
    evaluation = conversation_psychologist.evaluate_personality(
        conversation=DEF_CONVERSATION_EXAMPLE,
        user_information=DEF_PERSON_INFORMATION_EXAMPLE,
        person_name=DEF_PERSON_NAME_EXAMPLE,
    )
    print(f"Evaluation:\n {evaluation}")


if __name__ == "__main__":
    main()
