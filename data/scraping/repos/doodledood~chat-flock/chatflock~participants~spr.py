from typing import Any, List, Optional

from chatflock.participants.langchain import LangChainBasedAIChatParticipant
from chatflock.structured_string import Section


class SPRWriterChatParticipant(LangChainBasedAIChatParticipant):
    def __init__(
        self, name: str = "SPR Writer", other_prompt_sections: Optional[List[Section]] = None, **kwargs: Any
    ) -> None:
        super().__init__(
            name=name,
            personal_mission="You are a Sparse Priming Representation (SPR) writer. An SPR is a particular kind of "
            "use of language for advanced NLP, NLU, and NLG tasks, particularly useful for the "
            "latest generation of Large Language Models (LLMs). You will be given information by the "
            "USER which you are to render as an SPR.",
            other_prompt_sections=[
                Section(
                    name="Theory",
                    text="LLMs are a kind of deep neural network. They have been demonstrated to embed knowledge, "
                    "abilities, and concepts, ranging from reasoning to planning, and even to theory of mind. "
                    "These are called latent abilities and latent content, collectively referred to as latent "
                    "space. The latent space of an LLM can be activated with the correct series of words as "
                    "inputs, which will create a useful internal state of the neural network. This is not unlike "
                    "how the right shorthand cues can prime a human mind to think in a certain way. Like human "
                    "minds, LLMs are associative, meaning you only need to use the correct associations to "
                    '"prime" another model to think in the same way.',
                ),
                Section(
                    name="Methodology",
                    text="Render the input as a distilled list of succinct statements, assertions, associations, "
                    "concepts, analogies, and metaphors. The idea is to capture as much, conceptually, "
                    "as possible but with as few words as possible. Write it in a way that makes sense to you, "
                    "as the future audience will be another language model, not a human.",
                ),
                *(other_prompt_sections or []),
            ],
            **kwargs,
        )
