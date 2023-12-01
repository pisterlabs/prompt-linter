from __future__ import annotations
from re import I
from typing import Dict, List, Optional, ForwardRef
from typing_extensions import Literal  # just to be safe
from types import SimpleNamespace
from altair import Field
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchText
from sentence_transformers import SentenceTransformer
from langchain.prompts.example_selector.base import BaseExampleSelector
import streamlit as st
from pydantic import BaseModel
import os

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import BasePromptTemplate, PromptTemplate

from dotenv import load_dotenv

import prompt_templates

load_dotenv()


class DialogueExample(BaseModel):
    type: Literal["cue", "response", "thought", "keyword"]
    text: str
    score: Optional[float] = None
    pair: Optional[DialogueExample] = None
    similar: List[DialogueExample] = []

    # we have to avoid the circular reference when printing, so overriding these methods
    def __repr__(self):
        pair_repr = repr(
            self.pair) if self.pair is None else "DialogueExample(...)"
        similar_repr = repr(self.similar) if not (
            self.similar and self in self.similar) else "DialogueExample(...)"
        return f"DialogueExample(type={repr(self.type)}, text={repr(self.text)}, pair={pair_repr}, similar={similar_repr})"

    def __str__(self):
        pair_str = str(self.pair) if self.pair is None else "..."
        return f"DialogueExample(type={self.type}, text={self.text}, pair={pair_str}, similar={self.similar})"

    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def parse_obj(obj):
        return DialogueExample(**obj)


DialogueExample.model_rebuild()


@st.cache_resource
def load_model():
    return SentenceTransformer("thenlper/gte-large")


class PersonaExampleSelector(BaseExampleSelector, BaseModel):
    """Select examples from persona data"""

    persona_id: str

    def add_example(self, example: Dict[str, str]) -> None:
        """Add an example to the list of examples."""
        pass

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Select which examples to use based on the inputs."""
        semantic_model = load_model()

        qdrant = QdrantClient(
            url=os.environ.get("QDRANT_URL") or st.secrets.qdrant_url,
            api_key=os.environ.get(
                "QDRANT_API_KEY") or st.secrets.qdrant_api_key
        )

        examples = []
        deferred = []

        cues = qdrant.search(
            collection_name="cues",
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="persona_id", match=MatchValue(value=self.persona_id)
                    )
                ]
            ),
            query_vector=semantic_model.encode(
                input_variables.get("human_input")
            ).tolist(),
            limit=5,
            with_payload={"exclude": ["precontext", "postcontext"]},
            score_threshold=0.75,
        )

        for cue in cues:

            if cue.payload["cue"] == cue.payload["response"]:
                deferred.append(cue)
            else:

                dialogue_example = DialogueExample(
                    type="cue", text=cue.payload["cue"], score=cue.score)
                dialogue_example.pair = DialogueExample(
                    type="response", text=cue.payload["response"])

                responses = qdrant.search(
                    collection_name="responses",
                    query_filter=Filter(
                        must=[
                            FieldCondition(
                                key="persona_id", match=MatchValue(value=self.persona_id)
                            )
                        ]
                    ),
                    query_vector=semantic_model.encode(
                        cue.payload["response"]).tolist(),
                    limit=2,
                    score_threshold=0.75,  # this definitely needs to be higher, just not sure how high yet
                )

                for response in responses:
                    # we only want responses that aren't included,  there will always be at least one exact match.
                    if response.payload["response"] != cue.payload["response"]:
                        dialogue_example.pair.similar.append(DialogueExample(
                            type="response", text=response.payload["response"], score=response.score))

                examples.append(dialogue_example)

        thoughts = qdrant.search(
            collection_name="thoughts",
            query_filter=Filter(
                must=[
                    FieldCondition(
                            key="persona_id", match=MatchValue(value=self.persona_id)
                    )
                ]
            ),
            query_vector=semantic_model.encode(
                input_variables.get("human_input")).tolist(),
            limit=4,
            with_payload=True,
            score_threshold=0.75,  # this definitely needs to be higher, just not sure how high yet
        )

        # now let's move internal dialogue, monologues, etc to the end
        if len(deferred) > 0:
            for solo in deferred:
                examples.append(DialogueExample(
                    type="thought", text=solo.payload["response"], score=solo.score))

        # followed by actual chunks from the thoughts collection
        if len(thoughts) > 0:
            for thought in thoughts:
                # this is a quick hacky way to handle this.. now it will be treated similarly to how a internal monologue or something a character says without a cue
                # I just have to make a custom few shot template to handle it
                thought.payload["cue"] = thought.payload["thought"]
                thought.payload["response"] = thought.payload["thought"]

                examples.append(DialogueExample(
                    type="thought", text=thought.payload["thought"], score=thought.score))

        keyword_extraction = LLMChain(
            llm=ChatOpenAI(
                openai_api_key=st.session_state.openai_api_key_p,
                model="gpt-3.5-turbo",
                temperature=0,
            ),
            prompt=PromptTemplate.from_template(
                prompt_templates.KEYWORD_EXTRACTION),
            verbose=True,
        )

        keyword = keyword_extraction.run(input_variables.get("human_input"))

        keyword_matches, _ = qdrant.scroll(
            collection_name="responses",
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                            key="persona_id", match=MatchValue(value=self.persona_id)
                    ),
                    FieldCondition(
                        key="response", match=MatchText(text=keyword))
                ]
            ),
            limit=5,
            with_payload=True
        )

        if len(keyword_matches) > 0:
            for match in keyword_matches:
                examples.append(DialogueExample(
                    type="keyword", text=match.payload["response"]))

        debug_info = create_debug_info(
            input_variables["human_input"], examples)
        st.session_state[f"debug_info_{self.persona_id}"] = debug_info

        return examples


def create_debug_info(human_input, examples):
    cues = [ex for ex in examples if ex.type == "cue"]
    thoughts = [ex for ex in examples if ex.type == "thought"]
    keyword_matches = [ex for ex in examples if ex.type == "keyword"]

    output = f"- human input: {human_input.strip()}\n"
    output += "  retrieval_results:\n"
    if len(cues) > 0:
        output += "  # The following cues are similar to the user's last message and show the character's response and other semantically similar responses.\n"

        for cue in cues:
            output += f"    - cue: \"{cue.text}\"\n"
            output += f"      score: \"{cue.score}\"\n"
            output += f"      response: \"{cue.pair.text}\"\n"

            if len(cue.pair.similar) > 0:
                output += "      similar_responses:\n"
                output += "        # The following responses, don't necessarily respond to the same cue, but are semantically similar to the response to the above cue.\n"
                for response in cue.pair.similar:
                    output += f"        - response: \"{response.text}\"\n"
                    output += f"          score: \"{response.score}\"\n"

    if len(thoughts) > 0:
        output += "\n    # The remaining example messages are independant thoughts semantically matching the user's last message.\n"
        for thought in thoughts:
            output += f"    - thought: \"{thought.text}\"\n"
            output += f"      score: \"{thought.score}\"\n"

    if len(keyword_matches) > 0:
        output += "\n    # The remaining example messages are responses from the full-text match filtering over the response collection\n"
        for response in keyword_matches:
            output += f"    - response: \"{response.text}\"\n"

    return output


if __name__ == "__main__":
    selector = PersonaExampleSelector(persona_id="6513a240c54d7ab4cc90e370")

    examples = selector.select_examples(
        input_variables={"human_input": "What's your favorite drink?"})

    print(create_debug_info("What's your favorite drink?", examples))
