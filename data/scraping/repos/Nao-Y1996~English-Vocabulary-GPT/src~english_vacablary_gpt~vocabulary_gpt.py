import os

import openai
from openai import OpenAI
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion

from .vocabulary_parser import Vocabulary

try:
    os.environ["OPENAI_API_KEY"]
except KeyError:
    raise EnvironmentError("OPENAI_API_KEY not found.")

openai.api_key = os.environ["OPENAI_API_KEY"]

SYSTEM_CONTENT = """
You are an specialist of English vocabulary and it's meaning. You must teach correct English gamer and vocabulary to who is Japanese and learning English. You are assistant designed to output JSON.
When You receive an English word, you need to create json that explains the word flowing the example format below. 

JSON has 3 keys such as "word", "explanations" and "related_term".

The value of "word" is the word that is you have to explain.

The value of  "explanations" is array and has 5 keys such as "partOfSpeech", "meaningsJa", "sentences", "synonyms" and "antonymses".
The value of "partOfSpeech" must be in Japanese and it's type is string. And it also must not contain multiple part of speech.
If the "word" has multiple part of speech, response JSON must have multiple "partOfSpeech" object.
The value of "meaningsJa" is array and it`s element is object that has 2 keys such as "meaning" and "note". 
The values of "sentences", "synonyms" and "antonyms" is array and it`s element is object that has 2 keys such as "en" and "ja". 
 
The values of "related_term" is array and it`s element is object that has 2 keys such as "en" and "ja". 
 
Following is example JSON about the word "major".
{
  "word": "major",
  "explanations": [
    {
      "partOfSpeech": "形容詞",
      "meaningsJa": [
        {"meaning":"主要な", "note": ""},
        {"meaning":"大きな", "note": ""},
        {"meaning":"重要な", "note": ""}],
      "sentences": [
        {"en": "He played a major role.", "ja": "彼は主要な役割を果たした" },
        { "en": "He is a major shareholder.", "ja": "彼は大株主だ" },
        { "en": "He has a major responsibility.","ja": "彼は大きな責任を負っている。" }
      ],
      "synonyms": [],
      "antonyms": [{  "en": "minor", "ja": "マイナーな"}]
    },
    {
      "partOfSpeech": "動詞",
      "meaningsJa": [{"meaning":"~を専攻する", "note": "major in ~ の形で使う"}],
      "sentences": [
        {"en": "I major in computer science.",  "ja": "私はコンピュータサイエンスを専攻している。" }
      ],
      "synonyms": [{"en": "specialize in", "ja": "専攻する"}],
      "antonyms": []
    }
  ],
  "related_term": [{"en": "majority", "ja": "大多数"}]
}
"""


class GptUsage:
    def __init__(self, usage: dict[str, int]) -> None:
        """Usage of the API call

        :param usage: {"prompt_tokens": 12, "completion_tokens": 23, "total_tokens": 35}
        """
        self.prompt_tokens: int = usage["prompt_tokens"]
        self.completion_tokens: int = usage["completion_tokens"]
        self.total_tokens = usage["total_tokens"]

    def get_prompt_tokens(self):
        """Get the number of tokens used for the prompt"""
        return self.prompt_tokens

    def get_total_tokens(self):
        """Get the total number of tokens used"""
        return self.total_tokens

    def get_completion_tokens(self):
        """Get the number of tokens used for the completion"""
        return self.completion_tokens


class VocabularyGPTResponse:
    def __init__(self, response: ChatCompletion):
        self.vocabulary = Vocabulary(response.choices[0].message.content)
        self._usage: CompletionUsage = response.usage

    def get_vocabulary(self) -> Vocabulary:
        """Get vocabularyJSON string of the response"""
        return self.vocabulary

    def usage(self) -> str:
        """Show usage of the API call"""
        return f"<token usage>\n" \
               f"  prompt: {self.get_prompt_tokens()}\n" \
               f"  completion: {self.get_completion_tokens()}\n" \
               f"  total: {self.get_total_tokens()}"

    def get_prompt_tokens(self):
        """Get the number of tokens used for the prompt"""
        return self._usage.prompt_tokens

    def get_total_tokens(self):
        """Get the total number of tokens used"""
        return self._usage.total_tokens

    def get_completion_tokens(self):
        """Get the number of tokens used for the completion"""
        return self._usage.completion_tokens


class VocabularyGPT:

    @classmethod
    def create_vocabulary(cls, word) -> VocabularyGPTResponse:
        """
        Create vocabulary json from word.
        :param word:
        :return: json string
        """
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_CONTENT},
                {"role": "user", "content": word}
            ]
        )
        return VocabularyGPTResponse(response)
