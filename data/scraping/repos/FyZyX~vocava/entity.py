import datetime
import time
import typing

from vocava import llm, storage
from vocava.llm import anthropic, mock

Language: typing.TypeAlias = str
LANGUAGES: dict[Language, dict[str, str]] = {
    "🇺🇸 English": {"name": "English", "flag": "🇺🇸", "code": "en"},
    "🇩🇪 German": {"name": "German", "flag": "🇩🇪", "code": "de"},
    "🇵🇱 Polish": {"name": "Polish", "flag": "🇵🇱", "code": "pl"},
    "🇪🇸 Spanish": {"name": "Spanish", "flag": "🇪🇸", "code": "es"},
    "🇮🇹 Italian": {"name": "Italian", "flag": "🇮🇹", "code": "it"},
    "🇫🇷 French": {"name": "French", "flag": "🇫🇷", "code": "fr"},
    "🇵🇹 Portuguese": {"name": "Portuguese", "flag": "🇵🇹", "code": "pt"},
    "🇮🇳 Hindi": {"name": "Hindi", "flag": "🇮🇳", "code": "hi"},
    "🇸🇦 Arabic": {"name": "Arabic", "flag": "🇸🇦", "code": "ar"},
    "🇨🇳 Chinese": {"name": "Chinese", "flag": "🇨🇳", "code": "zh"},
    "🇬🇷 Greek": {"name": "Greek", "flag": "🇬🇷", "code": "el"},
    "🇮🇱 Hebrew": {"name": "Hebrew", "flag": "🇮🇱", "code": "he"},
    "🇯🇵 Japanese": {"name": "Japanese", "flag": "🇯🇵", "code": "ja"},
    "🇰🇷 Korean": {"name": "Korean", "flag": "🇰🇷", "code": "ko"},
    "🇷🇺 Russian": {"name": "Russian", "flag": "🇷🇺", "code": "ru"},
    "🇸🇪 Swedish": {"name": "Swedish", "flag": "🇸🇪", "code": "sv"},
    "🇵🇭 Tagalog": {"name": "Tagalog", "flag": "🇵🇭", "code": "tl"},
    "🇻🇳 Vietnamese": {"name": "Vietnamese", "flag": "🇻🇳", "code": "vi"},
}
VOCALIZED_LANGUAGES = {
    "🇫🇷 French",
    "🇩🇪 German",
    "🇮🇳 Hindi",
    "🇮🇹 Italian",
    "🇵🇱 Polish",
    "🇵🇹 Portuguese",
    "🇪🇸 Spanish",
}


class User:
    def __init__(self, native_language: Language, target_language: Language,
                 fluency: int, db: storage.VectorStore):
        self._native_language = native_language
        self._target_language = target_language
        self._fluency = fluency
        self._db = db
        self._languages: dict[Language, dict[str, str]] = LANGUAGES

    def _get_language_name(self, language: Language):
        return self._languages[language]["name"]

    def native_language_name(self) -> str:
        return self._get_language_name(self._native_language)

    def target_language_name(self) -> str:
        return self._get_language_name(self._target_language)

    def _get_language_code(self, language: Language):
        return self._languages[language]["code"]

    def target_language_code(self) -> str:
        return self._get_language_code(self._target_language)

    def add_translation(self, phrase, translation):
        self._db.save(storage.Document(
            content=phrase,
            metadata=dict(
                language=self.target_language_name(),
                native_language=self.native_language_name(),
                fluency=self._fluency,
                translation=translation,
                timestamp=time.time(),
                category="phrase",
            )
        ))

    def add_vocabulary_word(self, word: str, translations: str):
        self._db.save(storage.Document(
            content=word,
            metadata=dict(
                language=self.target_language_name(),
                native_language=self.native_language_name(),
                fluency=self._fluency,
                translations=translations,
                timestamp=time.time(),
                category="vocabulary",
            )
        ))

    def add_grammar_mistake(self, phrase, correct, translation, explanation):
        self._db.save(storage.Document(
            content=phrase,
            metadata=dict(
                language=self.target_language_name(),
                native_language=self.native_language_name(),
                correct=correct,
                translation=translation,
                explanation=explanation,
                timestamp=time.time(),
                category="grammar-mistake",
            )
        ))

    def known_phrases(self):
        results = self._db.query_by_metadata(
            language=self.target_language_name(),
            native_language=self.native_language_name(),
            category="phrase",
        )
        docs = results["documents"]
        metadatas = results["metadatas"]
        phrases = []
        for doc, metadata in zip(docs, metadatas):
            item = {
                self.target_language_name(): doc,
                self.native_language_name(): metadata["translation"],
                "timestamp": datetime.datetime.fromtimestamp(metadata["timestamp"]),
            }
            phrases.append(item)
        return phrases

    def known_vocabulary(self):
        results = self._db.query_by_metadata(
            language=self.target_language_name(),
            native_language=self.native_language_name(),
            category="vocabulary",
        )
        docs = results["documents"]
        metadatas = results["metadatas"]
        vocabulary = []
        for doc, metadata in zip(docs, metadatas):
            item = {
                self.target_language_name(): doc,
                self.native_language_name(): metadata["translations"],
                "timestamp": datetime.datetime.fromtimestamp(metadata["timestamp"]),
            }
            vocabulary.append(item)
        return vocabulary

    def known_mistakes(self):
        results = self._db.query_by_metadata(
            language=self.target_language_name(),
            native_language=self.native_language_name(),
            category="grammar-mistake",
        )
        docs = results["documents"]
        metadatas = results["metadatas"]
        mistakes = []
        for doc, metadata in zip(docs, metadatas):
            item = {
                "mistake": doc,
                "correct": metadata["correct"],
                "explanation": metadata["explanation"],
                "translation": metadata["translation"],
                "timestamp": datetime.datetime.fromtimestamp(metadata["timestamp"]),
            }
            mistakes.append(item)
        return mistakes

    def fluency(self):
        return self._fluency


class Tutor:
    def __init__(self, model: llm.LanguageModel):
        self._model = model

    def ask(self, prompt: str, max_tokens: int = 250):
        return self._model.generate(prompt, max_tokens=max_tokens)


def get_tutor(model, key=None) -> Tutor:
    if model == "Claude":
        model = anthropic.Claude(api_key=key)
    else:
        model = mock.MockLanguageModel()
    tutor = Tutor(model)
    return tutor
