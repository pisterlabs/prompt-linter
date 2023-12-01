import spacy
from spacy.matcher import Matcher
import openai
from files.trie import Trie, language_keywords  




nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)


# Define patterns
web_search_patterns = [
    [{"LEMMA": "look"}, {"LOWER": "up"}],
    [{"LEMMA": "find"}, {"LEMMA": "information"}, {"LOWER": "on"}],
    [{"LEMMA": "can"}, {"LEMMA": "you"}, {"LEMMA": "search"}, {"LOWER": "for"}],
    [{"LEMMA": "want"}, {"LEMMA": "to"}, {"LEMMA": "know"}, {"LOWER": "about"}],
    [{"LEMMA": "tell"}, {"LEMMA": "me"}, {"LOWER": "about"}],
    [{"LEMMA": "search"}, {"LEMMA": "the"}, {"LEMMA": "web"}, {"LOWER": "for"}],
    [{"LEMMA": "google"}],
    [{"LEMMA": "what"}, {"LEMMA": "do"}, {"LEMMA": "you"},
        {"LEMMA": "know"}, {"LOWER": "about"}],
    [{"LEMMA": "find"}, {"LEMMA": "me"}, {"LEMMA": "details"}, {"LOWER": "about"}],
    [{"LEMMA": "investigate"}]
]

code_execute_patterns = [
    [{"LEMMA": "execute"}, {"LEMMA": "this"}, {"LEMMA": "code"}],
    [{"LEMMA": "can"}, {"LEMMA": "you"}, {"LEMMA": "run"},
        {"LEMMA": "this"}, {"LEMMA": "script"}],
    [{"LEMMA": "please"}, {"LEMMA": "compile"}, {"LEMMA": "this"}],
    [{"LEMMA": "test"}, {"LEMMA": "this"}, {
        "LEMMA": "code"}, {"LEMMA": "for"}, {"LEMMA": "me"}],
    [{"LEMMA": "want"}, {"LEMMA": "to"}, {"LEMMA": "see"}, {"LEMMA": "the"}, {
        "LEMMA": "output"}, {"LEMMA": "of"}, {"LEMMA": "this"}, {"LEMMA": "code"}],
    [{"LEMMA": "run"}, {"LEMMA": "this"}, {"LEMMA": "function"}],
    [{"LEMMA": "execute"}, {"LEMMA": "the"}, {
        "LEMMA": "following"}, {"LEMMA": "lines"}],
    [{"LEMMA": "can"}, {"LEMMA": "you"}, {"LEMMA": "debug"},
        {"LEMMA": "this"}, {"LEMMA": "code"}],
    [{"LEMMA": "compile"}, {"LEMMA": "and"}, {"LEMMA": "run"}, {"LEMMA": "this"}],
    [{"LEMMA": "check"}, {"LEMMA": "if"}, {"LEMMA": "this"},
        {"LEMMA": "code"}, {"LEMMA": "works"}]
]
# Add patterns to matcher
for pattern in web_search_patterns:
    matcher.add("WEB_SEARCH", [pattern])

for pattern in code_execute_patterns:
    matcher.add("CODE_EXECUTE", [pattern])

def detect_intent_with_gpt(text):
    prompt = f"What is the intent of the following user input: '{text}'?"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=50
    )
    verbose_intent = response.choices[0].text.strip().lower()
    
    # Map verbose intent to simple label
    if "weather" in verbose_intent:
        return "weather_check"
    elif "code" in verbose_intent and "execute" in verbose_intent:
        return "code_execution"
    elif "search" in verbose_intent or "find" in verbose_intent:
        return "web_search"
    elif "question" in verbose_intent and "answer" in verbose_intent:
        return "qa"
    elif "music" in verbose_intent:
        return "play_music"
    elif "reminder" in verbose_intent or "alarm" in verbose_intent:
        return "set_reminder"
    elif "translate" in verbose_intent:
        return "translation"
    elif "directions" in verbose_intent or "navigate" in verbose_intent:
        return "get_directions"
    else:
        return "general_conversation"


def detect_intent(message):
    doc = nlp(message)
    matches = matcher(doc)
    for match_id, start, end in matches:
        string_id = nlp.vocab.strings[match_id]
        span = doc[start:end]
        if string_id == "WEB_SEARCH":
            return "web_search"
        elif string_id == "CODE_EXECUTE":
            return "code_execution"
    return "general_conversation"


language_tries = {}
for language, keywords in language_keywords.items():
    trie = Trie()
    for keyword in keywords:
        trie.insert(keyword)
    language_tries[language] = trie
    
    
def detect_language(code):
    print("Debug: Starting language detection")
    detected_language = None
    max_count = 0

    for language, trie in language_tries.items():
        count = trie.search(code.lower())
        print(f"Debug: Count for {language} is {count}")
        
        if count > max_count:
            max_count = count
            detected_language = language

    print(f"Debug: Detected language is {detected_language}")
    return detected_language




def main_intent_detection(text):
    intent = detect_intent_with_gpt(text)
    detected_language = None  # Initialize to None
    
    if intent not in ["code_execution", "web_search", "general_conversation"]:
        intent = detect_intent(text)
        
    if intent in ["code_execution", "web_search"]:
        detected_language = detect_language(text)
        
    return intent, detected_language
