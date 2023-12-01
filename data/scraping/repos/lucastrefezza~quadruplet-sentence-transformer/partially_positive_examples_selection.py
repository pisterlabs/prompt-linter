from typing import Iterable, List, Optional, final
import math
import random
import re
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import nlpaug.augmenter.word as naw
from dataset.positive_examples_selection import back_translation
import openai
from dataset.constants import NO_REPLACE_WORDS, N_PART_EXAMPLES, ADAPTIVE_CROP_AUGMENT, CHAT_GPT, FALCON, ALPACA, \
    ADAPTIVE_CROP


MAX_WORDS_TO_REPLACE: final = 6
REPLACE_BERT: final = "replace_bert"
REPLACE_WORDNET: final = "replace_wordnet"
REPLACE_GLOVE: final = "replace_glove"
BACKTRANSL: final = "backtranslation"
MAX_INSERT_WORDS: final = 1
MIN_RESPONSE_NUM: final = 5


def mock_llm_response(caption: str, n_responses: int = MIN_RESPONSE_NUM) -> str:
    return "1. Woman wearing a hat;  2. Woman taking a photo;  3. Woman riding " \
           "a bike;  4. Parking lot surrounded by trees;  5. Woman standing in " \
           "the parking lot."


def parse_llm_response(llm_response: str, min_response_num: int = MIN_RESPONSE_NUM) -> List[str]:
    # Split the response on the numbers discarding the first, empty one
    responses = re.split(string=llm_response, pattern=r"[0-9]\.")[1:]

    # Check if the number of created sub-phrases is correct
    assert len(responses) >= min_response_num

    # Format the sub-phrases correctly, removing semi-column, spaces, ...
    for i, response in enumerate(responses):
        responses[i] = response.strip().lower().replace(";", "").replace(".", "")

    return responses


def crop_text_based_on_tagging(text: str,
                               crop_prefix: bool = False,
                               max_words_to_cut: Optional[int] = None,
                               augs: Iterable[str] = frozenset([REPLACE_WORDNET, BACKTRANSL]),
                               repeat: int = 1,
                               verbose: bool = False) -> List[str]:
    # Cut at most 3/4 of the words
    n_words = len(text.split(" "))
    if max_words_to_cut is None:
        max_words_to_cut = int(4 / 5 * n_words)
    else:
        max_words_to_cut = min(max_words_to_cut, int(4 / 5 * n_words))

    new_texts = []
    for _ in range(0, repeat):
        # Choose the words to cut
        n_words_to_cut = random.randint(int(3 / 4 * max_words_to_cut), max_words_to_cut)

        # Get part-of-speech tags
        tags = pos_tag(word_tokenize(text), tagset='universal')
        new_text = text

        if not crop_prefix:
            # Find the last word to keep, it must be either 'VERB' or 'NOUN'
            count_cut_words = 0
            last_word = None
            last_word_idx = None
            for i, el in enumerate(reversed(tags)):
                word, tag = el
                if tag == 'NOUN' or tag == 'VERB':
                    last_word = word
                    last_word_idx = len(tags) - 1 - i  # reversed index

                # Increase cut word count if the current word is not punctuation
                if tag != ".":
                    count_cut_words += 1

                # Stop searching for the last word if we reached the limit
                if count_cut_words >= n_words_to_cut:
                    break

            if verbose:
                print(f"Found last word {last_word} on position {last_word_idx}.")

            # Cut all the words after the last one
            new_text = " ".join([word for word, tag in tags[:last_word_idx + 1]])

        else:
            # Find the first word to keep, it must be either 'VERB' or 'NOUN' or 'DET'
            count_cut_words = 0
            first_word = None
            first_word_idx = None
            for i, el in enumerate(tags):
                word, tag = el
                if tag == 'NOUN' or tag == 'VERB' or 'DET':
                    first_word = word
                    first_word_idx = i  # first word idx

                # Increase cut word count if the current word is not punctuation
                if tag != ".":
                    count_cut_words += 1

                # Stop searching for the last word if we reached the limit
                if count_cut_words >= n_words_to_cut:
                    break

            if verbose:
                print(f"Found first word {first_word} on position {first_word_idx}.")

            # Cut all the words after the last one
            new_text = " ".join([word for word, tag in tags[first_word_idx:]])

        # Remove spaces before the punctuation
        new_text = re.sub(r'\s([?.!",](?:\s|$))', r'\1', new_text)

        # Apply augmentation
        if BACKTRANSL in augs:
            back_translation_aug = naw.BackTranslationAug(
                from_model_name='Helsinki-NLP/opus-mt-en-fr',
                to_model_name='Helsinki-NLP/opus-mt-fr-en'
            )
            new_text = back_translation_aug.augment(new_text)
        if REPLACE_BERT in augs:
            aug = naw.ContextualWordEmbsAug(
                model_path='roberta-base',
                action="substitute",
                aug_min=1,
                aug_max=2
            )
            new_text = aug.augment(new_text)
        if REPLACE_WORDNET in augs:
            aug = naw.SynonymAug(
                aug_src='wordnet',
                aug_min=1,
                aug_max=MAX_WORDS_TO_REPLACE,
                stopwords=NO_REPLACE_WORDS,
                verbose=True
            )
            new_text = aug.augment(new_text)

        # Add the created text to list
        new_text = new_text[0] if isinstance(new_text, list) else new_text
        new_texts.append(new_text)

    return new_texts


def adaptive_crop_part_pos_examples(caption: str,
                                    n_part_pos_examples: int,
                                    augment_backtranslation: bool = False,
                                    augment_insert: bool = False) -> List[str]:
    part_pos_examples_suffix = crop_text_based_on_tagging(
        text=caption,
        crop_prefix=True,
        max_words_to_cut=None,
        augs=frozenset([REPLACE_WORDNET]),
        repeat=math.ceil(n_part_pos_examples / 2)
    )

    part_pos_examples_prefix = crop_text_based_on_tagging(
        text=caption,
        crop_prefix=True,
        max_words_to_cut=None,
        augs=frozenset([REPLACE_WORDNET]),
        repeat=math.floor(n_part_pos_examples / 2)
    )
    part_pos_examples = part_pos_examples_suffix + part_pos_examples_prefix

    if augment_backtranslation:
        part_pos_examples = back_translation(part_pos_examples)
    if augment_insert:
        aug = naw.ContextualWordEmbsAug(
            model_path='roberta-base',
            action="insert",
            aug_min=0,
            aug_max=MAX_INSERT_WORDS
        )
        part_pos_examples = aug.augment(part_pos_examples)

    return part_pos_examples


def get_falcon_response(caption: str,
                        n_part_pos_examples: int = N_PART_EXAMPLES) -> str:
    raise NotImplementedError("Not implemented yet!")


def get_alpaca_response(caption: str,
                        n_part_pos_examples: int = N_PART_EXAMPLES) -> str:
    raise NotImplementedError("Not implemented yet!")


def get_chatgpt_response(caption: str,
                         n_part_pos_examples: int = N_PART_EXAMPLES) -> str:
    prompt = f"Given the sentence '{caption}' describing a scene, " \
             "identity the main objects/elements and provide 5 very " \
             "short numbered sentences that contain just some " \
             "elements, objects or subjects from sentence and not " \
             "all of them. Do not add any new element, object " \
             "or subject, only use the nouns identified in the given sentence. " \
             "Format the output giving the identified objects and " \
             "the numbered sentences."
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    response = completion.choices[0].message
    response = response["content"]
    return response


# Implement real LLM interrogation here
def get_part_pos_examples(caption: str,
                          n_part_pos_examples: int = N_PART_EXAMPLES,
                          algorithm_type: str = ADAPTIVE_CROP_AUGMENT) -> List[str]:
    if algorithm_type == CHAT_GPT:
        return parse_llm_response(get_chatgpt_response(caption,
                                                       n_part_pos_examples))
    elif algorithm_type == FALCON:
        return parse_llm_response(get_falcon_response(caption, n_part_pos_examples))

    elif algorithm_type == ALPACA:
        return parse_llm_response(get_alpaca_response(caption, n_part_pos_examples))

    elif algorithm_type == ADAPTIVE_CROP:
        return adaptive_crop_part_pos_examples(caption, n_part_pos_examples)

    elif algorithm_type == ADAPTIVE_CROP_AUGMENT:
        return adaptive_crop_part_pos_examples(caption,
                                               n_part_pos_examples,
                                               augment_insert=False,
                                               augment_backtranslation=True)

    else:
        return parse_llm_response(mock_llm_response(caption, n_part_pos_examples))
