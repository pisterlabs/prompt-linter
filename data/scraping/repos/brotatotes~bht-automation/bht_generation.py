# IMPORTS

import datetime
import math
import os
import threading
import time
import traceback
import re

import openai
import spacy
import tiktoken
from bibleref import BibleRange, BibleVerse
from gensim.utils import tokenize

from bht.bht import BHT
from bht.multi_threaded_work_queue import MultiThreadedWorkQueue
from bht.bht_semantics import BHTSemantics
from bht.bht_common import *

class BHTGenerator:
    def __init__(self):
        openai.api_key = open('openai-api-key.txt', 'r').read().strip()
        self.ENCODING = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.bht_semantics = BHTSemantics()

    def get_commentary_choicests(self, verse_ref, choicest_prompt, commentators):
        book, chapter, verse = get_book_chapter_verse(verse_ref)

        commentator_choicests = {}

        # Load commentator choicest pieces from files
        for commentator in commentators:
            commentator_choicest_file = f'{WORKING_DIRECTORY}/{OUTPUT_FOLDER}/{CHOICEST_FOLDER_NAME}/{choicest_prompt}/{book}/Chapter {chapter}/Verse {verse}/{commentator}.txt'

            if not os.path.exists(commentator_choicest_file):
                # raise Exception(f"No choicest file found for {commentator} for {choicest_prompt} for {verse_ref}, file path: {commentator_choicest_file}")
                continue
            
            file_contents = ""
            with open(commentator_choicest_file, 'r', encoding='utf-8') as file:
                file_contents = file.read()

            if not file_contents:
                # print(f'No choicest quotes found for {commentator}.')
                continue
            
            commentator_choicests[commentator] = file_contents

        return commentator_choicests


    def get_bht_output_path(self, choicest_prompt, bht_prompt, book, chapter, verse):
        return f'{WORKING_DIRECTORY}/{OUTPUT_FOLDER}/{BHT_FOLDER_NAME}/{choicest_prompt} X {bht_prompt}/{book}/Chapter {chapter}/{book} {chapter} {verse} bht.md'

    def get_choicest_output_path(self, choicest_prompt, book, chapter, verse, commentator):
        return f'{WORKING_DIRECTORY}/{OUTPUT_FOLDER}/{CHOICEST_FOLDER_NAME}/{choicest_prompt}/{book}/Chapter {chapter}/Verse {verse}/{commentator}.txt'

    def get_commentator_tier(self, commentator):
        if commentator in ("Henry Alford", "Jamieson-Fausset-Brown", "Marvin Vincent", "Archibald T. Robertson"):
            return 1
        elif commentator in ("Albert Barnes", "Philip Schaff"):
            return 2
        elif commentator in ("John Wesley", "John Gill", "John Calvin"):
            return 3
        else:
            raise Exception(f"No tier defined for commentator: {commentator}")


    # Generate Choicest Piece

    def ask_gpt_choicest(self, commentator, commentary, verse_ref, choicest_prompt, extra_messages):
        prompt_text = get_prompt(CHOICEST_FOLDER_NAME, choicest_prompt)
        messages = []

        messages.append({
            "role": "system",
            "content": prompt_text
        })

        messages.append({
            "role": "user",
            "content": f'[Commentary]\n{commentary}\n'
        })

        messages.extend(extra_messages)

        model = "gpt-3.5-turbo"
        token_count = sum(len(self.ENCODING.encode(message["content"])) for message in messages)
        if token_count > 4097:
            print(f"ℹ️  {verse_ref} {commentator} Too many tokens. Using 16k Context instead.")
            model += "-16k"

        try:
            chat_completion = openai.ChatCompletion.create(
                model=model, 
                messages=messages,
                request_timeout=15,
            )
        except openai.error.InvalidRequestError:
            print(f"ℹ️  {verse_ref} {commentator} Something went wrong. Trying 16k Context.")
            chat_completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k", 
                messages=messages,
                request_timeout=15
            )

        return chat_completion.choices[0].message["content"]


    def ask_gpt_choicest_with_retry(self, commentator, commentary, verse_ref, choicest_prompt, extra_messages, tries=0, try_limit=10):
        if tries >= try_limit:
            raise Exception(f"❌ Failed {try_limit} times to get choicest. Quitting. ❌")
        
        try:
            return self.ask_gpt_choicest(commentator, commentary, verse_ref, choicest_prompt, extra_messages)
        except TimeoutError:
            print(f"Attempt {tries} timed out. Trying again.")
            return self.ask_gpt_choicest_with_retry(commentator, commentary, verse_ref, choicest_prompt, extra_messages, tries + 1, try_limit)


    def record_gpt_choicest(self, verse_ref, choicest_prompts, commentators, force_redo=False):
        for commentator in commentators:        
                for choicest_prompt in choicest_prompts:

                    # print(f"🟧 {verse_ref} {commentator} {choicest_prompt}")
                    
                    book, chapter, verse = get_book_chapter_verse(verse_ref)

                    out_path = self.get_choicest_output_path(choicest_prompt, book, chapter, verse, commentator)

                    choicest_not_empty = (os.path.exists(out_path) and not not open(out_path, 'r', encoding='utf-8').read().strip())
                    commentary = get_commentary(commentator, verse_ref)
                    no_commentary = not commentary

                    if no_commentary or (not force_redo and choicest_not_empty):
                        msg = f"✅ {verse_ref} {commentator} {choicest_prompt}"
                        if no_commentary:
                            msg += f" No Commentary found. "
                        if choicest_not_empty:
                            msg += f" Choicest already exists. "

                        # print(msg)
                        continue
                    
                    commentary_tokens_set = set(tokenize(commentary.lower()))

                    commentary_length_limit = 25

                    extra_messages = []

                    if len(commentary_tokens_set) < commentary_length_limit:
                        choicest = f"1. {commentary}"
                    else:
                        max_tries = 5
                        tries = 0
                        while tries < max_tries:
                            tries += 1
                            
                            choicest = self.ask_gpt_choicest_with_retry(commentator, commentary, verse_ref, choicest_prompt, extra_messages)
                            choicest = choicest.replace('\n\n', '\n')
                            choicest_tokens = list(tokenize(choicest.lower()))
                            choicest_tokens_set = set(choicest_tokens)
                            word_count = len(choicest_tokens)

                            token_diff_limit = 2
                            word_count_limit = 300

                            diffs = len(choicest_tokens_set - commentary_tokens_set)
                            too_many_diffs = diffs > token_diff_limit
                            too_long = word_count > word_count_limit

                            if tries > max_tries:
                                raise Exception(f"❌ {verse} {commentator} Failed {max_tries} times to get choicest. Quitting. ❌")

                            if not too_many_diffs and not too_long:
                                break
                            else:
                                extra_messages.append({
                                    "role": "assistant",
                                    "content": choicest
                                })

                                complaints = []

                                info_msg = [f"🔄 {verse_ref} {commentator}"]
                                info_msg.append(f"({diffs} injected words, {word_count} words)")

                                if too_many_diffs:
                                    info_msg.append(f"MORE THAN {token_diff_limit} INJECTED WORDS!")
                                    complaints.append(f"Please try again using only words from the original commentary. Do not add any of your own words. Do not include any other comments in your response.")

                                if too_long:
                                    info_msg.append(f"MORE THAN {word_count_limit} WORDS!")
                                    complaints.append(f"Please do not exceed {word_count_limit} words.")

                                extra_messages.append({
                                    "role": "user",
                                    "content": ' '.join(complaints)
                                })
                                print(' '.join(info_msg))


                    os.makedirs(os.path.dirname(out_path), exist_ok=True)

                    with open(out_path, 'w', encoding='utf-8') as out_file:
                        out_file.write(choicest)

                    if choicest:
                        print(f"✅ {verse_ref} {commentator} {choicest_prompt} Done!")

                    # time.sleep(0.017) # follow rate limits


    # Generate BHT! 
    def ask_gpt_bht(self, verse_ref, choicest_prompts, bht_prompts, commentator_choicests, extra_messages):
        if not commentator_choicests:
            print(f"No commentary choicests found for {verse_ref}")
            return ""

        prompt_text = get_prompt(BHT_FOLDER_NAME, bht_prompts)

        messages = []
        messages.append({
            "role": "system",
            "content": prompt_text
        })

        tiers = {}
        for i in range(1, 4):
            tiers[i] = []

        for commentator, choicest in commentator_choicests.items():
            tier = self.get_commentator_tier(commentator)
            tiers[tier].append(choicest)

        join_commentary = lambda choicests: '\n'.join(choicests)

        messages.append({
            "role": "user",
            "content": f"[First tier commentary]\n{join_commentary(tiers[1])}\n\n[Second tier commentary]\n{join_commentary(tiers[2])}\n\n[Third tier commentary]\n{join_commentary(tiers[3])}\n\n"
        })

        messages.extend(extra_messages)

        model = "gpt-3.5-turbo"
        token_count = sum(len(self.ENCODING.encode(message["content"])) for message in messages)
        if token_count > 4097:
            print(f"ℹ️  {verse_ref} Too many tokens. Using 16k Context instead.")
            model += "-16k"

        try:
            chat_completion = openai.ChatCompletion.create(
                model=model, 
                messages=messages,
                request_timeout=15
            )
        except openai.error.InvalidRequestError:
            print(f"ℹ️  {verse_ref} {commentator} Something went wrong. Trying 16k Context.")
            chat_completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k", 
                messages=messages,
                request_timeout=15
            )

        return chat_completion.choices[0].message["content"]


    def ask_gpt_bht_with_retry(self, verse_ref, choicest_prompts, bht_prompts, commentator_choicests, extra_messages, tries=0, try_limit=10):
        if tries >= try_limit:
            raise Exception(f"❌ Failed {try_limit} times to get bht. Quitting. ❌")
        
        try:
            return self.ask_gpt_bht(verse_ref, choicest_prompts, bht_prompts, commentator_choicests, extra_messages)
        except TimeoutError:
            print(f"Attempt {tries} timed out. Trying again.")
            return self.ask_gpt_bht_with_retry(verse_ref, choicest_prompts, bht_prompts, commentator_choicests, extra_messages, tries + 1)


    def record_gpt_bht(self, verse_ref, choicest_prompts, bht_prompts, commentators, force_redo=False):
        for choicest_prompt in choicest_prompts:
            for bht_prompt in bht_prompts:
                debug_logs = []
                book, chapter, verse = get_book_chapter_verse(verse_ref)

                out_path = self.get_bht_output_path(choicest_prompt, bht_prompt, book, chapter, verse)

                # print(f"🟧 {verse_ref} {bht_prompt}")

                if not force_redo and os.path.exists(out_path) and not not open(out_path, 'r', encoding='utf-8').read().strip():
                    msg = f"✅ {verse_ref} {bht_prompt} File already populated."
                    debug_logs.append(msg)
                    print(msg)
                    continue

                commentator_choicests = self.get_commentary_choicests(verse_ref, choicest_prompt, commentators)

                # these should probably be constants or something
                proportion_limits = (0.5, 0.9)
                strict_proportion_limits = (0.5, 0.9)
                target_proportion = 0.7
                word_limits = (50, 100)
                strict_word_limits = (25, 130)
                target_word_count = 80
                min_proportion_limit, max_proportion_limit = proportion_limits
                min_word_limit, max_word_limit = word_limits

                extra_messages = []
                attempts_limit = 10
                current_attempt = 0

                best_bht = None

                while current_attempt < attempts_limit:
                    current_attempt += 1
                    bht_text = self.ask_gpt_bht_with_retry(verse_ref, choicest_prompt, bht_prompt, commentator_choicests, extra_messages)

                    choicest_quotes = {}
                    for commentator in commentator_choicests:
                        choicest_quotes[commentator] = []

                        for quote in commentator_choicests[commentator].splitlines():
                            quote = re.sub(r'^\d. *', '', quote)
                            quote = re.sub(r'^"', '', quote)
                            quote = re.sub(r'"$', '', quote)
                            choicest_quotes[commentator].append(quote)

                    current_bht = BHT(verse_ref, bht_text, choicest_quotes)
                    current_bht.run_generation_time_checks(self.bht_semantics.get_stop_words(), word_limits, proportion_limits, strict_word_limits, strict_proportion_limits, target_word_count, target_proportion)

                    # Keep track of best BHT we've seen across all attempts.
                    if current_bht > best_bht:
                        best_bht = current_bht

                    if current_bht.passes_checks():
                        break

                    else:
                        extra_messages.append({
                            "role": "assistant",
                            "content": bht_text
                        })

                        complaints = []

                        info_msg = [f"🔄 {verse_ref} (attempt {current_attempt}, {current_bht.word_count} words, {current_bht.proportion_percentage}% quotes", f"quality score: {current_bht.quality_score}, V2 normalized quality score: {current_bht.v2_normalized_quality_score}, commentator tiers 1-3: {(current_bht.t1_percent)}%, {(current_bht.t2_percent)}%, {(current_bht.t3_percent)}%)"]

                        if current_bht.too_many_words:
                            complaints.append(f"Please limit your response to {max_word_limit} words.")
                            info_msg.append(f"\n\t- BHT WAS OVER 100 WORDS!")
                        elif current_bht.not_enough_words:
                            complaints.append(f"Please make sure your response is at least {min_word_limit} words.")
                            info_msg.append(f"\n\t- BHT WAS UNDER {min_word_limit} WORDS!")
                        
                        # if current_bht.not_enough_from_quotes:
                        #     complaints.append(f"Please make sure at least {min_proportion_limit * 100}% of the words in your response come from the quotes.")
                        #     info_msg.append(f"\n\t- LESS THAN {min_proportion_limit * 100}% OF BHT WAS FROM QUOTES!")

                        elif current_bht.too_much_from_quotes:
                            complaints.append(f"Please make sure you are not just copying the quotes.")
                            info_msg.append(f"\n\t- OVER {max_proportion_limit * 100}% OF BHT WAS FROM QUOTES!")

                        if current_bht.commentator_in_tokens:
                            complaints.append(f"Please do not use the word 'commentator' in your response.")
                            info_msg.append(f"\n\t- 'COMMENTATOR(S)' FOUND IN BHT!")

                        if current_bht.list_detected:
                            complaints.append(f"Please do not provide any kind of list. Please make sure your response is a short paragraph of sentences.")
                            info_msg.append(f"\n\t- LIST FORMAT DETECTED!")

                        if current_bht.verse_in_tokens:
                            complaints.append(f"Please do not use the word 'verse' in your response.")
                            info_msg.append(f"\n\t- 'VERSE' FOUND IN BHT!")

                        if current_bht.passage_in_tokens:
                            complaints.append(f"Please do not use the word 'passage' in your response.")
                            info_msg.append(f"\n\t- 'PASSAGE' FOUND IN BHT!")

                        extra_messages.append({
                            "role": "user",
                            "content": ' '.join(complaints)
                        })
                        
                        msg = ' '.join(info_msg)
                        debug_logs.append(f"Attempt {current_attempt} BHT: {current_bht.bht}")
                        debug_logs.append(msg)
                        print(msg)

                os.makedirs(os.path.dirname(out_path), exist_ok=True)

                debug_logs.append(f"✅ {verse_ref} {bht_prompt} ({best_bht.word_count} words, {best_bht.proportion_percentage}% quotes)")
                debug_logs.append(f"quality score: {best_bht.quality_score}, V2 normalized quality score: {best_bht.v2_normalized_quality_score}, commentator tiers 1-3: {(best_bht.t1_percent)}%, {(best_bht.t2_percent)}%, {(best_bht.t3_percent)}%)")

                with open(out_path, 'w', encoding='utf-8') as out_file:
                    out_file.write(f"# {verse_ref} Commentary Help Text\n\n")
                    out_file.write(f"## BHT:\n{best_bht.bht}\n\n")

                    out_file.write(f"## Choicest Commentary Quotes:\n")
                    for commentator, choicest in self.get_commentary_choicests(verse_ref, choicest_prompt, commentators).items():
                        out_file.write(f"### {commentator}:\n{choicest}\n\n")

                    out_file.write("\n")

                    out_file.write(f"## Debug Info\n")
                    out_file.write(f"### Generation Details\n")
                    out_file.write(f"- Timestamp: {datetime.datetime.now().strftime('%m-%d-%Y %H:%M:%S')}\n")
                    out_file.write(f"- Choicest Prompt: \"{choicest_prompt}\"\n")
                    out_file.write(f"- BHT Prompt: \"{bht_prompt}\"\n")
                    out_file.write(f"- Commentators: \"{', '.join(commentators)}\"\n")
                    out_file.write(f"- BHT Word Count: {best_bht.word_count}\n")
                    out_file.write(f"- BHT Commentary Usage: {best_bht.proportion_percentage}%\n")
                    out_file.write(f"- BHT Quality Score: {best_bht.quality_score}\n")
                    out_file.write(f"- Generate Attempts: {current_attempt} / {attempts_limit}\n")
                    out_file.write(f"- ChatGPT injected words ({len(best_bht.injected_words)}):\n\t{best_bht.injected_words}\n")
                    out_file.write(f"- ChatGPT injected words (significant words only) ({len(best_bht.injected_significant_words)}):\n\t{best_bht.injected_significant_words}\n")
                    out_file.write('\n')
                    out_file.write(f"### Logs\n")
                    out_file.write('- ' + '\n- '.join(debug_logs))
                
                print(f"✅ {verse_ref} {bht_prompt} ({best_bht.word_count} words, {best_bht.proportion_percentage}% quotes, quality score: {best_bht.quality_score})")

                # time.sleep(0.017) # follow rate limits


    # Get all choicests and generate the bht from scratch.

    def generate_bht(self, verse_ref, choicest_prompts, bht_prompts, commentators, redo_choicest, redo_bht):
        self.record_gpt_choicest(verse_ref, choicest_prompts, commentators, redo_choicest)
        self.record_gpt_bht(verse_ref, choicest_prompts, bht_prompts, commentators, redo_bht)

    def generate_bhts(self, verse_refs, choicest_prompts, bht_prompts, commentators, redo_choicest=False, redo_bht=False):
        work_queue = MultiThreadedWorkQueue()

        for verse_ref in verse_refs:
            work_queue.add_task(self.generate_bht, (verse_ref, choicest_prompts, bht_prompts, commentators, redo_choicest, redo_bht))

        input(f"About to generate BHTs for {len(verse_refs)} verses. OK? ")

        work_queue.start()
        work_queue.wait_for_completion()
        work_queue.stop()
        