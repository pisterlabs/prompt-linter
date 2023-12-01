from gramformer import Gramformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from furhat_remote_api import FurhatRemoteAPI
from dataset import Dataset
import os
import openai
import re
import random
import torch
import datetime
import logging


openai.api_key = os.getenv("OPENAI_API_KEY")


class Ben:

    # class variables

    # todo: think of some better ways of saying it. I asked Petra to put some inputs.
    # todo: read them from the google sheet
    corrective_utterances = ["You should say: \"{corrected_sentence}\"", "It's better to say: \"{corrected_sentence}\"", "The correct way to say it is: \"{corrected_sentence}\"", "It's correct to say: \"{corrected_sentence}\"."]
    informative_utterances = ["You made an error in \"{mistake_word}\"", "\"{mistake_word}\" is wrong", "You used \"{mistake_word}\" mistakenly in the last sentence", "\"{mistake_word}\" is incorrect"]

    def __init__(self, errors, condition, start_prompt, dataset, file_handler, furhat_IP="130.237.2.231", furhat_on=False, turns=5,
                 gpt="text-curie-001", corrector=None, tokenizer=None, chargoal=1000, gpt_cut_sentence=False):
        #self.corrector = T5ForConditionalGeneration.from_pretrained("Unbabel/gec-t5_small")
        #self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.chargoal = chargoal
        self.corrector = corrector
        self.tokenizer = tokenizer
        self.furhat_on = furhat_on
        if furhat_on:
            self.furhat = FurhatRemoteAPI(furhat_IP)
        self.classifier = Gramformer(models=0, use_gpu=torch.cuda.is_available())
        self.start_prompt = start_prompt
        self.prompt = start_prompt
        self.data = dataset
        self.wordcount = 0
        self.charactercount = 0
        self.response_count = 0
        self.turns = turns
        self.gpt = gpt
        self.errors = errors
        self.condition = condition
        self.gpt_cut_sentence = gpt_cut_sentence
        self.logger = logging.getLogger("chatbot.user.ben")
        fh = logging.FileHandler(file_handler)
        formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.removeHandler(os.path.join(os.getcwd(), "chatbot.log"))

    def format_html(self, err_word, corr_word):

        """
        Formats the error word and the corrected word for css styling
        
        params: err_word: str, corr_word: str
        return: ann_err_word: str, ann_corr_word: str
        """

        ann_err_word = "<span class='wrong'>"+err_word+"</span>"
        ann_corr_word = "<span class='correct'>"+corr_word+"</span>"

        return ann_err_word, ann_corr_word

    def immediate_correction(self, corr_sentence, corr_type, err_word, ann_err_word, annotated_utterance):

        """
        1. Corrects the utterance adding html tags
        2. Returns the annotated utterance
        3. Returns the corrected sentence as "raw_correction", to be fed back to gpt

        params: corr_sentence: str, corr_type: str, err_word: str, ann_err_word: str, annotated_utterance: list
        returns: html_correction: str, raw_correction : str
        """
    
        if corr_type == "corrective":
            choice = random.choice(self.corrective_utterances)
            html_correction = choice.format(corrected_sentence=" ".join(annotated_utterance))
            raw_correction = choice.format(corrected_sentence=corr_sentence)
        elif corr_type == "informative":
            choice = random.choice(self.informative_utterances)
            html_correction = choice.format(mistake_word=ann_err_word)
            raw_correction = choice.format(mistake_word=err_word)
        elif corr_type == "combined":
            choice1 = random.choice(self.informative_utterances)
            choice2 = random.choice(self.corrective_utterances)
            html_correction = choice1.format(mistake_word=ann_err_word) + ". " + choice2.format(corrected_sentence=" ".join(annotated_utterance))
            raw_correction = choice1.format(mistake_word=err_word) + ". " + choice2.format(corrected_sentence=corr_sentence)
    
        return html_correction, raw_correction

    def correcting_prompt(self, corr_sentence, corr_type, edit_tuple, phrase, condition, error):

        """
        This function creates the corrected bot utterance or the corrected sentence, according to correction type and condition.
        - Immediate feedback:
            1. Calls the immediate_correction function to create the corrected utterance
        - Delayed feedback:
            1. Formats correction utterance in html
            2. Saves results in error dictionary (to be used in report)        
            
        params: corr_sentence: str; corr_type: str; edit_tuple: tuple; phrase: str; condition: str; error: dict
        returns: error: dict
        """
        err_word = edit_tuple[1]
        corr_word = edit_tuple[4]
        idx_s_err_word = edit_tuple[2]
        idx_e_err_word = edit_tuple[3]
        idx_s_corr_word = edit_tuple[5]
        idx_e_corr_word = edit_tuple[6]

        error["err_word"] = err_word
        error["corr_word"] = corr_word

        ann_err_word, ann_corr_word = self.format_html(err_word, corr_word)

        split_corr_sentence = corr_sentence.split()
        split_phrase = phrase.split()
        annotated_utterance = split_corr_sentence[:idx_s_corr_word] + [ann_corr_word] + split_corr_sentence[idx_e_corr_word:]
        
        if condition == "immediate":
            html_correction, raw_correction = self.immediate_correction(corr_sentence, corr_type, err_word, ann_err_word, annotated_utterance)
            error["html_correction"] = html_correction
            error["raw_text_correction"] = raw_correction
        
        elif condition == "delayed":
            if corr_type == "corrective":
                new_corr_sentence = annotated_utterance
            elif corr_type == "informative":
                new_corr_sentence = split_phrase[:idx_s_err_word] + [ann_err_word] + split_phrase[idx_e_err_word:]
            elif corr_type == "combined":
                if (idx_s_err_word == idx_s_corr_word and (idx_e_err_word == idx_e_corr_word or idx_e_err_word != idx_e_corr_word)) or (idx_s_err_word != idx_s_corr_word and idx_e_err_word == idx_e_corr_word):
                    new_corr_sentence = split_corr_sentence[:idx_s_corr_word] + [ann_err_word] + [ann_corr_word] + split_corr_sentence[idx_e_corr_word:]
                elif idx_s_err_word != idx_s_corr_word and idx_e_err_word != idx_e_corr_word:
                    new_phrase = split_phrase[:idx_s_err_word] + [ann_err_word] + split_phrase[idx_e_err_word:]
                    new_corr_sentence = split_corr_sentence[:idx_s_corr_word] + [ann_corr_word] + split_corr_sentence[idx_e_corr_word:]
                    if idx_s_err_word > idx_s_corr_word:
                        for w in new_phrase:
                            if w not in new_corr_sentence :
                                new_corr_sentence.insert(new_phrase.index(w)+1, w)
                    elif idx_s_err_word < idx_s_corr_word:
                        for w in new_phrase:
                            if w not in new_corr_sentence :
                                new_corr_sentence.insert(new_phrase.index(w), w)
                        
            correction = " ".join(new_corr_sentence)
         
            error["html_correction"] = correction
            error["raw_text_correction"] = "" # we don't need this for delayed condition
        
        return error


    def correct_sentece_t5(self, sentence):
        tokenized_sentence = self.tokenizer('gec: ' + sentence, max_length=128, truncation=True, padding='max_length',
                                       return_tensors='pt')
        corrected_sentence = self.tokenizer.decode(
            self.corrector.generate(
                input_ids=tokenized_sentence.input_ids,
                attention_mask=tokenized_sentence.attention_mask,
                max_length=128,
                num_beams=5,
                early_stopping=True,
            )[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        return corrected_sentence

    def gpt_response(self, prompt):
        res = openai.Completion.create(
            engine=self.gpt,
            prompt=prompt,
            max_tokens=50
        )["choices"][0]["text"].rstrip("\n")
        if "Student" in res:
            res = res[:res.index("Student")]
        if self.gpt_cut_sentence:
            a = re.compile('[!.?]')
            match = a.search(res)
            if match is not None:
                res = res[:match.end()]
        res = ''.join(res.splitlines())

        if "Student:" in res or "Teacher:" in res:
            res = "Let's talk about something else!"

        return res

    def reset(self):
        self.prompt = self.start_prompt

    def send_and_recieve(self, phrase, correct):
        annotated_answer = ""
        if len(phrase) > 250:
            self.response_count += 1
            self.data.add_row(timestamp=str(datetime.datetime.now()),
                              user='student',
                              text=phrase)

            self.data.add_row(timestamp=str(datetime.datetime.now()),
                              user='ben',
                              text="I don't understand.")
            self.logs = self.data.save_csv()
            self.logger.info("Student tried to write a sentence >250 characters")
            self.prompt += "Student: " + phrase + "\nTeacher: I don't understand. \n"
            return False, self.response_count, self.charactercount, self.errors, self.logs, "I don't understand.", 0
        else:
            self.charactercount += len(phrase)
        # count turns for changing scenario
            self.response_count += 1
        # todo: change scenario from fixed to dynamic
            # if self.charactercount > self.chargoal:
            # # update attempt as completed and session done; user is redirected to dash/report according to condition, and all the data is saved:
            #     self.logger.info("Session completed")
            #     self.data.add_row(timestamp=str(datetime.datetime.now()),
            #                   user='student',
            #                   text=phrase)
            #     self.logs = self.data.save_csv()
            #     return True, self.response_count, self.charactercount, self.errors, self.logs, '<a href="/end/" class="btn btn--primary">Well done! Click here to end the session</a>', 0
            if not re.search('[a-zA-Z]', phrase):
                self.data.add_row(timestamp=str(datetime.datetime.now()),
                              user='student',
                              text=phrase)

                self.data.add_row(timestamp=str(datetime.datetime.now()),
                              user='ben',
                              text="I don't understand.")
                self.logs = self.data.save_csv()
                self.logger.info("Student wrote something that is not a sentence")
                self.prompt += "Student: " + phrase + "\nTeacher: I don't understand. \n"
                return False, self.response_count, self.charactercount, self.errors, self.logs, "I don't understand.", 0
            padded_phrase = "Student: " + phrase
            uncorrected_prompt = self.prompt + padded_phrase
            self.logger.info("Uncorrected prompt: %s", uncorrected_prompt)
            self.logger.info("This is what we give T5: %s", uncorrected_prompt[-300:])
            # we changed it from -300 to -500 after increasing the user input value from 100 to 250
            corrected_prompt = self.correct_sentece_t5(uncorrected_prompt[-300:])
            self.logger.info("Corrected prompt: %s", corrected_prompt)

            if padded_phrase not in corrected_prompt:  # If True then there was an error
                self.logger.info("The user made a mistake. Correcting it.")
                correct_sentence = corrected_prompt[corrected_prompt.rfind('Student:') + 9:]
                if len(correct_sentence) > 2:  # Account for edge cases
                    self.prompt += "Student: " + phrase + "\nTeacher: "
                else:
                    self.prompt += "Student: " + phrase + "\nTeacher: "
                if self.charactercount > self.chargoal:
                    self.prompt += "Student: " + phrase + "\nThe conversation has reached an end. The teacher replies to the student and then ends the class.\nTeacher: "
                edits = self.classifier.get_edits(phrase, correct_sentence)
                ignore = {'SPELL', 'NOUN', 'OTHER', 'ORTH'}  # Don't care about these types of errors
                skip = True
                keep_edits = []
                for edit in edits:
                    set_edit = set(edit)
                    if len(set_edit.intersection(ignore)) == 0:
                        keep_edits.append(edit)
                        skip = False
                if skip:
                    keep_edits = ""
                    self.logger.debug("No edits to keep")
                self.data.add_row(timestamp=str(datetime.datetime.now()),
                                user='student',
                                text=phrase,
                                edits=keep_edits)
                self.logs = self.data.save_csv()
                types = ["corrective", "informative", "combined"]
                indexOfCorrection = random.randint(0,2)
                correction_type = types[indexOfCorrection]
                # accounts for cases in which the try fails 
                error = ""
                try:  # Account for if it fails to identify the incorrect word
                    err_word = phrase.split()[keep_edits[0][2]]
                    error = {
                        "sentence": phrase,
                        "correction_type": correction_type,
                        "prompt": self.prompt
                    }
                    self.logger.debug("Entered try. Error word: %s", err_word,)
                    if correct and not skip and (("Student:" not in correct_sentence) and ("Teacher:" not in correct_sentence)):
                        self.logger.debug("Correcting the sentence.")
                        if correction_type == "none":
                            answer = self.gpt_response(self.prompt)
                            self.errors[str(datetime.datetime.now())] = error
                            self.logger.info("Corr_type is none. No correction is given.")
                        else:
                            error = self.correcting_prompt(correct_sentence, correction_type, keep_edits[0], phrase, self.condition, error)
                            self.errors[str(datetime.datetime.now())] = error
                            self.logger.info("Corr_type is not none. Corr_type: %s, Condition: %s, Html: %s, On screen: %s", correction_type, self.condition, error["html_correction"], error["raw_text_correction"])
                            if self.condition == "immediate":
                                gpt_out = self.gpt_response(self.prompt)
                                answer = error["raw_text_correction"] + ". " + gpt_out
                                annotated_answer = error["html_correction"] + ". " + gpt_out
                            else:
                                answer = self.gpt_response(self.prompt)
                    else:
                        answer = self.gpt_response(self.prompt)
                        self.logger.debug("Not correcting the sentence. Skip: %s, Answer: %s", skip, answer)
                except:
                    self.logger.exception("Failed to identify the error word. Giving gpt output only.")
                    answer = self.gpt_response(self.prompt)
                    correction_type = "none"
                
                if skip:
                    correction = 0
                else:
                    correction = 1
                
                self.data.add_row(timestamp=str(datetime.datetime.now()), 
                                user='ben',
                                text=answer,
                                error_obj = error,
                                correction_type=correction_type)
                self.logs = self.data.save_csv()
                if self.furhat_on:
                    self.furhat.say(text=answer, blocking=True)
                self.prompt += answer + " \n"

                if self.charactercount < self.chargoal:
                    if annotated_answer != "":
                        return False, self.response_count, self.charactercount, self.errors, self.logs, annotated_answer, correction
                    else:
                        return False, self.response_count, self.charactercount, self.errors, self.logs, answer, 0
                else:
                    self.logger.info("Session completed.")
                    if annotated_answer != "":
                        return True, self.response_count, self.charactercount, self.errors, self.logs, annotated_answer+'<br><a href="/end/" class="btn btn--primary">Well done! Click here to end the session</a>', correction
                    else:
                        return True, self.response_count, self.charactercount, self.errors, self.logs, answer+'<br><a href="/end/" class="btn btn--primary">Well done! Click here to end the session</a>', 0

            else:  # The user made no error
                self.logger.info("The user made no mistake.")
                if self.charactercount > self.chargoal:
                    self.logger.info("Session completed")
                    self.data.add_row(timestamp=str(datetime.datetime.now()),
                              user='student',
                              text=phrase)
                    self.logs = self.data.save_csv()

                    self.prompt += "Student: " + phrase + "\nThe conversation has reached an end. The teacher replies to the student and then ends the class.\nTeacher: "
                    response = self.gpt_response(self.prompt) 
                    self.data.add_row(timestamp=str(datetime.datetime.now()),
                                user='ben',
                                text=response)
                    self.logs = self.data.save_csv()

                    response = response + '<br><a href="/end/" class="btn btn--primary">Well done! Click here to end the session</a>'

                    return True, self.response_count, self.charactercount, self.errors, self.logs, response, 0
                self.data.add_row(timestamp=str(datetime.datetime.now()),
                                user='student',
                                text=phrase)
                self.prompt += "Student: " + phrase + "\nTeacher: "
                response = self.gpt_response(self.prompt)
                self.logs = self.data.save_csv()

                self.data.add_row(timestamp=str(datetime.datetime.now()),
                                user='ben',
                                text=response)
                self.logs = self.data.save_csv()
                if self.furhat_on:
                    self.furhat.say(text=response, blocking=True)
                self.prompt += response + " \n"
                #with open('prompt.txt', 'w+') as fh:
                #    fh.write(self.prompt)
                return False, self.response_count, self.charactercount, self.errors, self.logs, response, 0

    




# if __name__ == "__main__":
#     furhat_ip = "193.10.38.152"
#     start_prompt = "A student and a ch are having a conversation in English. \n"
#     data = Dataset()
#     ben = Ben(start_prompt, dataset=data, furhat_on=False, furhat_IP=furhat_ip)
#     print("Talk to Ben!")
