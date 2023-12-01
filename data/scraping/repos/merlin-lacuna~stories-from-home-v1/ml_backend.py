import openai
import streamlit as st
import os
import re
import random

class ml_backend:

    openai.api_key = st.secrets["OPENAI_API_KEY"]
    # openai.api_key = os.getenv("OPENAI_API_KEY")

    def trim_output(completion):
        try:
            if completion[-1] in ('.', '?', '!'):
                # print("matched end")
                trimmedoutput = completion
            else:
                try:
                    # print("matched incomplete")
                    re.findall(r'(\.|\?|\!)( [A-Z])', completion)
                    indices = [(m.start(0), m.end(0)) for m in re.finditer(r'(\.|\?|\!)( [A-Z])', completion)]
                    splittuple = indices[len(indices) - 1]
                    trimmedoutput = completion[0:splittuple[0] + 1]
                except:
                    trimmedoutput = completion
        except:
            trimmedoutput = completion

        return trimmedoutput

    def clean_response(self,rawresponse):
        try:
            response = openai.Edit.create(
                model="text-davinci-edit-001",
                input=rawresponse,
                instruction="Remove any garbled text and correct any incorrectly duplicated characters such as too many full stops. Correct the grammar in all sentences and convert all text to sentence case. Make sure that each sentence is written in the first person tense and make all pronouns gender neutral",
                temperature=0.8,
                top_p=1
            )
            cleanedresp = response.choices[0].text
        except:
            print("The edit operation failed for some reason.\r Returning raw response:")
            cleanedresp = rawresponse

        return cleanedresp

    def generate_text(self, myprompt, maxt, element):
        #print(extra)
        print("max t is: " + str(maxt))
        lengthext = random.randint(1, 56)
        maxt = int(maxt) + int(lengthext)
        response = openai.Completion.create(
            model=element,
            prompt=myprompt,
            temperature=1,
            max_tokens=maxt,
            top_p=1,
            frequency_penalty=1,
            presence_penalty=1,
            stop=["ACT1", "ACT2", "ACT3", "ACT4"]
        )
        story = response.choices[0].text

        # START TEST
        #story = "I am a boring story that is a placeholder for testing that uses the model: " + element + 'with max tokens'  + str(maxt) + 'and the prompt' + myprompt

        lstory = story.replace("\n", " ")
        lstory = lstory.replace("I'm a forest,", "I am")
        lstory = lstory.replace("I am a forest,", "I am")
        lstory = lstory.replace("I'm just a forest,", "I am")
        lstory = lstory.replace("I am just a forest,", "I am")
        lstory = lstory.replace("Forest: ", "")

        return ' '.join(lstory.split())

    def gengpt4_text(self, myprompt, maxt, persona):
        # openai.api_key = os.getenv("OPENAI_API_KEY_MC")
        openai.api_key = st.secrets["OPENAI_API_KEY_MC"]
        myprompt = myprompt.replace("The first act starts like this:", "Continue the play using a mixture of the literary styles that you have been trained on. Also make sure that the lines in your text do not rhyme. For example, the poem 'Roses are red\nViolets are blue\nThe sun is shining\nAnd I love you' has a rhyming structure, because 'blue' rhymes with 'you'. You should avoid this structure and instead write in free prose.\n\nThe first act starts like this:")
        response = openai.ChatCompletion.create(
            model="gpt-4",
            max_tokens=int(maxt),
            messages=[
                {"role": "system", "content": persona},
                {"role": "user", "content": myprompt},
            ]
        )
        return "GPT4 OUTPUT:\n\n" + response.choices[0].message.content

    def get_act(self,myprompt, maxt, element):
        lengthext = random.randint(1, 56)
        maxt = maxt + lengthext
        print("Requesting generation with model: " + element)
        print("Requesting generation with maxlength: " + str(maxt))
        response = openai.Completion.create(
            model=element,
            prompt=myprompt,
            temperature=0.8,
            max_tokens=maxt,
            top_p=1,
            frequency_penalty=1,
            presence_penalty=1,
            stop=["ACT1","ACT2","ACT3","ACT4"]
        )
        story = response.choices[0].text

        # START TEST
        #story = "I am a boring story that is a placeholder for testing that uses the model: " + selectedmodel

        lstory = story.replace("\n", " ")
        lstory = lstory.replace("I'm a forest,", "I am")
        lstory = lstory.replace("I am a forest,", "I am")
        lstory = lstory.replace("I'm just a forest,", "I am")
        lstory = lstory.replace("I am just a forest,", "I am")
        lstory = lstory.replace("Forest: ","")

        return ' '.join(lstory.split())
