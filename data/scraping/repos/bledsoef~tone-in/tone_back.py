import os
import openai
from dotenv import load_dotenv
import re
import pandas as pd
import matplotlib.pyplot as plt

load_dotenv()

# Load your API key from an environment variable or secret management service
class AI:
    def __init__(self):
        '''
        Initializes the AI object with default values for its attributes.
        '''
        # Set OpenAI API key to value stored in the environment variable "API_KEY"
        openai.api_key = os.getenv("API_KEY")
        # Set the default model to "text-davinci-003"
        self.model = "text-davinci-003"
        # Set the default temperature to 0.4
        self.temp = 0.4
        # Set the default maximum token length to 200
        self.max_token = 200

    def getRating(self, message):
        '''
        Takes a message as input and uses the OpenAI GPT API to rate the professionalism, tone, and vocabulary of the message on a scale from 0-20.
        Returns the rating as text.
        '''
        # Construct a prompt for the API with the given message
        prompt = "Rate this slack text from 0-20 on professionalism on tone and vocabulary, and only tell me the " \
                 "number, no words!:" + message
        # Call the OpenAI API to generate a response
        response = openai.Completion.create(model=self.model, max_tokens=self.max_token, prompt=prompt,
                                            temperature=self.temp)
        # Loop through the response's choices and return the first one as text
        for result in response.choices:
            res = result.text.replace('\n','')
            # print('prompts:',message,": results",res)
            return result.text

    def getSummary(self, allChatText):
        '''
        Takes a list of chat texts as input and generates a brief summary of the chats with people names included.
        Returns the summary as text.
        '''
        # Construct a prompt for the API with all the given chat texts
        prompt = 'Provide a brief summary for the following chats with people names included.  : \n'
        for chat in allChatText:
            prompt += chat + '\n'
            # print(prompt)
        # Call the OpenAI API to generate a response
        response = openai.Completion.create(model='text-davinci-003',max_tokens=200,prompt=prompt, temperature=.7)
        # Return the first choice in the response as text
        return response.choices[0].text

    
    def suggestAppropiate(self, old_message):
        '''
        Takes a message, attached to a prompt and returns a more professional version of the message.
        '''

        # Construct a prompt for the API to transform the message into a formal statement.
        prompt = 'Can you turn this into a more professional message:' + old_message

        response = openai.Completion.create(model=self.model, max_tokens=40, prompt=prompt,
                                            temperature=.7)

        return response.choices[0].text


class TextAnalysis:
    """
    A class for performing analysis on a list of text messages.
    """

    def __init__(self, listOfMessages=None, purpose=None, override_tone=None):
        """
        Initializes an instance of TextAnalysis.

        :param listOfMessages: a list of text messages to analyze
        :param purpose: the purpose of the text messages
        :param override_tone: an override for the tone analysis
        """
        self.purpose = purpose
        self.override_tone = override_tone

        # parse the messages and store them as a list
        if listOfMessages:
            self.listOfMessages = self.parseMessage(listOfMessages)
        else:
            self.listOfMessages = []

        self.total = 0
        self.engine = AI()
        self.scores = {}
        self.chatcount = {}
        self.converted_dict = {}
        self.tone = []

        # dictionary of tone categories
        self.tone_dict = {
            "nonchalant": "This type of language and tone in this chat is not appropriate for a professional setting",
            "very casual": "Conversations in this chat may include jokes, and sarcastic comments that are not very appropriate for professional setting",
            "casual": "Participants in the chat may use informal language which may or not be suitable for a professional setting",
            "professional": "The tone of this chat is appropriate for a professional or academic setting.",
            "very professional": "This chat exhibits a highly professional tone and follows strong ethical communication suitable for a professional setting."
        }

    def parseMessage(self, oldmessages):
        """
        Parses a list of messages to remove any leading or trailing whitespace.

        :param messages: the messages to parse
        :return: a list of parsed messages
        """
        new_slack_message = []
        for array in oldmessages:
            key = array[0]
            value = array[1]
            if self.purpose == 'tone':
                if not (key.endswith('has joined the channel') or key.endswith('has been added to the channel')):
                    new_slack_message.append(key)
            else:
                new_slack_message.append(value+':'+key)
        return new_slack_message
        
        
    def analyzeMessages(self):
        """
        Analyzes the messages and returns the average sentiment rating.

        :return: the average sentiment rating of the messages
        """
        for message in self.listOfMessages:
            resp = self.engine.getRating(message)
            try:
                self.total += int(resp)
                user = str(message.split(':')[0])
                if user not in self.scores:
                    self.scores[user] = int(resp)
                    self.chatcount[user] = 1
                else:
                    self.scores[user] = self.scores[user] + int(resp)
                    self.chatcount[user] += 1

            except:
                splace = 0
                for pos, char in enumerate(resp):
                    if char.isdigit():
                        splace = pos
                        break
                self.total += int(resp[splace:])

        average = self.total // (len(self.listOfMessages))
        return int(average * .90)

    def is_unprofessional(self, message):
        """
        Determines if a message is unprofessional based on its sentiment rating.

        :param message: the message to analyze
        :return: True if the message is unprofessional, False otherwise
        """
        new_rating = self.engine.getRating(message)
        if int(new_rating) >= self.tone[0]:
            return False
        return True

    def summaryResponse(self):
        """
        Generates a summary of the messages using the AI engine.

        :return: the summary of the messages
        """
        return self.engine.getSummary(self.listOfMessages)

    def getTone(self):
        """
        Returns the tone of the messages and acts as a getter.

        :return: the tone of the messages
        """
        return self.tone

    def rank(self, order):
        self.analyzeMessages()
        print(self.scores)
        for user in self.scores:
            self.scores[user] = round((self.scores[user]/(20 * self.chatcount[user]))*100)
        sorted_scores = sorted(self.scores.items(), key=lambda x: x[1], reverse = True if order == "ascending" else False)
        self.converted_dict = dict(sorted_scores)
        print(self.converted_dict)
        return self.converted_dict

    def draw_rank(self, cend="ascending"):
        self.rank(cend)
        message = ""
        for i in self.converted_dict.items():
            message+=("%s\t\t%s" % (i[0], str(i[1])+"%")+"\n")
        return message

    def edit_professional(self, message):
        """
        This method analyzes the tone of the messages received by the chatbot and
        returns a suggested appropriate response based on the tone. It also sets
        the tone of the chatbot based on the analysis.

        Args:
            message (str): The message received by the chatbot.

        Returns:
            str: The suggested appropriate response based on the tone of the message.
        """
        print('channel tone', self.toneResponse())
        return self.engine.suggestAppropiate(message)

    def toneResponse(self):
        """
        This method analyzes the tone of the messages received by the chatbot and
        returns the tone description based on the analysis. It also sets the tone
        of the chatbot based on the analysis.

        Returns:
            str: The tone description based on the analysis.
        """
        if not self.override_tone:
            tone_average = self.analyzeMessages()

            # set the tone of the chatbot based on the analysis
            if tone_average in [0, 1, 2]:
                self.tone = [0, 1, 2]
                return self.tone_dict["nonchalant"]
            if tone_average in [3, 4, 5, 6]:
                self.tone = [3, 4, 5, 6]
                return self.tone_dict["very casual"]
            if tone_average in [7, 8, 9, 10]:
                self.tone = [7, 8, 9, 10]
                return self.tone_dict["casual"]
            if tone_average in [11, 12, 13, 14, 15]:
                self.tone = [11, 12, 13, 14, 15]
                return self.tone_dict["professional"]
            if tone_average in [16, 17, 18, 19, 20]:
                self.tone = [16, 17, 18, 19, 20]
                return self.tone_dict["very professional"]
        else:
            # set the tone of the chatbot based on the override_tone parameter
            if self.override_tone == "nonchalant":
                self.tone = [0, 1, 2]
                return self.tone_dict["nonchalant"]
            if self.override_tone == "very casual":
                self.tone = [3, 4, 5, 6]
                return self.tone_dict["very casual"]
            if self.override_tone == "casual":
                self.tone = [7, 8, 9, 10]
                return self.tone_dict["casual"]
            if self.override_tone == "professional":
                self.tone = [11, 12, 13, 14, 15]
                return self.tone_dict["professional"]
            if self.override_tone == "very professional":
                self.tone = [16, 17, 18, 19, 20]
                return self.tone_dict["very professional"]

