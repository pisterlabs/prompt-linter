import os
import openai
from online_dataset import OnlineDataset
import fileinput
import re




import requests

class API:
    '''
    Class to access the OpenAI API.
    '''

    url = "https://api.openai.com/v1/chat/completions"
    

    def generate_chat_response(self, topic, length=300):
        '''
        Take a topic as input and return a response from the GPT-3 model with corresonding length.
        '''
        url = self.url
        modified_topic = topic.lower()
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer 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"
        }
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": f"Without mentioning that you are an AI language model and using {length} words , {modified_topic} Add a \\n on the same line of your last sentence."}],
            "temperature": 1
        }
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 401:
            print("Error: You didn't provide an API key. You need to provide your API key in an Authorization header using Bearer auth (i.e. Authorization: Bearer YOUR_KEY), or as the password field (with blank username) if you're accessing the API from your browser and are prompted for a username and password. You can obtain an API key from https://platform.openai.com/account/api-keys.")
            return None
        else:
            try:
                return response.json()["choices"][0]["message"]["content"]
            except KeyError:
                print(f"The topic {topic} failed to generate a response.")
                

    def generate_train_dataset_from_api(self, output_path, topic_title, length=300, num_topics= 286, num_training_examples=4):
        '''
        Generate a training dataset of up to "k" topics of length "length" in topic_title by calling generate_chat_response() function. The number of training examples
        is specified by num_training_examples. Export the dataset to a csv file.

        The default number of topics is 286 because that is the number of topics in the dataset.
        The default number of training examples is 4 because that is the number of arguments in the dataset.
        When an IndexError is reached, all the data has been collected and the function returns.
        '''
        for k in range(num_topics):
            for i in range(num_training_examples):
                 with open(output_path, 'a', newline='') as file:
                    if file.tell() == 0:  # check if file is empty
                        file.write("Training examples generated from the OpenAI API.\n")
                    label = f"{k+1}_{i}"  # generate label for the kth *ith sample
                    try:
                        text = self.generate_chat_response(topic_title[k], length)
                    except ConnectionResetError: #to handle connection reset errors
                        print("ConnectionResetError: [Errno 54] Connection reset by peer")
                        continue
                    except ValueError: #to handle JSONDecodeError
                        print("ValueError: Expecting value: line 1 column 1 (char 0)")
                        continue
                    except IndexError:
                        print("Data collection complete.")
                        return
                    file.write(f"{label}\t{text}\n")
        return 

    def ai_remover(self, input_path, output_path, text_to_remove):
        '''
        Remove text_to_remove from the input_path and write the new text to output_path.
        '''
        with open(input_path, "r") as file:
            contents = file.read()
        new_contents = contents.replace(output_path, '').replace(text_to_remove, '')
        with open(output_path, 'w') as file:
            file.write(new_contents)
        
    
    def convert_text_to_lst(self, input_path):
        '''
        Convert the text in input_path to a list.
        '''
        lst = []
        sample = ""
        with open(input_path, 'r') as file:
            for line in file:
                if "\\n" in line:
                    lst.append(sample)
                    sample = ""
                elif "_" in line:
                    sample += line[4:].lstrip()
                else:
                    sample += line
        return lst
    
    def generate_label(self, input_lst):
        '''
        Generate labels for the input_lst. The labels are just [1] * len(input_lst).
        '''
        return [1]*len(input_lst)
    
    def train_dataset_characteristics(self, input_lst):
        '''
        Return characteristics of the AI-generated training dataset.
        - Number of training examples
        - Average length of training examples
        
        '''
        num_training_examples = len(input_lst)
        total_length = 0
        for sample in input_lst:
            total_length += len(sample)
        avg_length = total_length/num_training_examples
        print(f"Number of training examples: {num_training_examples}")
        print(f"Average length of training examples: {avg_length}")
        
            

    

if __name__ == "__main__":
    online_dataset = OnlineDataset()
    topic_title = online_dataset.extract_topics('arg_search_framework/data/essay/train.json')
    lst_of_arguments = online_dataset.extract_argument('arg_search_framework/data/essay/train.json')
    api = API()
    # api.ai_remover("new_train_dataset_api.txt", "new_train_dataset_api.txt", "As a language model designed to assist humans in their tasks, ")
    lst = api.convert_text_to_lst("new_train_dataset_api.txt")
    print(lst[100])
    # api.train_dataset_characteristics(api.convert_text_to_lst("new_train_dataset_api.txt"))
    
    
    
    
    
        
    