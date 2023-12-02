from datetime import datetime

def append_to_text_file(user_text: str, generated_text: str, filepath: str, agreeableness="disagree", length="medium"):
    '''
    Append text in "prompt" format to text file

    :param user_text: user input as a (string)
    :param generated_text: generated text from Cohere
    :param filepath
    :param agreeableness: string agree, disagree, statement, partially agree, curious
    :param length: short, medium and long
    '''

    values_for_prompt = [user_text, agreeableness, length, generated_text]
    final_text = generate_prompt(values_for_prompt=values_for_prompt)

    with open(filepath, "a") as file:
        file.write(final_text)

def generate_prompt(training_data="", history="", article="", values_for_prompt=["current_user","agreeableness","reply_length","cohere_user"]):
    """
    Generate layout for prompt as string
    """
    if article != "":
        article = "\ncontent:" + article

    current_user = "\ncurrent_user:" + values_for_prompt[0]
    agreeableness = "\nagreeableness:" + values_for_prompt[1]
    reply_length = "\nreply_length:" + values_for_prompt[2]
    cohere_user = "\ncohere_user:" + values_for_prompt[3]

    prompt = training_data + history + article + current_user + \
             agreeableness + reply_length + cohere_user

    return prompt

def length_classify(text: str):
    '''
    Classifies the length of the given text
    (this is to add to the train data)
    '''
    words = len(text.split())
    if words <= 25:
        return "short"
    elif words <= 50:
        return "medium"
    elif words > 50:
        return "long"

def log(txt, DEBUG=True):
    '''
    Logging on console for debugging
    '''

    if (DEBUG):
        print(datetime.now().strftime("[%d/%m/%Y %H:%M:%S] ") + str(txt))

def read_file_lines(DIR: str, lookback=-1):
    '''
    Read file with lookback (number of conversations OR num of batches of 5 lines)
    '''
    if lookback == -1:
        with open(DIR, 'r') as f:
            content = f.read()
    elif lookback > 0:
        with open(DIR) as f:
            content = f.readlines()[-5*int(lookback):]
            content = ''.join(content)
    else:
        raise ValueError('The lookback value has to be a positive integer')

    return content