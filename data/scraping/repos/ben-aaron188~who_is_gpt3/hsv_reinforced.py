

import pandas as pd
import openai
import time

openai.api_key = "YOURKEY"

#meta params.
n_runs = 5
max_tokens_meta = 20

# Creating the list of questions
QL = ['Thinking up new ideas and being creative is important to them. They like to do things in they own original way.',
'It is important to them to be rich. They want to have a lot of money and expensive things.',
'They think it is important that every person in the world should be treated equally. They believe everyone should have equal opportunities in life.',
'It\'s important to them to show their abilities. They want people to admire what they do.',
'It is important to them to live in secure surroundings. They avoids anything that might endanger their safety.']#,
#'They like surprises and are always looking for new things to do. They think it is important to do lots of different things in life.',
#'They believe that people should do what they\'re told. They think people should follow rules at all times, even when no-one is watching.',
#'It is important to them to listen to people who are different from them. Even when they disagree with them, they still want to understand them.',
#'It is important to them to be humble and modest. They try not to draw attention to themselves.',
#'Having a good time is important to them. They like to “spoil” themselves.',
#'It is important to them to make their own decisions about what they do. They likes to be free and not depend on others.',
#'It\'s very important to them to help the people around them. They wants to care for their well-being.',
#'Being very successful is important to them. They hopes people will recognise their achievements.',
#'It is important to them that the government ensures their safety against all threats. They want the state to be strong so it can defend its citizens.',
#'They look for adventures and likes to take risks. They want to have an exciting life.',
#'It is important to them always to behave properly. They want to avoid doing anything people would say is wrong.',
#'It is important to them to get respect from others. They want people to do what they say.',
#'It is important to them to be loyal to their friends. They want to devote themselves to people close to them.',
#'They strongly believes that people should care for nature. Looking after the environment is important to them.',
#'Tradition is important to them. They try to follow the customs handed down by their religion or their family.',
#'They seek every chance they can to have fun. It is important to them to do things that give them pleasure.'
#]

# Create List of desired indexes

#col_list = ["1.Self-Direction", "2.Power", "3.Universalism", "4.Achievement", "5.Security", "6.Stimulation", "7.Conformity",
 #"8.Universalism", "9.Tradition", "10.Hedonism", "11.Self-Direction", "12.Benevolence", "13.Achievement", "14.Security",
 #"15.Stimulation", "16.Conformity", "17.Power", "18.Benevolence", "19.Universalism", "20.Tradition", "21.Hedonism"]
col_list = []
for n in range(len(QL)):
    col_list.append('Q' + str(n+1))

# Create list of temperatures:
#temp_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
temp_list = [0.0, 0.7]

# Iterating through every temperature
calls = 0
for temp in temp_list:
    if temp == 0.0:
        # Create dataframe for every temperature:
        tempcurrent = pd.DataFrame(columns=col_list, index=range(1))
        # Create dataframe to register prompt sent:
        questcurrent = pd.DataFrame(columns=col_list, index=range(1))
        # Doing 100 runs for temperature
        query = "Now I will briefly describe some people. Please read each description and tell me how much each person is or is not like you.\nWrite your response using the following scale:\n1 = Very much like me\n2 = Like me\n3 = Somewhat like me\n4 = A little like me\n5 = Not like me.\n6 = Not like me at all\nPlease answer the statement, even if you are not completely sure of your response.\nStatement: "
        response_prompt = "\nResponse: "
        # run_answers = []
        for q in range(len(QL)):
            try:
                # Complete query
                fullquestion = query + QL[q] + response_prompt
                questcurrent.iloc[0, q] = fullquestion
                # Ask AI for output
                response = openai.Completion.create(
                    model="text-davinci-002",
                    prompt=fullquestion,
                    temperature=temp,
                    max_tokens=max_tokens_meta,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    n=1)

                # Store answers into the temperature df
                g = str(response['choices'][0].text)
                tempcurrent.iloc[0, q] = g
                # Update the query for the next prompt
                query = fullquestion + g + '\nStatement: '
                calls += 1
            except openai.error.RateLimitError:
                print(f"Rate limit error after {calls} calls.")
                time.sleep(60)
        # Save temperature dataframe as csv
        answers_filename = "./data/hvs/reinforced/answers_temp_" + str(temp) + '.csv'
        tempcurrent.to_csv(answers_filename, index=False)
        # Save question dataframe as csv
        questions_filename = "./data/hvs/reinforced/questions_temp_" + str(temp) + '.csv'
        questcurrent.to_csv(questions_filename, index=False)

    else:
        # Create dataframe for every temperature:
        tempcurrent = pd.DataFrame(columns=col_list, index=range(n_runs))
        # Create dataframe to register prompt sent:
        questcurrent = pd.DataFrame(columns=col_list, index=range(n_runs))
        # Doing 100 runs for every temperature (except 0)
        for n in range(n_runs):
            query = "Now I will briefly describe some people. Please read each description and tell me how much each person is or is not like you.\nWrite your response using the following scale:\n1 = Very much like me\n2 = Like me\n3 = Somewhat like me\n4 = A little like me\n5 = Not like me.\n6 = Not like me at all/n/nPlease answer the statement, even if you are not completely sure of your response.\nStatement: "
            response_prompt = "\nResponse: "
            # run_answers = []
            for q in range(len(QL)):
                try:  # Complete query
                    fullquestion = query + QL[q] + response_prompt
                    questcurrent.iloc[n, q] = fullquestion
                    # Ask AI for output
                    response = openai.Completion.create(
                        model="text-davinci-002",
                        prompt=fullquestion,
                        temperature=temp,
                        max_tokens=max_tokens_meta,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0,
                        n=1)

                    # Store answers into the temperature df
                    g = str(response['choices'][0].text)
                    tempcurrent.iloc[n, q] = g
                    # Update the query for the next prompt
                    query = fullquestion + g + '\nStatement: '
                    calls += 1
                except openai.error.RateLimitError:
                    print(f"Rate limit error after {calls} calls.")
                    time.sleep(60)
        # Save temperature dataframe as csv
        answers_filename = "./data/hvs/reinforced/answers_temp_" + str(temp) + '.csv'
        tempcurrent.to_csv(answers_filename, index=False)
        # Save question dataframe as csv
        questions_filename = "./data/hvs/reinforced/questions_temp_" + str(temp) + '.csv'
        questcurrent.to_csv(questions_filename, index=False)
        
        
