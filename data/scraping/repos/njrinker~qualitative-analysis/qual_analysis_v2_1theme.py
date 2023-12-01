#!/usr/bin/env python3
import os
import sys
import csv
import pandas as pd
import numpy as np
from io import StringIO
import openai
import dotenv
import glob

# Load OpenAI API key so calls can be made to the API
config = dotenv.dotenv_values(".env")
openai.api_key = config['OPENAI_API_KEY']


# Fetch the number of iterations to perform from the command line and sanity check it
iter_num = int(sys.argv[2])
if iter_num <= 0:
    raise Exception("Number of runs to perform must be greater than 0")
if iter_num > 10:
    raise Exception("Number of runs to perform must be less than or equal to 10")


tries = 3


# Fetch the folder to be operated on from the command line and convert it into a pandas dataframe
folder_in = sys.argv[1]
folder_name = os.path.basename(folder_in)
folder = os.path.splitext(folder_name)
path = os.path.normpath(folder_in).split(os.path.sep)


# Sanity check input and format output directory
if not os.path.exists(folder_in):
    raise Exception('Input folder does not exist')
if not os.path.exists('files_out\\' + path[1]):
    os.mkdir('files_out\\' + path[1])
if not os.path.exists('files_out\\' + path[1] + '\\' + folder[0]):
    os.mkdir('files_out\\' + path[1] + '\\' + folder[0])


# Read in the files from the input folder and format them
files_in = []
for fname in glob.glob(folder_in + '/*.csv'):
    if not os.path.exists('files_out\\' + fname[9:]):
        try:
            file_in = pd.read_csv(fname)
        except UnicodeDecodeError:
            print("UnicodeDecodeError recieved, reattempting {} in ANSI".format(fname))
            file_in = pd.read_csv(fname, encoding='ansi')
        if not file_in.empty:
            files_in += [(file_in, fname)]  
print("All files fetched and formatted")

# Perform Qualitative Analysis for each file in inputted directory
for f, name in files_in:
    print("Starting qualitative analysis on " + name)
    
    
    # Remove all strings less than 100 characters from the dataframe and put those strings into their own
    file = f.iloc[:, :1].replace(',|\"|\'|;|:|\.|!|/|’|\(|\)|[|]','', regex=True)
    file_short = file.loc[file[file.columns[0]].str.len()<150]
    file = file.loc[file[file.columns[0]].str.len()>=150]


    # Remove all strings more than 500 characters from the dataframe and put those strings into their own
    file_long = file.loc[file[file.columns[0]].str.len()>500]
    file = file.loc[file[file.columns[0]].str.len()<=500]


    # Split the dataframes into smaller sized dataframes, if the initial dataframe is too large
    file_len = len(file)
    if file_len > 10:
         files = np.array_split(file, -(file_len//-10), axis=0)
    elif file_len > 0:
        files = [file]
    else:
        files = []

    file_short_len = len(file_short)
    if file_short_len > 10:
        files_short = np.array_split(file_short, -(file_short_len//-10), axis=0)
        for short in files_short:
            files.append(short)
    elif 0 < file_short_len <= 10:
        files.append(file_short)

    file_long_len = len(file_long)
    if file_long_len > 5:
        files_long = np.array_split(file_long, -(file_long_len//-5), axis=0)
        for long in files_long:
            files.append(long)
    elif 0 < file_long_len <= 5:
        files.append(file_long)

    dfs_out = []
    print("Input data parsed and formatted for processing")


    # If the inputted csv file is too long, we split it into portions and run the program once for each portion, then merge the resulting data at the end
    for df in files:
        df_init = df.copy()
        df_init_len = str(len(df_init))
        sent_num, them_num, file_num = 1, 1, 1
        for x in range(iter_num):
            print("Starting iteration " + str(x+1))
            iter = 0
            for iter in range(tries):
                try:
                    # Write the prompt for the sentiment analysis function, using the entire csv file at once
                    # Send the prompt to the chat bot and record the response to the 'sentiment' list
                    prompt = """In one word, does each sentence in the following list have a positive, negative, or neutral sentiment. 
                                The list has {} sentences, so there should be exactly {} words. 
                                Output should be a comma seperated list: \n""".format(df_init_len, df_init_len) + df_init.to_csv(index=False, header=False)
                    response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "user", "content": prompt},
                            ],
                            temperature=0.5,
                        )
                    sentiment = response['choices'][0]['message']['content'].split(',')
                    sentiment = [s.strip(' ') for s in sentiment]
                    sentiment = [x.lower() for x in sentiment]
                    
                    
                    # Save the responses to the pandas dataframe
                    while 'Sentiment {}'.format(sent_num) in df.columns:
                        sent_num += 1
                    df['Sentiment {}'.format(sent_num)] = sentiment
                    print("Finished sentiment analysis for rows " + str(df.index.values))

                # If the AI returns malformed data that is unusable, catch the error and move on
                except KeyboardInterrupt:
                    print("KeyboardInterrupt recieved, aborting process")
                    sys.exit()
                except:
                    e = sys.exc_info()
                    if iter < tries - 1:
                        print("\n{} exception in sentiment analysis for dataframe containing rows \n{}\n".format(e[0], str(df.index.values)))
                        print("Retrying {} out of {} tries\n".format(iter+1, tries))
                        continue
                    else:
                        print("\n{} exception in sentiment could not be resolved\n".format(e[0]))
                        break
                break

            
            iter = 0
            for iter in range(tries):
                try:    
                    # Write the prompt for the themes analysis function, using the entire csv file at once
                    # Send the prompt to the chat bot and record the response to the 'themes' list
                    prompt = """Give the single most common theme in one word for each of the {} sentences in the list below.
                                Sentences are delimited by new lines. Do not index your answers.
                                You should output exactly {} sets of themes. Do not output more or less than {} rows.
                                Follow your instructions exactly: 
                                \n""".format(df_init_len, df_init_len, df_init_len) + df_init.to_csv(index=False, header=False)
                    response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "user", "content": prompt},
                            ],
                            temperature=0.5,
                        )
                    themes = [response['choices'][0]['message']['content']]


                    # Reformat the data retrieved from the themes prompt into a format suitable for the dataframe
                    split = []
                    for t in themes:
                        split += [t.split('\n')]
                    theme = []
                    for s in split:
                        for t in s:
                            t = t.split(',')
                            t = [s.strip(' ') for s in t]
                            t = [x.lower() for x in t]
                            if t[0][0].isnumeric():
                                t[0] = t[0][3:]
                            ts = ['none']
                            if len(t) > 3:
                                break
                            for i in range(len(t)):
                                ts[i] = t[i]
                            theme += [ts[0]]


                    # Save the responses to the pandas dataframe
                    while 'Theme {}'.format(them_num) in df.columns:
                        them_num += 1
                    df['Theme {}'.format(them_num)] = theme
                    them_num += 1
                    print("Finished theme analysis for rows " + str(df.index.values))


                # If the AI returns malformed data that is unusable, catch the error and move on
                except KeyboardInterrupt:
                    print("KeyboardInterrupt recieved, aborting process")
                    sys.exit()
                except:
                    e = sys.exc_info()
                    if iter < tries - 1:
                        print("\n{} exception in theme analysis for dataframe containing rows \n{}\n".format(e[0], str(df.index.values)))
                        print("Retrying {} out of {} tries\n".format(iter+1, tries))
                        continue
                    else:
                        print("\n{} exception in theme analysis could not be resolved\n".format(e[0]))
                        break
                break
        dfs_out.append(df)


    # Merge the outputted dataframes back togther into a single dataframe
    df_out = pd.DataFrame()
    while len(dfs_out) != 0:
        df_out = pd.concat([df_out, dfs_out.pop()])
    df_out = df_out.sort_index()
    head = [list(df_out.columns)[0]]
    headers = list(df_out.columns)[1:]
    headers.sort()
    headers = head + headers
    df_out = df_out[headers]
    print("Merged all output dataframes back into one dataframe")

    iter = 0
    for iter in range(tries):
        try:
            # Write the prompt for the summary function, using the entire csv file at once
            # Send the prompt to the chat bot and record the response to the 'summary' list
            prompt = """Write a paragraph that summarizes the data in the following csv file. 
                        Specifically, you should answer with respect to the original question, {} 
                        Make certain to address both the themes of the responses and the overall sentiment.
                        """.format(df_out.columns[0]) + df_out.to_csv(index=False)
            response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.5,
                )
            summary = [response['choices'][0]['message']['content']]


            # Save the responses to the pandas dataframe 
            df_out['Summary'] = pd.Series(summary)
            print("Finished summary analysis on dataframe")
        except KeyboardInterrupt:
            print("KeyboardInterrupt recieved, aborting process")
            sys.exit()
        except:
            e = sys.exc_info()
            if iter < tries - 1:
                print("\n{} exception in summary for dataframe containing \n{}\n".format(e[0], str(df)))
                print("Retrying {} out of {} tries\n".format(iter+1, tries))
                continue
            else:
                print("\n{} exception in summary could not be resolved\n".format(e[0]))
                break
        break



    # Find the percentage of responses that share a sentiment value and add those percentage to the dataframe
    sum_sent = []
    df_sent = pd.DataFrame()
    for head in headers:
        if head[:-2] == 'Sentiment':
            for h in df_out[head]:
                sum_sent.append(h)
    v, c = np.unique(sum_sent, return_counts=True)
    df_sent['Sentiment Values'], df_sent['Sentiment Totals'], df_sent['Sentiment Percentages']  = pd.Series(v), pd.Series(c), pd.Series((c/len(sum_sent))*100)
    df_out = pd.concat([df_out, df_sent], axis=1)
    print("Statistical analysis of sentiment analysis responses completed")


    # Find the percentage of responses that share a theme value and add those percentage to the dataframe
    sum_them = []
    df_them = pd.DataFrame()
    for head in headers:
        if head[:-2] == 'Theme':
            for h in df_out[head]:
                sum_them.append(h)
    v, c = np.unique(sum_them, return_counts=True)
    df_them['Theme Values'], df_them['Theme Totals'], df_them['Theme Percentages']  = pd.Series(v), pd.Series(c), pd.Series((c/len(sum_them))*100)
    df_out = pd.concat([df_out, df_them], axis=1)
    print("Statistical analysis of theme analysis responses completed")


    # Save the dataframe as a csv file in a subfolder of the files out folder with the same name as the input file
    while os.path.exists('files_out\\' + name[9:]):
        file_num += 1
    file_out = 'files_out\\' + name[9:]
    df_out.iloc[:, :1] = f.iloc[:, :1]
    df_out.to_csv(file_out)
    print("File finished, output saved to " + file_out + "\n")
print("Run completed")