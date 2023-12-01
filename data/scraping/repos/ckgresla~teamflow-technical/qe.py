# Utils to Power the Question Endpoint

import os
from openai_requests import *




# Get Statistics from a Transcript (n people + n lines per person for a single file/transcript str)
def transcript_stats(script: str=None):
    """
    Function to get a list of lines spoken and speaker stats for a given;
      - Transcript String
      - Transcript Filepath
    """
    # Case When script is path on disk
    if os.path.exists(script):
        file = open(script)
        data = file.read()
        data = data.split("\n")
    # Else if not a path, assume passed string *is* the script
    else:
        data = script
    
    # Remove LITERAL duplicate strs + get number of speakers and info
    data = data.split("\n")
    data = set(data)
    if "" in data:
        data.remove("") #remove the empty line chars
    stats = {}

    for seq in data:
        speaker = seq.split(":")[0] #assuming that real data follows sample conventions
        if speaker not in stats:
            stats[speaker] = 0
        else: 
            stats[speaker] += 1

    stats["n_people"] = len(stats) #number of folks in transcription str
    return data, stats


# Get Unique Lines in Transcript
def transcript_lines(script: str):
    # Case When script is path on disk
    if os.path.exists(script):
        file = open(script)
        data = file.read()
        data = data.split("\n")
    # Else if not a path, assume passed string *is* the script
    else:
        data = script.split("\n") #assume string for transcript follows sample data format
    
    # Drop empty lines
    data = [i for i in data if i != ""]
    return data


# Get all lines said by clients -- pass these for question checking
def parse_client_lines(transcript_lines: list):
    speakers = {}
    for line in transcript_lines:
        speaker, context = line.split(":", 1)
        if "client" in speaker.lower():
            if speaker not in speakers:
                speakers[speaker] = []
                speakers[speaker].append(context)
            else: 
                speakers[speaker].append(context)

    return speakers


# Get a Dict of Clients and their respective & relevant questions 
def parse_client_questions(client_trancsripts: dict):
    qcs = {client: [] for client in client_trancsripts}
    for client in client_trancsripts:
        for line in client_trancsripts[client]:
            result = get_questions_gpt(line)
            if result != "NO_QUESTIONS": 
                result = result.replace("NO_QUESTIONS", "")
                results = result.split("\n")
                for res in results:
                    if res == "" or "###" in res or "input_text" in res.lower():
                        pass
                    else:
                        res = res.strip()
                        qcs[client].append(res)
    return qcs