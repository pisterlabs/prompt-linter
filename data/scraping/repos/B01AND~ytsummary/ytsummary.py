# In this program we will read the transcript of a youtube video and summarize it
# Where the transcript is longer than 10 minutes, we will split it into 10 minute chunks

import setcreds
import openai
import sys
from youtube_transcript_api import YouTubeTranscriptApi

diagnostics = 0
include_mentions = 0

def get_video_id_from_video_id_or_url(video_id_or_url):
    # a youtube video id is 11 characters long
    # if the video id is longer than that, then it's a url
    if len(video_id_or_url) > 11:
        # it's a url
        # the video id is the last 11 characters
        return video_id_or_url[-11:]
    else:
        # it's a video id
        return video_id_or_url

def get_chunks_from_youtube(video_id):
    # this function will get the transcript of a youtube video
    # and return it as an array of chunks
    # where each chunk is an array of lines

    # first get the transcript
    transcript = YouTubeTranscriptApi.get_transcript(video_id)

    # now break the transcript into 10 minute chunks
    # The transcript includes timestamps, so we can use that to break it up.
    # An example from the transcript:
    # [
    #     {
    #         'text': 'Hey there',
    #         'start': 7.58,
    #         'duration': 6.13
    #     },
    #     {
    #         'text': 'how are you',
    #         'start': 14.08,
    #         'duration': 7.58
    #     },
    #     # ...
    # ]

    chunks = []

    start_timestamp = 0.0
    current_timestamp_mins = 0.0

    current_chunk = []

    for entry in transcript:
        current_timestamp_mins = entry['start'] / 60.0

        # if the current timestamp is more than 10 minutes after the start timestamp
        # then we have a chunk
        if current_timestamp_mins - start_timestamp > 10:
            # add the current chunk to the list of chunks
            chunks.append(current_chunk)
            # reset the start timestamp
            start_timestamp = current_timestamp_mins
            # reset the current chunk
            current_chunk = []

        # add the line to the current chunk
        current_chunk.append(entry['text'])

    # add the last chunk
    if len(current_chunk) > 0:
        chunks.append(current_chunk)

    print(f"Found {len(chunks)} chunks")

    return chunks

def summarize_chunk(index, chunk):
    chunk_str = "\n".join(chunk)
    prompt = f"""The following is a section of the transcript of a youtube video. It is section #{index+1}:

    {chunk_str}

    Summarize this section of the transcript."""

    if diagnostics:
        # print each line of the prompt with a leading # so we can see it in the output
        for line in prompt.split('\n'):
            print(f"# {line}")

    completion = openai.Completion.create(
        engine="text-davinci-003", 
        max_tokens=500, 
        temperature=0.2,
        prompt=prompt,
        frequency_penalty=0
    )

    msg = completion.choices[0].text

    if diagnostics:
        print(f"# Response: {msg}")

    return msg

def summarize_the_summaries(summaries):

    summaries_str = ""
    for index, summary in enumerate(summaries):
        summaries_str += f"Summary of chunk {index+1}:\n{summary}\n\n"

    prompt = f"""The following are summaries of a youtube video in 10 minute chunks:"

    {summaries_str}

    Summarize the summaries."""

    if diagnostics:
        # print each line of the prompt with a leading # so we can see it in the output
        for line in prompt.split('\n'):
            print(f"# {line}")

    completion = openai.Completion.create(
        engine="text-davinci-003", 
        max_tokens=500, 
        temperature=0.2,
        prompt=prompt,
        frequency_penalty=0
    )

    msg = completion.choices[0].text

    if diagnostics:
        print(f"# Response: {msg}")

    return msg

def people_and_entities_mentioned_in_chunk(index, chunk):
    # this function will return a list of people who appear to be speaking in the chunk,
    # and a list of people mentioned in the chunk, but not speaking.

    # first get the people speaking:
    chunk_str = "\n".join(chunk)
    prompt = f"""The following is a section of the transcript of a youtube video. It is section #{index+1}:

    {chunk_str}

    Who is mentioned in this section of the transcript? Include people, companies or organizations, and other non-human entities."""

    if diagnostics:
        # print each line of the prompt with a leading # so we can see it in the output
        for line in prompt.split('\n'):
            print(f"# {line}")

    completion = openai.Completion.create(
        engine="text-davinci-003", 
        max_tokens=500, 
        temperature=0.2,
        prompt=prompt,
        frequency_penalty=0
    )

    mentioned = completion.choices[0].text

    if diagnostics:
        print(f"# Response: {mentioned}")

    # now we have the people speaking and the people mentioned
    # we can return them
    return mentioned
    
def summarize_mentions(mentioneds):
    mentioneds_str = ""
    for index, mentioned in enumerate(mentioneds):
        mentioneds_str += f"People mentioned in chunk {index+1}:\n{mentioned}\n\n"

    prompt = f"""The following are the people and other entities mentioned in a youtube video in 10 minute chunks:"

    {mentioneds_str}

    Summarize the people and entitites mentioned."""

    if diagnostics:
        # print each line of the prompt with a leading # so we can see it in the output
        for line in prompt.split('\n'):
            print(f"# {line}")

    completion = openai.Completion.create(
        engine="text-davinci-003", 
        max_tokens=500, 
        temperature=0.2,
        prompt=prompt,
        frequency_penalty=0
    )

    msg = completion.choices[0].text

    if diagnostics:
        print(f"# Response: {msg}")

    return msg


def main():
    # Get the transcript of the video
    if len(sys.argv) < 2:
        print("Usage: python3 sumvid.py <video id or url>")
        sys.exit(1)

    # transcript_file_name = sys.argv[1]
    video_id_or_url = sys.argv[1]

    # if the video id or url is a url, extract the video id
    video_id = get_video_id_from_video_id_or_url(video_id_or_url)

    if len(sys.argv) > 2:
        for arg in sys.argv[2:]:
            if arg == "--diagnostics":
                global diagnostics
                diagnostics = True

            if arg == "--mentions":
                global include_mentions
                include_mentions = True

    # chunks = get_chunks(transcript_file_name)
    chunks = get_chunks_from_youtube(video_id)

    if len(chunks) == 0:
        print("No chunks found")
    elif len(chunks) == 1:
        summary = summarize_chunk(0, chunks[0])
        print(f"\nSummary: {summary}")

        if include_mentions:
            mentioned = people_and_entities_mentioned_in_chunk(0, chunks[0])
            print(f"\nPeople mentioned: {mentioned}")
    else:
        # Now we have the chunks, we can summarize each one
        summaries = []
        mentioneds = []
        for index, chunk in enumerate(chunks):
            summary = summarize_chunk(index, chunk)
            summaries.append(summary)
            print(f"\nSummary of chunk {index+1}: {summary}")

            if include_mentions:
                mentioned = people_and_entities_mentioned_in_chunk(index, chunk)
                mentioneds.append(mentioned)
                print(f"\nPeople mentioned in chunk {index+1}: {mentioned}")

        # Now we have the summaries, we can summarize the summaries
        summary_of_summaries = summarize_the_summaries(summaries)

        print(f"\nSummary of summaries: {summary_of_summaries}")

        if include_mentions:
            # Now we have the people speaking and mentioned, we can summarize them
            # summary_of_people = summarize_the_people(speakings, mentioneds)
            summary_of_people = summarize_mentions(mentioneds)

            print(f"\nSummary of people: {summary_of_people}")

if __name__ == "__main__":
    main()
