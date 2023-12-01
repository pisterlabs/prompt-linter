from py_youtube import Data
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
import boto3
import json
from shared import exc_to_string
import openai
import os

openai.api_key = os.environ['openai_api_key']

def is_youtube_url(url):
    # is this an actual url?
    try:
        result = urlparse(url)
        is_real_url = all([result.scheme, result.netloc])
        # deal with youtube.com and youtu.be
        return is_real_url and (result.netloc.endswith('youtube.com') or result.netloc.endswith('youtu.be'))
    except:
        return False
    
def get_youtube_id(url):
    if not is_youtube_url(url):
        return None

    # youtube.com format is https://www.youtube.com/watch?v=HhHzCfrqsoE
    # youtu.be format is https://youtu.be/HhHzCfrqsoE

    # let's parse the url properly
        
    youtube_url = urlparse(url)

    if youtube_url.netloc == 'youtu.be':
        youtube_id = youtube_url.path[1:]
    else:
        # youtube.com
        # get the query string parameters as a dictionary
        query_params = parse_qs(youtube_url.query)
        youtube_id = query_params['v'][0]

    return youtube_id

def get_youtube_data(url):
    '''
    format returned:
    {
        'id': 'HhHzCfrqsoE',
        'title': 'How To Create MongoDB Database  Url', 
        'thumbnails': 'https://i.ytimg.com/vi/HhHzCfrqsoE/hqdefault.jpg?sqp=-oaymwEiCKgBEF5IWvKriqkDFQgBFQAAAAAYASUAAMhCPQCAokN4AQ==\\u0026rs=AOn4CLBOkJZAdEpYxQOVdaUxFHdbThH_DQ',  
        'views': '709', 
        'likes': '27', 
        'dislikes': '1', 
        'publishdate': '2021-08-04', 
        'category': 'Howto \\u0026 Style', 
        'channel_name': 'Ln Technical', 
        'subscriber': '1.67K', 
        'keywords': 'video, sharing, camera phone, video phone, free, upload'
    }
    '''

    if not is_youtube_url(url):
        return None
    
    data = Data(url).data()

    return data

def get_youtube_transcript(youtube_url):
    '''
    This function returns the transcript of a youtube video.
    '''

    if not is_youtube_url(youtube_url):
        return None

    youtube_id = get_youtube_id(youtube_url)

    languages = ('en', 'en-US', 'en-GB', 'en-CA', 'en-AU', 'en-NZ')
    transcript = YouTubeTranscriptApi.get_transcript(youtube_id, languages=languages)

    return transcript


def get_youtube_transcript_url(spelunk_rec, s3_bucket):
    ''' 
    This function checks whether the youtube transcript is already created and written to s3.
    If not, it creates the transcript and writes it to s3.
    Then it creates a signed url to the transcript and returns it.
    '''
    print ('get_youtube_transcript')

    spelunk_id = spelunk_rec['id']
    s3_key = f'transcripts/{spelunk_id}.json'

    youtube_url = ((spelunk_rec.get('lenses') or {}).get('source') or {}).get('url')

    if not youtube_url:
        print ('no youtube url')
        return None

    s3 = boto3.client('s3')

    object_exists = False

    try:
        s3.head_object(
            Bucket=s3_bucket,
            Key=s3_key
        )
        object_exists = True
    except:
        pass

    if not object_exists:
        # create the transcript and write to s3
        try:
            transcript = get_youtube_transcript(youtube_url)

            transcript_str = json.dumps(transcript)

            s3.put_object(
                Bucket=s3_bucket,
                Key=s3_key,
                Body=transcript_str
            )
        except Exception as e:
            print ('failed to create transcript: ' + exc_to_string(e))
            raise
    
    # create a signed url to the transcript
    try:
        signed_url = s3.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': s3_bucket,
                'Key': s3_key
            },
            ExpiresIn=3600
        )

        return signed_url
    except Exception as e:
        print ('failed to create signed url: ' + exc_to_string(e))
        raise


def get_chunks_from_transcript(transcript, chunk_length_mins=10.0):
    # this function converts a transcript of a video
    # into an array of chunks
    # where each chunk is an array of lines

    # An example of a transcript:
    # [
    #     {
    #         'speaker': 'speaker1',
    #         'text': 'Hey there',
    #         'start': 7.58,
    #         'duration': 6.13
    #     },
    #     {
    #         'speaker': 'speaker1',
    #         'text': 'how are you',
    #         'start': 14.08,
    #         'duration': 7.58
    #     },
    #     # ...
    # ]
    # 
    # The duration is optional.
    # The speaker is optional.
    # The start could also be a string like this: "00:00:24.410"
    # start might also be called start_time

    def add_new_chunk(chunks, new_chunk):
        if len(new_chunk) > 0:
            # get the previous chunk
            previous_chunk = None
            if len(chunks) > 0:
                previous_chunk = chunks[-1]

            # Add up to 3 lines from the previous chunk to the new chunk
            if previous_chunk is not None:
                # get the last 3 lines from the previous chunk
                last_lines = previous_chunk[-3:]
                # add them to the new chunk
                new_chunk = last_lines + new_chunk

            # Add up to 3 lines from the new chunk to the previous chunk
            if previous_chunk is not None:
                # get the first 3 lines from the new chunk
                first_lines = new_chunk[:3]
                # add them to the previous chunk
                previous_chunk = previous_chunk + first_lines

            # add the new chunk to the list of chunks
            chunks.append(new_chunk)

        return chunks

    chunks = []

    start_timestamp = 0.0
    current_timestamp_mins = 0.0

    current_chunk = []

    for entry in transcript:
        start = entry.get('start') or entry.get('start_time')

        # if the start is a string, convert it to a float
        if isinstance(start, str):
            # parse the string into a duration
            # the string is in the format "00:00:24.410"
            # where the first two numbers are the hours
            # the next two numbers are the minutes
            # the next two numbers are the seconds
            # the last three numbers are the milliseconds

            # split the string into hours, minutes, seconds, and milliseconds
            hours, minutes, seconds_and_milliseconds = start.split(":")
            seconds, milliseconds = seconds_and_milliseconds.split(".")
            # convert the strings to numbers
            hours = int(hours)
            minutes = int(minutes)
            seconds = int(seconds)
            milliseconds = int(milliseconds)
            # convert the duration to seconds
            start = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0

        try:
            current_timestamp_mins = start / 60.0
        except:
            pass # just use the previous timestamp

        # if the current timestamp is more than chunk_length_mins minutes after the start timestamp
        # then we have a chunk
        if current_timestamp_mins - start_timestamp > chunk_length_mins:
            # add the current chunk to the list of chunks
            chunks = add_new_chunk(chunks, current_chunk)
            # reset the start timestamp
            start_timestamp = current_timestamp_mins
            # reset the current chunk
            current_chunk = []

        # if we have a speaker, then the line should be <speaker>: <text>
        # otherwise, it's just <text>
        if 'speaker' in entry:
            line = f"{entry['speaker']}: {entry['text']}"
        else:
            line = entry['text']

        # add the line to the current chunk
        current_chunk.append(line)

    # add the last chunk
    if len(current_chunk) > 0:
        chunks.append(current_chunk)

    print(f"Found {len(chunks)} chunks")

    return chunks

def summarize_chunk(index, chunk, prompt_header, diagnostics=False):
    chunk_str = "\n".join(chunk)
    prompt = f"""The following is a section of the transcript of a youtube video. It is section #{index+1}:

{chunk_str}

{prompt_header}Summarize this section of the transcript. Don't mention that this is a section in the summary."""

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

def summarize_the_summaries(summaries, prompt_header, diagnostics=False):

    summaries_str = ""
    for index, summary in enumerate(summaries):
        summaries_str += f"Summary of chunk {index+1}:\n{summary}\n\n"

    prompt = f"""The following are summaries of a youtube video in 10 minute chunks:"

{summaries_str}

{prompt_header}Summarize the summaries."""

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

def summarize_audio_transcript_chunks(chunks, prompt_header, chunk_len_mins):
    # we're going to yield a summary of each chunk, and a summary of the summaries
    result = ""

    if len(chunks) == 0:
        output = "No chunks found"
        print(output)
        yield output
    elif len(chunks) == 1:
        summary = summarize_chunk(0, chunks[0], prompt_header)
        output = f"Summary: {summary}"
        print(output)
        yield output
    else:
        # Now we have the chunks, we can summarize each one
        summaries = []
        for index, chunk in enumerate(chunks):
            summary = summarize_chunk(index, chunk, prompt_header)
            summaries.append(summary)

            chunk_start_time_mins = index * chunk_len_mins

            display_chunk_start_time_h_m_s = f"{chunk_start_time_mins // 60}:{chunk_start_time_mins % 60:02d}:00"

            output = {
                "heading": f"Summary of section beginning at {display_chunk_start_time_h_m_s}",
                "summary": summary.strip()
            }

            yield output

        # Now we have the summaries, we can summarize the summaries
        summary_of_summaries = summarize_the_summaries(summaries, prompt_header)

        output = {
            "heading": "Overall summary",
            "summary": summary_of_summaries.strip()
        }
        
        yield output


def get_youtube_summary_url(spelunk_rec, s3_bucket):
    ''' 
    This function checks whether the youtube summary is already created and written to s3.
    If not, it creates the summary and writes it to s3.
    Then it creates a signed url to the summary and returns it.
    '''
    print ('get_youtube_summary_url')

    spelunk_id = spelunk_rec['id']
    s3_key = f'youtube_summaries/{spelunk_id}.json'

    youtube_url = ((spelunk_rec.get('lenses') or {}).get('source') or {}).get('url')

    if not youtube_url:
        print ('no youtube url')
        return None

    s3 = boto3.client('s3')

    object_exists = False

    try:
        s3.head_object(
            Bucket=s3_bucket,
            Key=s3_key
        )
        object_exists = True
    except:
        pass

    if not object_exists:
        # create the summary and write it to s3
        try:
            # first get the transcript
            transcript = get_youtube_transcript(youtube_url)

            # now summarize the transcript
            chunk_len = 10
            chunks = get_chunks_from_transcript(transcript, chunk_len)

            if len(chunks) == 0:
                print ('no chunks found')
                return None
            elif len(chunks) == 1:
                chunk_len = 2
                chunks = get_chunks_from_transcript(transcript, chunk_len)
            elif len(chunks) == 2:
                chunk_len = 5
                chunks = get_chunks_from_transcript(transcript, chunk_len)
            else:
                while len(chunks) > 15:
                    chunk_len *= 2
                    chunks = get_chunks_from_transcript(transcript, chunk_len)

            # summarize_audio_transcript_chunks yields chunks. Get each one and write it.
            summaries = []
            for summary_chunk in summarize_audio_transcript_chunks(chunks, "", chunk_len):
                print (summary_chunk)
                summaries.append(summary_chunk)

            summary_doc = json.dumps(summaries)

            s3.put_object(
                Bucket=s3_bucket,
                Key=s3_key,
                Body=summary_doc
            )
        except Exception as e:
            print ('failed to create summary: ' + exc_to_string(e))
            raise
    
    # create a signed url to the summary
    try:
        signed_url = s3.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': s3_bucket,
                'Key': s3_key
            },
            ExpiresIn=3600
        )

        return signed_url
    except Exception as e:
        print ('failed to create signed url: ' + exc_to_string(e))
        raise
