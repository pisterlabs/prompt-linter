import asyncio
from openai import AsyncOpenAI
import time
import cv2
import base64
import json
#from moviepy.editor import VideoFileClip
import os
import sys

# api_key = "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"
api_key = "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" # wjxia
client = AsyncOpenAI(api_key=api_key)

DURATION = 30
SELECTED_FPS = 1
SYSTEM_PROMPT = """
You are a public speaking coach. You are coaching a client who will be giving a presentation, but don't mention who you are in your response. 
Try to not summarize or describe the presentation but instead give feedback about possible improvements.
You will be given a transcript and frames from the video of the presentation.
Do not comment on enunciation, pronunciation, or audio tone because that will not be available to you.
Your output should be formatted as a paragraph and DO NOT include any extra information that is not in the transcript or video.
Pretend you are watching the presentation live and give feedback as if you were there.
For example, do not respond with phrases like "from the frames provided", "from the transcript provided", etc.
"""
USER_PROMPT = """
{context} Please critique and give feedback this short excerpt of my presentation.
Try to focus on the content and accuracy of the presentation and give suggests on how I can improve.
Please be concise and keep your output to 100 words or less. 
The transcript may be cutoff in the middle of a sentence, so don't worry about unfinished sentences.
Here is the transcript and video.

Transcript: \"""
{selected_transcript}
\"""
"""


def convert_video_to_audio_moviepy(video_file, output_ext="mp3"):
    filename, _ = os.path.splitext(video_file)
    clip = VideoFileClip(video_file)
    clip.audio.write_audiofile(f"{filename}.{output_ext}")

def split_transcripts(transcript) -> list[str]:
    splits = []
    start_time = 0

    while start_time < transcript.duration:
        if transcript.duration - start_time < 10:
            # print("Video has is less than 10 seconds left, skipping last split.")
            break
        
        segments = transcript.segments
        selected = ""
        for seg in segments:
            if seg["start"] <= start_time + DURATION and seg["end"] >= start_time:
                selected += seg["text"]
        splits.append(selected)
        start_time += DURATION

    return splits


def split_frames(frames: list[bytes], fps: int) -> list[list[bytes]]:
    splits = []
    start_time = 0

    while start_time < len(frames) / fps:
        selected = frames[
            start_time * fps : (start_time + DURATION) * fps : fps // SELECTED_FPS
        ]
        splits.append(selected)
        start_time += DURATION

    return splits


async def get_transcript(audio_link: str):
    # print(f"generating transcript at {time.strftime('%X')}")
    filler_words = "Umm, let me think like, hmm... Okay, here's what I'm, like, thinking."
    audio_file = open(audio_link, "rb")

    async def get_filler_words_transcript():
        return await client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
            prompt=filler_words,
        )

    async def get_clean_transcript():
        return await client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
        )

    # Run both requests in parallel
    filler_words_transcript, clean_transcript = await asyncio.gather(
        get_filler_words_transcript(),
        get_clean_transcript()
    )

    if not filler_words_transcript.text:
        raise Exception("transcript generation failed")

    # print(f"finished transcript at {time.strftime('%X')}")

    return filler_words_transcript, clean_transcript


def parse_video(video_link: str) -> tuple[list[bytes], int]:
    # print(f"started parsing video {time.strftime('%X')}")

    video = cv2.VideoCapture(video_link)

    base64Frames = []
    fps = int(video.get(cv2.CAP_PROP_FPS))

    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()

    # print(f"finished parsing video {time.strftime('%X')}")

    return base64Frames, fps


async def get_vision_completion(
    frames: list[bytes], context: str, transcript: str
) -> str:
    data_uris = [f"data:image/jpeg;base64,{f}" for f in frames]

    image_dicts = [
        {"type": "image_url", "image_url": {"url": data_uri, "detail": "low"}}
        for data_uri in data_uris
    ]

    prompt_messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                USER_PROMPT.format(context=context, selected_transcript=transcript),
                *image_dicts,
            ],
        },
    ]

    params = {
        "model": "gpt-4-vision-preview",
        "messages": prompt_messages,
        "max_tokens": 200,
        "temperature": 1,
    }

    result = await client.chat.completions.create(**params)
    return result.choices[0].message.content


async def get_annotations(video_link: str, transcript, context: str) -> list[str]:
    # TODO: figure out how audio is going to be sent
    transcript_list = split_transcripts(transcript)

    video_frames, fps = parse_video(video_link)

    frame_list = split_frames(video_frames, fps)

    # if len(frame_list) != len(transcript_list):
        # print("Warning: number of frames and transcripts do not match")
        # print(f"Number of frames: {len(frame_list)}")
        # print(f"Number of transcripts: {len(transcript_list)}")

    data = zip(frame_list, transcript_list)

    # print(f"started generating annotations {time.strftime('%X')}")

    tasks = [
        get_vision_completion(frames, context, transcript)
        for frames, transcript in data
    ]

    result = await asyncio.gather(*tasks)

    # print(f"finished generating annotations {time.strftime('%X')}")

    return result, video_frames[0]


FILTER_PROMPT_1 = """
Can you condense each paragraph down? I want something clear, straightforward, and in a conversational tone, but try to keep as much information as possible.

Please use this format for your output and keep each paragraph separate:
\"""
{{paragraph 1}}
{{paragraph 2}}
{{paragraph 3}}
continue...
\"""

Here are the paragraphs:
\"""
{annotations}
\"""
"""

FILTER_PROMPT_2 = """
Can you edit each of these paragraphs by removing any duplicate information or repetitive language between them?
If two paragraphs reuse the same term or phrase, include it only once.
Also remove any overall or general feedback.
DO NOT add any extra information that is not in the feedback.
For example, these two sentences are very similar and should be removed:
\"""
"Your use of a visual aid to illustrate the anatomy of the brain is effective"
"Your use of a physical model to explain the different parts of the brain is an excellent visual aid"
\"""

Please use this format for your output and keep each paragraph separate:
\"""
{{paragraph 1}}
{{paragraph 2}}
{{paragraph 3}}
continue...
\"""

Here are the paragraphs:
\"""
{annotations}
\"""
"""


async def filter_annotations(annotations: list[str]) -> list[str]:
    # print(f"started filtering annotations {time.strftime('%X')}")

    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    messages.append(
        {
            "role": "user",
            "content": FILTER_PROMPT_1.format(annotations="\n\n".join(annotations)),
        }
    )

    first_filter_result = await client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
    )

    first_filter_result = clean_output(first_filter_result.choices[0].message.content)
    # print(len(first_filter_result.split("\n")))
    # first_filter_result = "\n\n".join(annotations)

    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    messages.append(
        {
            "role": "user",
            "content": FILTER_PROMPT_2.format(annotations=first_filter_result),
        }
    )

    second_filter_result = await client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
    )

    output = clean_list(clean_output(second_filter_result.choices[0].message.content).split("\n"))
    # print(len(output))

    # print(f"finished filtering annotations {time.strftime('%X')}")

    return output


def clean_output(text: str) -> str:
    text =  text.replace('"""', "")
    text = text.replace('```', "")
    text = text.replace("\n\n", "\n")
    return text

def clean_list(text: list[str]) -> list[str]:
    return list(filter(lambda x: x != "", text))


SUMMARY_PROMPT = """
I just did a presentation and received a lot of feedback! Can you summarize and rewrite this feedback as if it were your own? Try to use the following categories in your feedback.

Content, Reasoning, Organization, Style, Mechanics

You can also add extra feedback either based on the summary or by looking through the transcript of my presentation. Don't mention the transcript specifically in the summary. 
Use the transcript I provided to see if I used a lot of filler words. Please give feedback if I did and list what filler words I used.
Do not comment on enunciation, pronunciation, or audio tone because that will not be available to you.
Be as concise as possible and do not add any extra information that is not in the feedback or transcript.
Begin your feedback with \"""Feedback on Presentation\"""
Please use a conversational but professional tone.

Here are some metrics to use (ignore non-applicable bullet points):
Content
-Accuracy and originality of facts and evidence presented (both orally and visually)
-Adequacy and persuasiveness of presentation relative to topics covered
-Use of appropriate range and quantity of sources, clear identification of sources
Reasoning
-Clarity and memorability of key points
-Connections between facts and theories, critical evaluation of evidence
-Separation of facts from opinions, consideration of alternative viewpoints
Organization
-Orderliness, clear citation of sources
-Purposefulness, clear identification of topics to be addressed
-Smoothness of flow
Style
-Engagement and vigor (holding audience's attention)
-Facilitation of discussion (posing of questions to audience)
-Responsiveness to audience's questions
-Spontaneity (sparing use of notes, with no reading aloud)
Mechanics
-Eye contact with entire audience, facial expressiveness
-Fluency (complete sentences, with no filled pauses (uh, like, well, okay?)
-Hand and arm gestures, body movement, with no fidgeting
-Use of visual aids (chalkboard, computer graphics, etc.)

transcript:
\"""
{transcript}
\"""

annotations:
\"""
{annotations}
\"""
"""

SCORE_PROMPT = """
Could you also give me a score out of 100 for each of the five categories (Content, Reasoning, Organization, Style, Mechanics)? Please use the same metrics as before. 

Only output the scores and format as a json object.
"""


async def generate_summary(annotations: list[str], transcript) -> str:
    # print(f"started generating summary {time.strftime('%X')}")

    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    messages.append(
        {
            "role": "user",
            "content": SUMMARY_PROMPT.format(
                transcript=transcript.text, annotations="\n\n".join(annotations)
            ),
        }
    )

    summary = await client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
        temperature=0,
    )

    summary = summary.choices[0].message.content
    summary = summary.split("\n")
    summary = "\n".join(summary[1:])

    messages.append({"role": "assistant", "content": summary})
    messages.append(
        {
            "role": "user",
            "content": SCORE_PROMPT,
        }
    )

    score = await client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
    )

    score = score.choices[0].message.content
    score = score.replace("```json\n", "").replace("\n```", "")
    score = json.loads(score)

    filler_words = ["um", "uh", "er", "well", "okay", "you know", "basically", "i mean"]

    filler_words_count = sum(
        [
            transcript.text.lower().count(word)
            for word in filler_words
        ]
    )

    score["filler_words_count"] = filler_words_count
    score["filler_words_ratio"] = filler_words_count / len(transcript.text.split(" "))

    # print(f"finished generating summary {time.strftime('%X')}")

    return summary, score


def generate_json(
    filtered_annotations: list[str], transcript, summary: str, score: dict
):
    timestamps = {}
    annotation_index = 0
    time = 0
    for seg in transcript.segments:
        if seg["start"] >= time + DURATION:
            annotation_index += 1
            annotation_index = min(annotation_index, len(filtered_annotations) - 1)
            time += DURATION
        timestamps[f"{round(seg['start'], 2)}-{round(seg['end'], 2)}"] = {
            "feedback": filtered_annotations[annotation_index],
            "transcript": seg['text'],
        }

    # for line in filtered_annotations:
    #     selected_transcript = ""

    #     for seg in transcript.segments:
    #         if seg["start"] >= start_time and start_time + DURATION >= seg["start"]:
    #             selected_transcript += seg["text"]

    #     timestamps[f"{start_time}-{start_time + DURATION}"] = {
    #         "feedback": line,
    #         "transcript": selected_transcript,
    #     }

    #     start_time += DURATION

    output_dict = {}
    output_dict["timestamps"] = timestamps
    output_dict["summary"] = summary
    output_dict["score"] = score

    return output_dict

async def main() -> None:
    if len(sys.argv) < 4:
        context = ""
    else:
        context = sys.argv[3]

    audio_link = sys.argv[2]
    video_link = sys.argv[1]

    filler_words_transcript, clean_transcript  = await get_transcript(audio_link)
    annotations, first_frame = await get_annotations(video_link, clean_transcript, context)

    filtered_annotations = await filter_annotations(annotations)
    summary, score = await generate_summary(filtered_annotations, filler_words_transcript)

    output = generate_json(
            filtered_annotations,
            clean_transcript,
            summary,
            score)
    output["image"] = first_frame

    print(json.dumps(output))
    # await time.sleep(10)
    # print(json.dumps({'timestamps': {'0.0-7.28': {'feedback': 'Delve into specific functions of brain components, incorporate engaging questions or facts.', 'transcript': ' So first, we start off with the spinal cord, which leads up toward the brain.'}, '7.28-15.32': {'feedback': 'Delve into specific functions of brain components, incorporate engaging questions or facts.', 'transcript': ' As it goes closer to the brain, it expands and swells and becomes the brain stem.'}, '15.32-25.44': {'feedback': 'Delve into specific functions of brain components, incorporate engaging questions or facts.', 'transcript': ' At the top of the brain stem sits the thalamus, and right under the thalamus lies the hypothalamus,'}, '25.44-43.88': {'feedback': 'Delve into specific functions of brain components, incorporate engaging questions or facts.', 'transcript': ' hypo meaning under or lower, and under the hypothalamus is the pituitary gland.'}, '43.88-53.24': {'feedback': 'Correct hippocampus and amygdala locations in your visual aid, and highlight each part when mentioned.', 'transcript': ' Going around from both sides of the thalamus is the hippocampus that extends out to the'}, '53.24-62.4': {'feedback': 'Correct hippocampus and amygdala locations in your visual aid, and highlight each part when mentioned.', 'transcript': ' front of the brain, and at the front ends of the brain is the amygdala, and on the other'}, '62.4-72.56': {'feedback': 'Emphasize distinct functions of each brain structure, and relate structure to behavior for better engagement.', 'transcript': ' side inside the temporal lobe lies the other hippocampus and amygdala pair.'}, '72.56-80.04': {'feedback': 'Emphasize distinct functions of each brain structure, and relate structure to behavior for better engagement.', 'transcript': ' So for the lobes, at the front you have the frontal lobe, and at the top the parietal'}, '80.04-82.32': {'feedback': 'Emphasize distinct functions of each brain structure, and relate structure to behavior for better engagement.', 'transcript': ' lobe, and at the back the occipital lobe.'}, '82.32-86.36': {'feedback': 'Emphasize distinct functions of each brain structure, and relate structure to behavior for better engagement.', 'transcript': ' The bottom, the yellow, is the temporal lobe.'}, '86.36-97.2': {'feedback': 'Emphasize distinct functions of each brain structure, and relate structure to behavior for better engagement.', 'transcript': ' For the cortexes, the cortex in the frontal lobe is the motor cortex, and that sits at'}, '97.2-101.52': {'feedback': 'Provide additional details on brain cortexes, clearly label and point out each section on the model, and connect location to function for a comprehensive understanding.', 'transcript': ' the end of the frontal lobe.'}, '101.52-109.12': {'feedback': 'Provide additional details on brain cortexes, clearly label and point out each section on the model, and connect location to function for a comprehensive understanding.', 'transcript': ' At the front of the parietal lobe sits the sensory cortex, and these two cortexes border'}, '109.12-112.56': {'feedback': 'Provide additional details on brain cortexes, clearly label and point out each section on the model, and connect location to function for a comprehensive understanding.', 'transcript': ' each other.'}, '112.56-122.28': {'feedback': 'Provide additional details on brain cortexes, clearly label and point out each section on the model, and connect location to function for a comprehensive understanding.', 'transcript': ' Lastly we have the cerebellum, which is under all the lobes and sits behind the brain stem.'}}, 'summary': '\nContent:\nYour presentation provided a foundational overview of the brain\'s anatomy. However, there\'s room to enhance the accuracy and originality of the facts presented, particularly in the visual aids. It would be beneficial to delve deeper into the specific functions of the brain components to strengthen the persuasiveness of your presentation. Additionally, ensuring that the hippocampus and amygdala locations are correctly depicted in your visual aids is crucial for clarity.\n\nReasoning:\nWhile you outlined the basic structure of the brain, further clarity and memorability could be achieved by emphasizing the distinct functions of each brain structure and relating these structures to behavior. This would not only make the key points more memorable but also provide a critical evaluation of the evidence.\n\nOrganization:\nThe organization of your presentation was generally orderly, but there is an opportunity to enhance the purposefulness by clearly identifying and highlighting each part of the brain when mentioned. This would improve the smoothness of flow and help the audience follow along more easily.\n\nStyle:\nTo hold the audience\'s attention more effectively, consider incorporating engaging questions or facts throughout the presentation. Facilitating discussion by posing questions to the audience can also increase engagement. Responsiveness to the audience\'s questions was good, but adding spontaneity by sparing the use of notes could further improve your delivery.\n\nMechanics:\nYour eye contact and use of visual aids were adequate, but there\'s room for improvement in fluency. Try to avoid filler words such as "uh," "um," and "like," which were noticeable during your talk. These can distract from the content and disrupt the flow of your presentation. Additionally, more expressive hand and arm gestures could be used to emphasize points without fidgeting.\n\nExtra Feedback:\nFor a more comprehensive understanding, provide additional details on the brain cortexes. Clearly label and point out each section on the model, and connect the location to its function. This will help the audience grasp the complexity of the brain\'s anatomy and its relevance to human behavior.\n\nOverall, with a few adjustments to the accuracy of visual aids, a deeper exploration of the brain\'s functions, and a reduction in the use of filler words, your presentation will be more impactful and engaging.', 'score': {'Content': 70, 'Reasoning': 75, 'Organization': 80, 'Style': 65, 'Mechanics': 60, 'filler_words_count': 31, 'filler_words_ratio': 0.13304721030042918}}))

    # print(first_frame)


asyncio.run(main())
