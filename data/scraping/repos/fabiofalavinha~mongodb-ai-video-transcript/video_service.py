import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from langchain.document_loaders import YoutubeLoader

from pytube import YouTube

import youtube_link
from link import Link
from openai_service import OpenAIService

from viceo_configuration_actions import VideoConfigurationActions
from video_transcript import VideoTranscript
from video_transcription_result import VideoTranscriptResult
from youtube_link import YouTubeLink

import cv2
import pytesseract


class VideoService:
    def __init__(self):
        self.locker = Lock()
        self.executor = ThreadPoolExecutor()

    def generateTranscription(self, link: YouTubeLink, video_actions: VideoConfigurationActions) -> VideoTranscriptResult:
        print(f"Loading YouTube video [{link.url}]...")

        loader = YoutubeLoader.from_youtube_url(link.url, add_video_info=True)
        result = loader.load()

        video_transcription = VideoTranscript()
        video_transcription.videoUrl = link.url
        for document in result:
            video_transcription.metadata = document.metadata
            video_transcription.transcript = document.page_content

        if video_actions.ai_enabled:
            try:
                print(f"Summarizing YouTube video transcript [{link.url}]...")
                summarized_response = video_actions.openai_service.promptChatCompletion(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant who has access to a video transcript. "
                                       "Use this information to provide detailed and context-rich explanations."
                        },
                        {
                            "role": "user",
                            "content": f"Here is the transcript of a video that I would like you to analyze and explain. "
                                       f"Provide a summary of the key points and an analysis of the main themes presented "
                                       f"in the video using the below context? Context: {video_transcription.transcript}"
                        }
                    ]
                )
                video_transcription.summary = summarized_response.strip()
            except Exception as ex:
                print(f"Error summarizing YouTube video transcript [{link.url}]: {ex}")
                video_transcription.summary = f"Error summarizing YouTube video transcript [{link.url}]: {ex}"

            if video_actions.code_analysis_enabled:
                video_transcription.code_analysis = self.analyzeCodeInVideo(link, video_actions.openai_service)

        # Return result
        return VideoTranscriptResult(video_transcription)

    @staticmethod
    def cleanCode(raw_code):
        return ''.join([char for char in raw_code if char.isalnum() or char in ' \n\t(){}[];,.+-*/='])

    def analyzeCode(self, codeTextToBeAnalyzed, analyzed_codes: [], openAiService: OpenAIService):
        with self.locker:
            cleanedCode = VideoService.cleanCode(codeTextToBeAnalyzed)

            if cleanedCode is not None and len(cleanedCode) > 0:
                with open('extracted_code.txt', 'a') as file:
                    file.write(cleanedCode + '\n')
                    code_analyzed = openAiService.promptChatCompletion(
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a helpful assistant who has access to a video transcript. "
                                           "Use this information to provide detailed and context-rich explanations."
                            },
                            {
                                "role": "user",
                                "content": f"Analyze the following text and determine if it's a programming language code. If it is, "
                                           f"analyze the code and explain the key points and expected output. "
                                           f"If it's not a programming language just return an empty string. "
                                           f"Context: {cleanedCode}"
                            }
                        ]
                    )
                    if code_analyzed is not None and len(code_analyzed) > 0:
                        analyzed_codes.append(code_analyzed)

    def analyzeCodeInVideo(self, link: Link, openAiService: OpenAIService):
        videoUrl = link.url
        print(f"Load YouTube video [{videoUrl}]...")
        yt = YouTube(videoUrl)
        stream = yt.streams.get_highest_resolution()
        videoId = str(uuid.uuid4())
        stream.download(filename=f'video_{videoId}.mp4')
        cap = cv2.VideoCapture(f'video_{videoId}.mp4')
        lastCodeText = ""
        analyzed_codes = []

        totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames in the video: {totalFrames}")

        frameIndex = 0
        while cap.isOpened():
            print(f'\rProcessing frame {frameIndex} out of {totalFrames}', end='', flush=True)
            ret, frame = cap.read()

            if not ret:
                break

            try:
                codeText = pytesseract.image_to_string(frame)

                if codeText != lastCodeText:
                    self.executor.submit(self.analyzeCode, codeText, analyzed_codes, openAiService)
                    lastCodeText = codeText

            except Exception as ex:
                print(ex)

            frameIndex += 1

        self.executor.shutdown()
        cap.release()
        cv2.destroyAllWindows()

        return analyzed_codes
