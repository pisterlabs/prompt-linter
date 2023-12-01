import sys
import youtube_api as yt
import openai_api as op
import text_to_speach as tts

def __main__():
    youtube_link = sys.argv[1]
    output_language = sys.argv[2]
    video_id = yt.getVideoId(youtube_link)
    transcript = yt.getTranscriptText(video_id)

    summary = ""
    for text in transcript:
        summary_piece = op.summarize(text, output_language)
        summary += summary_piece

    tts.save(summary_piece, output_language)

if __name__ == "__main__":
    __main__()