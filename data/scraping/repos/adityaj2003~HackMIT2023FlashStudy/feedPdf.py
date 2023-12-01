import openai
import sys
import os

def generate_summary(transcript):
    system_prompt = "Answer the question based on the pdf text given earlier. "
    openai.api_key = "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"
    response = openai.Completion.create(
        engine="gpt-3.5-turbo",
        prompt=system_prompt + transcript,
    )
    return response.choices[0].text.strip()

if __name__ == "__main__":
    transcript_text = sys.argv[1]
    summary_text = generate_summary(transcript_text)
    print(summary_text)

