import cohere
import os
cohere_api_token=os.environ.get('cohere_api_token')
#text="A character can be any letter, number, punctuation, special character, or space. Each of these characters takes up one byte of space in a computer's memory. Some Unicode characters, like emojis and some letters in non-Latin alphabets, take up two bytes of space and therefore count as two characters. Use our character counter tool below for an accurate count of your characters."
def main(event, context):
      text = event.get("text")
      if text == None:
            return {"Error": "provide text"}
      co = cohere.Client(cohere_api_token)
      response = co.summarize(
            text=text,
            model='command',
            length='long',
            format='bullets',
            extractiveness='low',
            additional_command='focusing on the key ideas from the lecture' # of form: Generate a summary _". Eg. "focusing on the next steps"
      )
      summary = response.summary
      return {"summary": summary}
