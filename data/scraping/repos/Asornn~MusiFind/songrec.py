import os
import openai
import argparse

MAX_INPUT_LENGTH = 32

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True)
    args = parser.parse_args()
    user_input = args.input
    if (validate_length(user_input)):
      print(f"User input: {user_input}")
      result = generate_song(user_input)
      print(result)
    else:
       raise ValueError(f"Input length is too long. Must be under {MAX_INPUT_LENGTH}. Submitted input is {user_input}")

def validate_length(prompt):
   return len(prompt) <= MAX_INPUT_LENGTH

def generate_song(artist):
  openai.api_key = os.getenv("OPENAI_API_KEY")

  prompt = f"Give me a song I would like if I enjoy {artist} that isn't by {artist} and that you haven't suggested before in the exact format Artist - Song without any extra text"

  completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "user", "content": prompt}
    ],
  )
  song = completion.choices[0].message["content"]
  return song

if __name__ == "__main__":
   main()