from dotenv import load_dotenv
import os
import cohere
import text_to_speech as tts

load_dotenv()

api_key = os.getenv("COHERE_API_TOKEN_KEY")

def lyrics_generation(prompts):
  """
  Generate lyrics based on the given dance moves.

  Parameters:
  - prompts (list): List of dance moves, first is mandatory, others are optional.

  Returns:
    str: The generated lyrics.
  """

  # Initialize the Cohere client
  co = cohere.Client(api_key)

  # Base prompt
  base_prompt = (
    'Your job is to generate the lyrics for a song\'s first verse and the chorus that fits the mood of the varying dance moves prompt. '
    'Each dance move will be responsible for a portion of the lyrics. The chorus should have a vibe to match the energy of these dance moves. '
    'Put less emphasis on the actual dance move that is prompted, and rather incorporate the themes associated with these various dance moves to create the chorus. '
  )

  # Ensure at least one prompt is available
  if len(prompts) < 1:
    return 'At least one dance move prompt is required.'

  # Add the mandatory first dance move prompt for the verse
  formatted_prompt = f"{base_prompt}Here is the mandatory dance move prompt for the first verse:\n\n{prompts[0]}\n\n"

  # Check for additional prompts and add them
  additional_prompts = []
  for i in range(1, 5):  # This will loop through indexes 1, 2, 3, 4
    if i < len(prompts) and prompts[i] is not None:
      location = "for the verse" if i == 1 else "for the chorus"
      additional_prompts.append(f'Additional dance move {location}: {prompts[i]}')

  formatted_prompt += '\n'.join(additional_prompts)

  # Generate lyrics
  response = co.generate(
    model='command',
    prompt=formatted_prompt,
    max_tokens=128,
    temperature=1.5,
    k=0,
    stop_sequences=[],
    return_likelihoods='NONE'
  )

  # TODO: save lyrics ot a text file

  return response.generations[0].text