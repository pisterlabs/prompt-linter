from pathlib import Path
from openai import OpenAI
client = OpenAI()

# Path to save the audio file
speech_file_path = Path(__file__).parent / "voicemail_test_1.mp3"

# The voicemail script
voicemail_script = """
Hello, and thank you for taking the time to listen.

My name is Alice, and I represent OpenAI, where we believe in blending traditional financial wisdom with the exciting opportunities of the future.

I understand that the world of cryptocurrency might seem a bit daunting, especially when it's new and rapidly evolving. But, let's talk about something that's not just about the future - it's about your benefits today.

We've launched an innovative company credit card that offers Bitcoin cashback on every purchase. Now, you might be wondering, why Bitcoin?

Bitcoin, despite its novel nature, represents an opportunity for growth and diversification, something we know is important in any robust financial strategy. And the best part? You don't have to navigate this new terrain alone.

Our card is designed with the stability and familiarity of traditional credit cards, ensuring ease of use and trusted security. Each time you use the card, you'll earn cashback in Bitcoin, allowing you to gradually and securely step into the world of digital currency - without any extra effort.

This isn't just about jumping on a new trend. It's about making a smart choice for your company's financial future, one that harnesses both the reliability of the present and the potential of tomorrow.

We'd love to discuss how this card can align with your business needs and financial strategies. Please give us a call back at 999-999-9999, or visit our website at www.openai.com for more information.

Your time is valuable, and we appreciate it. Looking forward to connecting with you soon. Have a great day!
"""

# Create audio from the text
response = client.audio.speech.create(
  model="tts-1",
  voice="alloy",
  input=voicemail_script
)

# Save the audio file
response.stream_to_file(speech_file_path)
