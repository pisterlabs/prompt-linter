# # from gtts import gTTS
# import os
# from dotenv import load_dotenv
# from openai import OpenAI
# client = OpenAI(api_key="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")
# from pydub import AudioSegment
# import io
# # Load configurations from .en
# def voice(des):
#     audio = []
#     para = des.split('\n')
#     for i in para:
#         if i == '':
#             continue
#         if i[0:5] == "Guest":
#             tts_g = client.audio.speech.create(
#                 model="tts-1",
#                 voice="echo",
#                 input=i[7:]
#             )
#             if tts_g:
#                 audio.append(tts_g.content)
#         elif i[0:4] == "Host":
#             tts_h = client.audio.speech.create(
#                 model="tts-1",
#                 voice="alloy",
#                 input=i[6:]
#             )
#             if tts_h:
#                 audio.append(tts_h.content)

#     # Convert the bytes to AudioSegment and concatenate
#     audio_segments = [AudioSegment.from_mp3(io.BytesIO(segment)) for segment in audio]
#     final_audio = sum(audio_segments)

#     # Save the final audio to an MP3 file
#     final_audio.export('TestTTS.mp3', format='mp3')

#     print("done")

# text = '''Host: Welcome to our podcast today, where we'll be discussing one of the most successful companies in the world - Alibaba. Joining me is our guest expert, John, who has been following Alibaba's journey closely. John, welcome to the show!

# Guest: Thank you for having me! I'm excited to share my insights on Alibaba's success story.

# Host: For those who may not know much about Alibaba, can you give us a brief overview of the company and its history?

# Guest: Sure! Alibaba Group Holding Limited is a Chinese multinational conglomerate that specializes in e-commerce, retail, Internet, and technology. The company was founded in 1999 by Jack Ma, a former English teacher, along with a group of 18 other people. They started out by creating a platform for small businesses in China to sell their products online, and from there, the company rapidly grew into the massive conglomerate it is today.

# Host: That's incredible! So, what do you think has been the key factor in Alibaba's success?

# Guest: There are several factors that have contributed to Alibaba's success, but if I had to choose just one, it would be their focus on innovation. Alibaba has constantly pushed boundaries and disrupted traditional industries through cutting-edge technologies such as artificial intelligence, cloud computing, and data analytics. They've also invested heavily in research and development, which has allowed them to stay ahead of the curve and adapt quickly to changing market trends.

# Host: Interesting. Can you tell us more about Alibaba's e-commerce platforms, such as Taobao and Tmall? How have they managed to dominate the Chinese e-commerce market?

# Guest: Of course! Taobao Marketplace and Tmall are Alibaba's two main e-commerce platforms. Taobao is a consumer-to-consumer (C2C) platform that allows individuals to sell goods to other individuals, while Tmall is a business-to-consumer (B2C) platform that enables brands and businesses to sell directly to consumers. Both platforms have been wildly popular in China, with hundreds of millions of active users. One reason for their success is the wide range of products available on the platforms. Consumers can find everything from fashion and electronics to home appliances and even groceries. Another key factor is the user experience; Alibaba has invested heavily in developing intuitive interfaces, streamlined checkout processes, and reliable logistics networks to ensure customers receive their purchases quickly and efficiently. Finally, Alibaba has implemented various strategies to foster customer loyalty and drive sales, such as promotional events like "Singles Day" and "6.18 Mid-Year Sale," as well as integrated payment systems like AliPay and affiliated services like AlipayHK. These efforts have helped Alibaba capture an enormous share of the Chinese e-commerce market, making it difficult for competitors to catch up.'''

# voice(text)



# # import os
# # from functions.database import get_recent_messages
# # from dotenv import load_dotenv
# # from openai import OpenAI

# # # Load configurations from .env file
# # load_dotenv()




# # def convert_text_to_speech(message):
# #     try:
# #         response = client.audio.speech.create(
# #             model="tts-1",
# #             voice="echo",
# #             input=message
# #         )
# #         if response:
# #             return response.content
# #         else:
# #             return None
# #     except Exception as e:
# #         print(f"Error in OpenAI API call (convert_text_to_speech): {e}")
# #         return None
