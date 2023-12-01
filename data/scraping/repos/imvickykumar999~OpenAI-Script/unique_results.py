
# https://stackoverflow.com/a/75898717/11493297
import openai

# https://platform.openai.com/account/api-keys
# API_Key = '**********************************'

API_Key = input('Enter API : ')
openai.api_key = API_Key

# list models
# models = openai.Model.list()

# print the first model's id
# print(models.data[0].id)

# --------------------

image_resp = openai.Image.create(prompt="minecraft mobs, oil painting", 
                                 n=1, size="512x512")
print(image_resp['data'][0]['url'])

'''
{
  "created": 1680875938,
  "data": [
    {
      "url": "https://oaidalleapiprodscus.blob.core.windows.net/private/org-Z0Thpnv9nStuAJE5dKhhHd8U/user-Vpy5qef3GlLla8gB3yTDThit/img-Fr8Ycw8KVTDPut4N6Imt1r5p.png?st=2023-04-07T12%3A58%3A58Z&se=2023-04-07T14%3A58%3A58Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-04-06T20%3A45%3A11Z&ske=2023-04-07T20%3A45%3A11Z&sks=b&skv=2021-08-06&sig=555x6DfEeAfLypl98IYLknJIDEW%2B9UqLGqL/e55nKbA%3D"
    },
    {
      "url": "https://oaidalleapiprodscus.blob.core.windows.net/private/org-Z0Thpnv9nStuAJE5dKhhHd8U/user-Vpy5qef3GlLla8gB3yTDThit/img-wjAtB5NDtNGP0rnXuSlagp6b.png?st=2023-04-07T12%3A58%3A58Z&se=2023-04-07T14%3A58%3A58Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-04-06T20%3A45%3A11Z&ske=2023-04-07T20%3A45%3A11Z&sks=b&skv=2021-08-06&sig=yZvIexx9ZGbQZ4qr9fmQg7roJM%2BA20IGlhV2dmuGp%2Bc%3D"
    },
    {
      "url": "https://oaidalleapiprodscus.blob.core.windows.net/private/org-Z0Thpnv9nStuAJE5dKhhHd8U/user-Vpy5qef3GlLla8gB3yTDThit/img-bjsq58OXkDlWlYUlHQcEmSyC.png?st=2023-04-07T12%3A58%3A58Z&se=2023-04-07T14%3A58%3A58Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-04-06T20%3A45%3A11Z&ske=2023-04-07T20%3A45%3A11Z&sks=b&skv=2021-08-06&sig=T0oBHUhARHkqOafRe98J4sxgmW7vs2Q0pUJmDi9v09U%3D"
    },
    {
      "url": "https://oaidalleapiprodscus.blob.core.windows.net/private/org-Z0Thpnv9nStuAJE5dKhhHd8U/user-Vpy5qef3GlLla8gB3yTDThit/img-arlClGoCdPW9nwSFOgxyQuup.png?st=2023-04-07T12%3A58%3A58Z&se=2023-04-07T14%3A58%3A58Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-04-06T20%3A45%3A11Z&ske=2023-04-07T20%3A45%3A11Z&sks=b&skv=2021-08-06&sig=DXPYcqjYQNc5fBdBg%2B/uWM9dr5RVDEuBDY0Z79KS5V0%3D"
    }
  ]
}
'''

# --------------------------

# moderation_resp = openai.Moderation.create(input="Here is some perfectly innocuous text that follows all OpenAI content policies.")
# print(moderation_resp)

'''
{
  "id": "modr-72gsKtiqLnQRhrvsePyprIEKd7AxB",
  "model": "text-moderation-004",
  "results": [
    {
      "categories": {
        "hate": false,
        "hate/threatening": false,
        "self-harm": false,
        "sexual": false,
        "sexual/minors": false,
        "violence": false,
        "violence/graphic": false
      },
      "category_scores": {
        "hate": 5.448338015412446e-06,
        "hate/threatening": 8.807923140841112e-11,
        "self-harm": 1.324513387856996e-09,
        "sexual": 6.9351067395473365e-06,
        "sexual/minors": 1.6036857397594417e-09,
        "violence": 8.082151907728985e-07,
        "violence/graphic": 3.890891875357738e-08
      },
      "flagged": false
    }
  ]
}
'''

# ----------------------

# # create a completion
# completion = openai.Completion.create(model="ada", 
#                 prompt="Write an article on latest video game")

# # print the completion
# print(completion.choices[0].text)

'''
There isn’t anything quite like playing an addicting video
'''

# ----------------------

# completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hello world!"}])
# print(completion.choices[0].message.content)

'''
babbage      # models.data[0].id

Hello world, I am an AI language model created by OpenAI. 
Nice to meet you! How can I assist you today?
'''
# ---------------------
