import openai,voice



class Friday:
    def __init__(self) -> None:
        self.key = openai.api_key = "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"
        self.model = "gpt-3.5-turbo"
        self.voice = voice.Voice()

    def LPU(self,inp):
        while True:
            # inp = input("Enter: ")
            completion  = openai.ChatCompletion.create(
                model = self.model,
                messages=[
                    {"role": "system", "content": inp},
                
                ],
                temperature=0.1,
                
            )
               
            return completion.choices[0].message.content
            # print(completion.choices[0].message.content)

    


# def LPU():
#         openai.api_key = "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"
#         while True:
#             inp = input("Enter: ")
#             completion  = openai.ChatCompletion.create(
#                 model = "gpt-3.5-turbo",
#                 messages=[
#                     {"role": "system", "content": inp},
                
#                 ],
#                 temperature=0.1,
                
#             )
               
#             # return completion.choices[0].message.content
#             print(completion.choices[0].message.content)
    

# LPU()

# friday  = Friday()
# friday.LPU("Tell me if I want to see the map of a place or not answer in yes or no: 'Hey show me the map of India'")  