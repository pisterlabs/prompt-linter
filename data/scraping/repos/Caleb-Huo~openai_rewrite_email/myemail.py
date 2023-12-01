from openai import OpenAI
client = OpenAI(api_key='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')        

class Myemail:
    
    def __init__(self):
        self.before = ""
        self.after = ""
        self.system = {"role": "system", 
                   "content": "My name is Alex. I am a grad student in Department of Biostatistics. Please help me rewrite my email in a polite, professional and concise way."}
        self.user = {"role": "user", 
                 "content": self.before}
    
    def get_input(self):
        self.before = input("This is personalized email rewriter. Q for quit application. Please provide your original email: \n")
        self.user.update({"content": self.before})
    
    def rewrite(self):
        completion = client.chat.completions.create(  
            model="gpt-3.5-turbo",
            messages=[
                self.system,
                self.user
            ]
        )
        self.after = completion.choices[0].message.content
        print(self.after)

myemail = Myemail()
while myemail.before != "Q":
    myemail.get_input()
    if myemail.before != "Q":
        myemail.rewrite()