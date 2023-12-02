from fetchtodb import Fetcher
import openai
# import openapi
# openapi.__init__()
a = Fetcher
testdata="""Full Name: Zardasht Mudur
                Email: ZardashtKaya@gmail.com
                Phone: +964 (751) 101-4123
                Address: N/A
                Content Creation: 1.0 
                Photography: 0.75
                Graphic Design: 0.90"""

completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """I will give you a resume of someone in the following format
                Full Name: [name]
                Email: [email]
                Phone: [phone]
                Address: [address]
                [category]: [score in percentage from 0.0 to 1.0]
                your job is to respond with only a comma seperated output for only the skills, so you only seperate the skills with a comma and no spaces between them.
                """},
                {"role": "user", "content": testdata}])
test = completion.choices[0].message.content.strip().split(',')
a.add_skill(test)
Fetcher.get_skills()