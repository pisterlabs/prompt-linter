import openai
from os import getenv as os_getenv

class Sugerencias(object):
    def __init__(self,title,description):
        self.title = title
        self.description = description

class GiftAI(object):
    """Interfaciong with chatGPT to generate the gift selection"""

    def __init__(self,name,relationship_type,age,intrests,story,state):
        promptText = """name: {}
type of relationship: {}
Age: {}
Intrests: {}
How we met: {}
How the relationship is now: {}""".format(name,relationship_type,age,intrests,story,state)

        # self.products = [
        # ("bear.jpg","teddybear","https://www.amazon.com/Gund-Little-Buddy-Bear-Plush/dp/B0074JU5YQ?&_encoding=UTF8&tag=technova043-20&linkCode=ur2&linkId=af97b4dba1b3e5fa230e7ab2915fdfdb&camp=1789&creative=9325"),
        # ("kittens.jpg","Exploding Kittens Party Pack Card Game","https://www.amazon.com/Exploding-Kittens-Party-Pack-Players/dp/B07CTXHNSL/?&_encoding=UTF8&tag=technova043-20&linkCode=ur2&linkId=7e22f55a17b1403c843e2ab5ffde3826&camp=1789&creative=9325"),
        # ("cookbook.jpg","Let's Eat: 101 Recipes to Fill Your Heart & Home - A Cookbook","https://www.amazon.com/Lets-Eat-Recipes-Heart-Cookbook/dp/1454946393/?&_encoding=UTF8&tag=technova043-20&camp=1789&creative=9325"),
        # ("kissChocolate.jpg","Hershey Kisses Milk Chocolate Silver Foil Wrap Candy","https://www.amazon.com/Hershey-Kisses-Chocolate-Approx-60-Tundras/dp/B0BX7H4P1F/?&_encoding=UTF8&tag=technova043-20"),
        # ("teamug.jpg"," HAPPINESS APPLY HERE Ceramic 15oz Elephant Tea Mug Green","https://www.amazon.com/Volar-Ideas-15oz-Elephant-Green/dp/B071HLGYN9/?&_encoding=UTF8&tag=technova043-20"),
        # ("coloring_book.jpg","ColorIt Modern Art Adult Coloring Book","https://www.amazon.com/ColorIt-Coloring-Drawings-Inspired-Paintings/dp/B09L7L9Z2M/?&_encoding=UTF8&tag=technova043-20"),
        # ("granada.jpg","PenghaiYunfei Collapsible Travel Water Bottle18oz","https://www.amazon.com/Collapsible-Bottle18oz-Reuseable-Carabiner-camouflage/dp/B08P4JKTPV/?&_encoding=UTF8&tag=technova043-20"),
        # ("popcorn.jpg","MOVIE NIGHT Popcorn Kernels and Popcorn Seasoning","https://www.amazon.com/Urban-Accents-Popcorn-Kernels-Seasoning/dp/B07JN5467R/?&_encoding=UTF8&tag=technova043-20"),
        # ("toolkit.jpg","Mini Tool Set for Dorm, Travel, Office, Home","https://www.amazon.com/ABN-25-Piece-Tri-Fold-Travel-Office/dp/B0B4KQ2QKH/?&_encoding=UTF8&tag=technova043-20"),
        # ("deskorg.jpg","Desk Organizers and Accessories","https://www.amazon.com/EOOUT-Organizers-Accessories-Compartments-Stationery/dp/B0BMDNG7QY/?&_encoding=UTF8&tag=technova043-20")
        # ]

        # self.consultarGPT(promptText)
        self.consultarCorto(promptText)


    def apiGPT(self,inputText):
        openai.organization = "org-tl9ABVBxqV5cH1ReIQwL0hBR"
        openai.api_key=os_getenv("GPT_API_KEY", "PLACEHOLDER-KEY")
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": inputText},
            ]
        )
        return (response.choices[0].message.content)


    def consultarCorto(self,description_text):
        promptText="""
Give me a list of 3 alternative SPECIFIC presents that whould fit this person:

Information about the person:
{}

Output format JUST THE ITEMS:
1. <item 1>:<explanation>
2. <item 2>:<explanation>
3. <item 3>:<explanation>

Respod with the leat amount of words
        """.format(description_text)
        response = self.apiGPT(promptText)
        self.arr = []
        for line in response.splitlines():
            if line and (line[0] == '1' or line[0] == '2' or line[0] == '3'):
                sugerencia = Sugerencias(line.split(".")[1].split(":")[0],line.split(".")[1].split(":")[1])
                self.arr.append(sugerencia)

        # print(promptText)
        # print("================================")
        # print(response)
        # print("================================")
        # print(self.arr)

        self.url = "https://www.amazon.com/s?k={}&_encoding=UTF8&tag=technova043-20".format(self.arr[0])

    def consultarGPT(self,description_text):

        selected_prod = 0

        promptText =""" Here 10 amazon products:
id=1 product="{}"
id=2 product="{}"
id=3 product="{}"
id=4 product="{}"
id=5 product="{}"
id=6 product="{}"
id=7 product="{}"
id=8 product="{}"
id=9 product="{}"
id=10 product="{}"

Respond only with a number as the fist character of your response ONLY awser ONE gitft:
<id>
<justification  50 words>

Information about the person:
{}
""".format(self.products[0][1],self.products[1][1],self.products[2][1],self.products[3][1],self.products[4][1],self.products[5][1],self.products[6][1],self.products[7][1],self.products[8][1],self.products[9][1],description_text)

        print(promptText)
        response = self.apiGPT(promptText)
        print("================================")
        print(response)
        selected_prod = int(response[0]) - 1


        self.url = self.products[selected_prod][2]
        self.name = self.products[selected_prod][1]
        self.image = self.products[selected_prod][0]
        self.justificacion = response[2:]
