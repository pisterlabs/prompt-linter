import asyncio,json
import openai
from IPython.display import display, Markdown
from pyppeteer import launch
from bs4 import BeautifulSoup
api_key = "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"
openai.api_key=api_key

async def main():

    browser = await launch(Headless = True,Args = ["--no-sandbox","--disable-dev-shm-usage","--disable-gpu","--disable-setuid-sandbox","--no-first-run","--no-zygote","--single-process"])
    prompt=input("Enter User Query:")
    prompt=prompt.replace(" ","+")
    page = await browser.newPage()
    print('https://www.google.com/search?q='+prompt)
    await page.setViewport({"width": 1920,"height": 1080})
    await page.goto('https://www.google.com/search?q='+prompt)
    
    titles = await page.querySelectorAll('.VwiC3b,.yXK7lf,.MUxGbd,.yDYNvb,.lyLwlc')
    links = await page.querySelectorAll('.fG8Fp,.uo4vr')
   
    productlinks=[]
    price=[]
    for i in range(3):
        title1 = await page.evaluate('(element) => element.textContent', titles[i])
##        print("title"+str(i+1)+"=",title1)
        links1 = await page.evaluate('(element) => element.textContent', links[i])
##        print("links"+str(i+1)+"=",links1, "\n")
        price.append(links1)
        productlinks.append(title1+links1)
  
    await browser.close()
    gprompt="create a description with equals 400 words from the given list of paragraphs ,"+str(productlinks)
    
    #print(gprompt)
    
    response = openai.Completion.create( 
            engine="text-davinci-002", 
            prompt=gprompt,  
            temperature=0, 
            max_tokens=500, 
            top_p=1.00)
    gprompt="Fine a average prize and rating from the sentences list, The list of sentences are ,"+str(price)
    
    #print(gprompt)
    
    response1 = openai.Completion.create( 
            engine="text-davinci-002", 
            prompt=gprompt,  
            temperature=0, 
            max_tokens=50, 
            top_p=1.00) 

##    display(Markdown('**'+ gprompt +'**'))
    print(response['choices'][0]['text'].strip()+'\n')
    print(response1['choices'][0]['text'].strip()+'\n')
    
print("Starting...")
asyncio.get_event_loop().run_until_complete(main())