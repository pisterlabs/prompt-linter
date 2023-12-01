from datetime import datetime
from logging.config import listen
import speech_recognition as sr
import pyttsx3 
import webbrowser
import wikipedia
import wolframalpha
import openai
import os



# Speech engine initialisation
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id) # 0 = male, 1 = female
activationWord = 'jonas' # Single word

# Configure browser
# Set the path
chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
webbrowser.register('chrome', None, webbrowser.BackgroundBrowser(chrome_path))

#Openai API key
openai.api_key = 'insert your own openai API key here!'

#Stops gpt from running, you can edit this if you have another method
prediction_made = False

#send prompt via api to get our prediction with GPT-3
def predict_gpt(query = ''):
    global prediction_made
    prediction = openai.Completion.create(
        engine='text-davinci-001',
        prompt=query,
        temperature=0.5,
        max_tokens=150,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    text = prediction.choices[0].text
    return speak(text)

# Wolfram Alpha client
appId = 'Insert your API key here!'
wolframClient = wolframalpha.Client(appId)

#jonas' vocal chords 
def speak(text, rate = 120):
    engine.setProperty('rate', rate)
    engine.say(text)
    engine.runAndWait()

#jonas' ears 
def parseCommand():
    listener = sr.Recognizer()
    print('Listening for a command')

    with sr.Microphone() as source:
        listener.pause_threshold = 2
        input_speech = listener.listen(source)

    try: 
        print('Recognizing speech...')
        query = listener.recognize_google(input_speech, language='en_gb')
        print(f'The input speech was: {query}')
    except Exception as exception:
        print('I did not quite catch that')
        speak('I did not quite catch that')
        print(exception)
        return 'None'

    return query

#wikipedia search method
def search_wikipedia(query = ''):
    searchResults = wikipedia.search(query)
    if not searchResults:
        print('No wikipedia result')
        return 'No result received'
    try: 
        wikiPage = wikipedia.page(searchResults[0])
    except wikipedia.DisambiguationError as error:
        wikiPage = wikipedia.page(error.options[0])
    print(wikiPage.title)
    wikiSummary = str(wikiPage.summary)
    return wikiSummary

def listOrDict(var):
    if isinstance(var, list):
        return var[0]['plaintext']
    else:
        return var['plaintext']

#wolfram search method
def search_wolframAlpha(query = ''):
    response = wolframClient.query(query)

    # @success: Wolfram Alpha was able to resolve the query
    # @numpods: Number of results returned
    # pod: List of results. This can also contain subpods
    if response['@success'] == 'false':
        return 'Could not compute'
    
    # Query resolved
    else:
        result = ''
        # Question 
        pod0 = response['pod'][0]
        pod1 = response['pod'][1]
        # May contain the answer, has the highest confidence value
        # if it's primary, or has the title of result or definition, then it's the official result
        if (('result') in pod1['@title'].lower()) or (pod1.get('@primary', 'false') == 'true') or ('definition' in pod1['@title'].lower()):
            # Get the result
            result = listOrDict(pod1['subpod'])
            # Remove the bracketed section
            return result.split('(')[0]
        else: 
            question = listOrDict(pod0['subpod'])
            # Remove the bracketed section
            return question.split('(')[0]
            # Search wikipedia instead
            speak('Computation failed. Querying universal databank.')
            return search_wikipedia(question)



# Main loop
if __name__ == '__main__':
    speak('Welcome')

    while True:
        # Parse as a list
        query = parseCommand().lower().split()

        if query[0] == activationWord:
            query.pop(0)

            # List commands
            if query[0] == 'say':
                if 'hello' in query:
                    speak('Hello, my name is Jonas.')
                else: 
                    query.pop(0) # Remove say
                    speech = ' '.join(query)
                    speak(speech)

            # Navigation
            if query[0] == 'go' and query[1] == 'to':
                speak('Opening...')
                query = ' '.join(query[2:])
                webbrowser.get('chrome').open_new(query)

            # Wikipedia 
            if query[0] == 'wikipedia':
                query = ' '.join(query[1:])
                speak('Querying wiki')
                speak(search_wikipedia(query))
                
            # Wolfram Alpha
            if query[0] == 'compute' or query[0] == 'calculate':
                query = ' '.join(query[1:])
                speak('Computing')
                try: 
                    result = search_wolframAlpha(query)
                    speak(result)
                except:
                    speak('Unable to compute.')

            # Note taking
            if query[0] == 'log':
                speak('Ready to log')
                newNote = parseCommand().lower()
                now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                with open('note_%s.txt' % now, 'w') as newFile:
                    newFile.write(newNote)
                speak('Logged')
            
            #openai prediction via GPT-3
            if query[0] == 'gpt' and not prediction_made:
                prediction = predict_gpt(' '.join(query[1:]))
                prediction_made = True

            #program exit
            if query[0] == 'exit':
                speak('Goodbye')
                break
