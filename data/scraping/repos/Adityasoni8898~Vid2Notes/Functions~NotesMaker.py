import openai

def gpt(api_key, transcript, language, description):

    openai.api_key = api_key

    print("\n\nYou are almost there!")
    print("Making into notes....")
    

    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [
            {"role" : "user", "content" : "convert this transcribed text from a tutorial into good simple notes, write point wise with title content list for each point and write the notes in json."},
            {"role" : "user", "content" : f"This is the langauge the notes are needed in {language} and this is some extra description for understanding, ignore if not understood {description}"},
            {"role" : "user", "content" : transcript }
        ]
    )

    return response.choices[0].message.content