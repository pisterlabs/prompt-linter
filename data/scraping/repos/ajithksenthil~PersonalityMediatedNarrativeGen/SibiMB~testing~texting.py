
import openai

def gpt3(stext):
    openai.api_key = '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'
    response = openai.Completion.create(
        #        engine="davinci-instruct-beta",
        engine="text-davinci-003",
        prompt=stext,
        temperature=0.2,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    content = response.choices[0].text.split('.')
    # print(content)
    return response.choices[0].text

story = "Bob walks home.Ned notices Stan taking a shortcut through the park and decides to follow him.Ned follows Stan through the park, taking a shortcut to get home."
cutstory = story[-30:]
characters = ["Ben", "Ned", "Stan"]
usersuggestednextactionevent = "Ned throws his food."
print("user suggested next event", usersuggestednextactionevent)
queryuser = f"Given this event representation: {usersuggestednextactionevent}, please generate the next event in the story, taking into account the following context: {cutstory}."
nextevent = gpt3(queryuser)
nextevent = str(nextevent)
nextevent = nextevent.replace("\n", "")
print("user suggestion: ", usersuggestednextactionevent)
print("next event user suggested:", nextevent)