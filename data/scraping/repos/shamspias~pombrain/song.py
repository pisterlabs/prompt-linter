import openai

openai.api_key = ""


async def normal_song(prmt):
    """
    :param prmt:
    :return: context dist
    """
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt="write a song about {}.\n".format(prmt),
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    my_text = response['choices'][0]['text'].split("\n")
    context = {
        'data': [i for i in my_text if not (i == "" or i == " ")]
    }

    return context


async def categories_song(category, topic):
    """
    :param category:
    :param topic:
    :return: context dist
    """
    response = await openai.Completion.create(
        model="text-davinci-002",
        prompt="write a {} song about {}.\n".format(topic, category),
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    my_text = response['choices'][0]['text'].split("\n")
    context = {
        'data': [i for i in my_text if not (i == "" or i == " ")]
    }
    await context
