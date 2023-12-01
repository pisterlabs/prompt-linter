import openai


def generate_similar_prompt(name: str) -> str:
    """
    Generate prompt for similar movies
    :param name: keywords to search for movies
    :return: prompt for similar movies
    """
    print("Suggest 5 movies with name and year, and those movies are similar to movie {name} ".format(name=name))
    return "Suggest 5 movies with name and year, and those movies are similar to movie {name} ".format(name=name)


def similar_service(name: str) -> openai.Completion:
    """
    Search for similar movies
    :param name: keywords to search for movies
    :return: response from OpenAI
    """
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=generate_similar_prompt(name),
        temperature=0.5,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response
