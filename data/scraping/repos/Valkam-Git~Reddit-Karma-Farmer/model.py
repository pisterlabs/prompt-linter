from openai import OpenAI

# DON'T FORGET TO FILL IN YOUR API KEY HERE OR THE BOT WILL NOT BE ABLE TO GENERATE COMMENTS
client = OpenAI(api_key="YOUR OPENAI API KEY HERE")


def generate_comment(
    post_title: str, post_text: str, comments: list[tuple[str, int]]
) -> str:
    """
    Generates a comment for a Reddit post using OpenAI's GPT-3 API.

    :param post_title: The title of the Reddit post.
    :param post_text: The text content of the Reddit post.
    :param comments: A list of tuples containing the comment content and upvotes.
    :return: The generated comment for the post.
    """

    comments = sorted(comments, key=lambda comment: comment[1], reverse=True)
    if len(comments) >= 4:
        comments = comments[:4]
    else:
        comments = comments[: len(comments)]
    comments = [comment[0] for comment in comments]
    comments = ", ".join(comments)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are an avid reddit user that knows how to provide simple and short interesting comments that will get upvotes. Now I'll provide you the contents of the post and the most important comments and you will have to generate a comment that will get tons of upvotes, it is important that you integrate with the group, mimic their tone and align your opinions to theirs to be upvoted, your way to respond should be similar to the other comments.",
            },
            {
                "role": "user",
                "content": f"The post of title {post_title}, its text content is: {post_text}. The most voted comments are: {comments}. Now generate a comment that will fit in and earn upvotes, remember to speak in a similar tone to the other comments, short phrases and simple words are the best don't be too verbose, only one short phrase with natural language mimicking the others and their mood, don't be too simple or too happy and friendly as it looks bad, make yourself look a bit interesting and NEVER EVER BE AGGRESIVE OR OFFENSIVE. Only reply with the comment, the format of your response is quite important so don't reply anything else.",
            },
        ],
    )

    response = response.choices[0].message.content
    response = response.split("#")[0].replace('"', "")

    return response
