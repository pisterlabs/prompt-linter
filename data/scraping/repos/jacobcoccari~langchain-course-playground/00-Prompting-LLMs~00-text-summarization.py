from dotenv import load_dotenv
import openai
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def chat(message, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": message}],
    )
    return response.choices[0].message["content"]


def main():
    review = f"""I had high hopes for this reasonably priced 10 inch Fire HD Kindle tablet, but I have had nothing but trouble trying to sync my Google apps and Facebook. Can't sign in to my Google account where all my photos are and even locked out of Facebook. Tried 10 times to photo my ID to get into my locked Facebook account. So no matter how nice the screen quality or battery life I will be wiping it clean and sending it back or giving it away. Probably will buy a Google tablet and just make it easy on myself. Planned on taking this tablet with me when I travel, but I will have to find something else. 

    UPDATE- I received a call from Customer Service soon after I posted my review. The rep was able to talk me through a way to access my photos and Facebook. I will keep my tablet for the time being, but I would still not recommend it to people who really want to interface their Google account. Customer Service Rep Christian was great though!"""

    # Please limit your summary to 2-3 sentences.
    prompt = f"""Please concisely summarize the following customer review:

    ```{review}```
    """
    # print(prompt)
    result = chat(prompt)
    print(result)


if __name__ == "__main__":
    main()
