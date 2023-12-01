import datetime
import creds
import openai

openai.api_key = creds.open_ai_api_key

offline_greetings = [
    "Good morning",
    "Good afternoon",
    "Good evening",
]


def generate_greeting():
    try:
        prompt = ""
        now = datetime.datetime.now()
        if now.hour < 12:
            prompt = f"The time is {now}. You are a cynical and depressed AI who's only purpose is to generate a short morning greeting. Generate a 10ish word snarky greeting."
        elif now.hour < 18:
            prompt = f"The time is {now}. You are a cynical and depressed AI who's only purpose is to generate a short afternoon greeting. Generate a 10ish word snarky greeting."
        else:
            prompt = f"The time is {now}. You are a cynical and depressed AI who's only purpose is to generate a short evening greeting. Generate a 10ish word snarky greeting."
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=[{"role": "system", "content": prompt}]
        )

        greeting = response.choices[0].message
        return greeting["content"].replace('"', "")
    except:
        if now.hour < 12:
            return offline_greetings[0]
        elif now.hour < 18:
            return offline_greetings[1]
        else:
            return offline_greetings[2]


def main():
    now = datetime.datetime.now()
    now = now.hour
    print(generate_greeting())


if __name__ == "__main__":
    main()

# input("Press the any key: ")
