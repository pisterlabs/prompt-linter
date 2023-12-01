import os
import openai

# load in the LLM_URL from the environment
openai.api_base = os.environ["LLM_URL"]


prompt_template = """
Convert these emails that describe work shifts into a list of calendar entries with the following format:

event(start_date="%d/%m/%y", start_time="%H:%M", end_date="%d/%m/%y", end_time="%H:%M")

i.e
event(start_date="08/06/07", start_time="14:00", end_date="08/06/07", end_time="22:00")


Where:
%d: Day of the month as a zero-padded decimal number. (i.e. 08)
%m: Month as a zero-padded decimal number. (i.e. 06)
%y: Year without century as a zero-padded decimal number. (i.e. 07)
%H: Hour (24-hour clock) as a zero-padded decimal number. (i.e. 14)
%M: Minute as a zero-padded decimal number. (i.e. 00)


And the email you get is somewhat free form, which the recieved data present.

---

EMAIL:
recieved: 08/06/07

CONTENT:
Hey Olivia, here is your work schedule for the week of 10/06/07 - 17/06/07

Monday: 14:00 - 22:00
Tuesday: 10:00 - 18:00
Wednesday: 14:00 - 18:00
Thursday: 22:00 - 06:00

ANSWER:
event(start_date="10/06/07", start_time="14:00", end_date="10/06/07", end_time="22:00")
event(start_date="11/06/07", start_time="10:00", end_date="11/06/07", end_time="18:00")
event(start_date="12/06/07", start_time="14:00", end_date="12/06/07", end_time="18:00")
event(start_date="13/06/07", start_time="22:00", end_date="14/06/07", end_time="06:00")

---

EMAIL:
recieved: 18/06/07

CONTENT:
{content}

ANSWER:

"""


def main():
    content = "TODO"

    response = openai.Completion.create(
        model="vicuna-13b-v1.1-8bit",
        prompt=prompt_template.format(content=content),
        temperature=0,
        max_tokens=2000,
    )

    print(response)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    main()
