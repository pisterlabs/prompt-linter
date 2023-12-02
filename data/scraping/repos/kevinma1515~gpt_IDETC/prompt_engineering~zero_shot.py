import openai
from config import API
from pathlib import Path
import csv

design_prompts_base = [
    "Generate 100 design solutions for a lightweight exercise device that can be used while traveling",
    "Generate 100 design solutions for a lightweight exercise device that can collect energy from human motion",
    "Generate 100 design solutions for a new way to measure the passage of time",
    "Generate 100 design solutions for a device that disperses a light coating of a powdered substance over a surface",
    "Generate 100 design solutions for a device that allows people to get a book that is out of reach",
    "Generate 100 design solutions for an innovative product to froth milk",
    "Generate 100 design solutions for a way to minimize accidents from people walking and texting on a cell phone",
    "Generate 100 design solutions for a device to fold washcloths, hand towels, and small bath towels",
    "Generate 100 design solutions for a way to make drinking fountains accessible for all people",
    "Generate 100 design solutions for a measuring cup for the blind",
    "Generate 100 design solutions for a device to immobilize a human joint",
    "Generate 100 design solutions for a device to remove the shell from a peanut in areas with no electricity",
    "Generate 100 design solutions for a device that can help a home conserve energy"
]

design_prompts_novel = [
    "Generate 100 novel design solutions for a lightweight exercise device that can be used while traveling",
    "Generate 100 novel design solutions for a lightweight exercise device that can collect energy from human motion",
    "Generate 100 novel design solutions for a new way to measure the passage of time",
    "Generate 100 novel design solutions for a device that disperses a light coating of a powdered substance over a surface",
    "Generate 100 novel design solutions for a device that allows people to get a book that is out of reach",
    "Generate 100 novel design solutions for an innovative product to froth milk",
    "Generate 100 novel design solutions for a way to minimize accidents from people walking and texting on a cell phone",
    "Generate 100 novel design solutions for a device to fold washcloths, hand towels, and small bath towels",
    "Generate 100 novel design solutions for a way to make drinking fountains accessible for all people",
    "Generate 100 novel design solutions for a measuring cup for the blind",
    "Generate 100 novel design solutions for a device to immobilize a human joint",
    "Generate 100 novel design solutions for a device to remove the shell from a peanut in areas with no electricity",
    "Generate 100 novel design solutions for a device that can help a home conserve energy",
]

design_prompts_diverse = [
    "Generate 100 diverse design solutions for a lightweight exercise device that can be used while traveling",
    "Generate 100 diverse design solutions for a lightweight exercise device that can collect energy from human motion",
    "Generate 100 diverse design solutions for a new way to measure the passage of time",
    "Generate 100 diverse design solutions for a device that disperses a light coating of a powdered substance over a surface",
    "Generate 100 diverse design solutions for a device that allows people to get a book that is out of reach",
    "Generate 100 diverse design solutions for an innovative product to froth milk",
    "Generate 100 diverse design solutions for a way to minimize accidents from people walking and texting on a cell phone",
    "Generate 100 diverse design solutions for a device to fold washcloths, hand towels, and small bath towels",
    "Generate 100 diverse design solutions for a way to make drinking fountains accessible for all people",
    "Generate 100 diverse design solutions for a measuring cup for the blind",
    "Generate 100 diverse design solutions for a device to immobilize a human joint",
    "Generate 100 diverse design solutions for a device to remove the shell from a peanut in areas with no electricity",
    "Generate 100 diverse design solutions for a device that can help a home conserve energy",
]

design_prompts_unique = [
    "Generate 100 unique design solutions for a lightweight exercise device that can be used while traveling",
    "Generate 100 unique design solutions for a lightweight exercise device that can collect energy from human motion",
    "Generate 100 unique design solutions for a new way to measure the passage of time",
    "Generate 100 unique design solutions for a device that disperses a light coating of a powdered substance over a surface",
    "Generate 100 unique design solutions for a device that allows people to get a book that is out of reach",
    "Generate 100 unique design solutions for an innovative product to froth milk",
    "Generate 100 unique design solutions for a way to minimize accidents from people walking and texting on a cell phone",
    "Generate 100 unique design solutions for a device to fold washcloths, hand towels, and small bath towels",
    "Generate 100 unique design solutions for a way to make drinking fountains accessible for all people",
    "Generate 100 unique design solutions for a measuring cup for the blind",
    "Generate 100 unique design solutions for a device to immobilize a human joint",
    "Generate 100 unique design solutions for a device to remove the shell from a peanut in areas with no electricity",
    "Generate 100 unique design solutions for a device that can help a home conserve energy",
]


if __name__ == '__main__':
    openai.api_key = API

    design_prompts = {
                      "base": design_prompts_base,
                      "unique": design_prompts_unique,
                      "diverse": design_prompts_diverse,
                      "novel": design_prompts_novel}

    for name, prompts in design_prompts.items():
        i = 0
        print(name)
        print(prompts)
        for prompt in prompts:
            i += 1
            print(prompt)
            print("--------------------------------------------------")
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
                temperature=0.9,
                max_tokens=4000,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )

            result = response['choices'][0]['text']
            print(result)

            out = Path(f"data/zero_shot/{name}_{i}.csv")
            out.parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving to {out}")

            with out.open('w', newline='\n', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows([[x] for x in result.split("\n")])
