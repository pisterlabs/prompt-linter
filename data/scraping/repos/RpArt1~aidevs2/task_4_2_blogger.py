from tasks.abstractTask import AbstractTask
from openai import OpenAI
import json


class BloggerTask(AbstractTask):
    TASK_NAME = 'blogger'

    def process_task_details(self):
        system_content_new = 'based on provided input generate very short blog paragraphs and return them in json: {"paragraphs title" : "paragraphs content" }'

        joined_string = " ".join([self.assignment_body['msg'], '; '.join(self.assignment_body['blog'])])
        print(f"body: {joined_string}")

        client = OpenAI()
        messages = [
            {"role": "system", "content": system_content_new},
            {"role": "user", "content": joined_string}
        ]

        response = client.chat.completions.create(
            model=self.GPT_MODEL,
            messages=messages,
            temperature=0
        )

        response_message = response.choices[0].message.content

        json_object = json.loads(response_message)
        print(f"json_object {json_object}")

        array_with_content = list(json_object.values())
        print(f"array_with_content {array_with_content}")

        return array_with_content

    def solve_task(self):
        super().solve_task()

