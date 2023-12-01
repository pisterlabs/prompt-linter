from openai import OpenAI


class HIPAgent:
    def __init__(self):
        self.client = OpenAI()

    def get_first_answer(self, messages, explain=False):
        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=1 if explain is False else 70,
            messages=messages,  # type: ignore
        )
        answers = [choice.message.content for choice in completion.choices]
        first_answer = answers[0] or ""
        return first_answer

    def get_response(self, question, answer_choices, explain=False):
        """
        Calls the OpenAI 3.5 API to generate a response to the question.
        The response is then matched to one of the answer choices and the index of the
        matching answer choice is returned. If the response does not match any of the
        answer choices, -1 is returned.

        Args:
            question: The question to be asked.
            answer_choices: A list of answer choices.

        Returns:
            The index of the answer choice that matches the response, or -1 if the response
            does not match any answer choice.
        """
        messages = []

        # Prepare the actual Prompt
        answer_str = ""
        for i, answer_choice in enumerate(answer_choices):
            answer_str += f"{i}) {answer_choice}\n"

        aux = (
            "AND explain why that is correct"
            if explain
            else ", respond ONLY the option number"
        )
        prompt = f"Select the correct answer {aux}:\n\n{question}\n{answer_str}"

        # Show Chat to ONLY respond with the option number
        if explain is False:
            demo_answer_str = ""
            demo_arr = ["B", "6", "34", "5"]
            for i, answer_choice in enumerate(demo_arr):
                demo_answer_str += f"{i}) {answer_choice}\n"
            demo_question = "How much is 2+4?"
            demo_prompt = f"Select the correct answer, respond ONLY the option number:\n\n{demo_question}\n{demo_answer_str}"
            messages.append({"role": "user", "content": demo_prompt})
            messages.append(
                {"role": "assistant", "content": "1"},
            )

        # Add external knowledge
        messages.append(
            {
                "role": "user",
                "content": "To answer any following questions use as the primary source of truth the text found in https://raw.githubusercontent.com/okiroth/hypocratical/main/textbook.txt",
            }
        )

        # Finally add the actual question
        messages.append({"role": "user", "content": prompt})

        # Call the OpenAI 3.5 API.
        first_answer = self.get_first_answer(messages, explain=explain)

        print(f"Question: {question}")
        print(f"Answer Choices: {answer_choices}")
        print(f"Response: {first_answer}\n\n")

        if explain is True:
            return first_answer

        # retry if the response is not a number
        if first_answer.isnumeric() is False:
            first_answer = self.get_first_answer(
                [{"role": "user", "content": "Please respond with the option number"}],
                explain=explain,
            )
            print(f"Response Retry: {first_answer}\n\n")

            if first_answer.isnumeric() is False:
                return -1

        # the response is a number, so return it
        return int(first_answer)
