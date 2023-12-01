import os
import openai
import time

# Set your OpenAI API key
openai.api_key = "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"


def generate_code_snippet(prompt, language):
    extension = ""
    if language == "Python":
        extension = ".py"
    elif language == "Java":
        extension = ".java"
    elif language == "C++":
        extension = ".cpp"
    elif language == "C#":
        extension = ".cs"
    elif language == "PHP":
        extension = ".php"
    elif language == "JavaScript":
        extension = ".js"
    # Add more cases for other programming languages if needed.

    code = ""
    while True:
        try:
            response = openai.Completion.create(
                engine="text-davinci-002",
                prompt=prompt,
                max_tokens=50,  # Adjust the token limit as needed
                temperature=0.7,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            code_fragment = response.choices[0].text
            code += code_fragment

            if len(code) >= 4096:
                break  # Assume code is too long

        except openai.error.RateLimitError as e:
            # Handle rate limit error by waiting for the specified duration
            response_headers = e.__dict__.get("response")  # Access response headers
            if response_headers and "Retry-After" in response_headers.headers:
                wait_time = int(response_headers.headers["Retry-After"])
                print(f"Rate limit reached. Waiting for {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("Rate limit error encountered. No Retry-After provided.")
                break  # Break the loop if no retry duration is provided

    return code, extension


def save_code_snippet_to_folder(topic, language, extension, code):
    folder_name = f"{topic}_{language}_code"
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    file_name = f"snippet{extension}"
    file_path = os.path.join(folder_name, file_name)
    with open(file_path, "w") as file:
        file.write(code)


if __name__ == "__main__":
    topic = input("Enter the topic for the code: ")

    languages = ["Python", "Java", "C++", "C#", "PHP", "JavaScript"]

    for language in languages:
        code, extension = generate_code_snippet(
            f"Generate a {language} code snippet for {topic}", language
        )
        if code:
            save_code_snippet_to_folder(topic, language, extension, code)
            print(
                f"{language} code snippet generated and saved in {topic}_{language}_code folder."
            )
        else:
            print(f"{language} code snippet generation failed due to rate limit error.")
