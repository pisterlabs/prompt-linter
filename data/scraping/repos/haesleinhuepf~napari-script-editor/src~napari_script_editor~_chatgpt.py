
def ask_chat_gpt(message:str, model="gpt-4"):

    print("Question: ", message)

    # todo: make this a setting
    default_prompt = """
    Write Python code like an expert would.
    Do not provide installation instructions or detailed explanations.
    Provide Python code only, preferably using scientific image processing libraries such as numpy, scipy and sckit-image. 
    Try to avoid libraries such as opencv, PIL and pillow.
    In case the task is to process or segment an image, you can get the image using this code: 
        image = list(viewer.layers.selection)[0].data
    In case the result is an image, you can add it to the viewer using this code:
        viewer.add_image(result)
    In case the result is a segmentation or label image, you can add it to the viewer using this code:
        viewer.add_labels(result)
    """

    answer = prompt(default_prompt + message, model=model)

    print("Answer: ", answer)

    if "```" in answer:
        # clean answer by removing blabla outside the code block
        answer = answer.replace("```java", "```")
        answer = answer.replace("```javascript", "```")
        answer = answer.replace("```python", "```")
        answer = answer.replace("```jython", "```")
        answer = answer.replace("```macro", "```")
        answer = answer.replace("```groovy", "```")

        temp = answer.split("```")
        answer = temp[1]

    return answer


def prompt(message:str, model="gpt-4"):
    """A prompt helper function that sends a message to openAI
    and returns only the text response.
    """
    import openai
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": message}]
    )
    return response['choices'][0]['message']['content']