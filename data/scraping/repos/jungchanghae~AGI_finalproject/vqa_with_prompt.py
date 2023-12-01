from datasets import load_dataset
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import random
import base64
from io import BytesIO
from openai import OpenAI

client = OpenAI()
def get_LLM_prompt(img_string):

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Create prompt of image to feed to VQA"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_string}"
                        }
                    },
                ],
            }
        ],
        max_tokens=300,
    )

    return response.choices[0].message.content

def convert_to_rgb(image):
    if image.mode != 'RGB':
        return image.convert('RGB')
    return image

# Load the dataset
dataset = load_dataset("taesiri/imagenet-hard", split='validation[:10%]')

# Load the processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

correct_answers = 0
total_questions = 0

for i, data in enumerate(dataset):
    correct_label = dataset[i]['english_label']
    print(correct_label)
    # Create answer options including the correct answer
    answer_options = [correct_label, "Sea Snake", "Parrot", "Lion", "Dinosaur", "Tiger", "Cloud", "Snail", "Human", "Dog"]
    random.shuffle(answer_options)
    correct_answer_index = answer_options.index(correct_label) + 1  # Adding 1 because options are 1-indexed

    # Formulate the question
    question = f'What is this picture? Choose your number between 10,2,3,4,7,5,6,8,9,1; 1: "{answer_options[0]}", 2: "{answer_options[1]}", 3: "{answer_options[2]}", 4: "{answer_options[3]}", 5: "{answer_options[4]}", 6: "{answer_options[5]}", 7: "{answer_options[6]}", 8: "{answer_options[7]}", 9: "{answer_options[8]}", 10: "{answer_options[9]}"'

    # Process the image and question
    image = convert_to_rgb(data['image'])
    buffered = BytesIO()
    b_image = image
    b_image.save(buffered, format="JPEG")  # Or PNG, depending on your image format
    # Encode the bytes buffer to Base64
    img_base64 = base64.b64encode(buffered.getvalue())
    img_base64_str = img_base64.decode()
    prompt_llm = get_LLM_prompt(img_base64_str)
    inputs = processor(image, question, return_tensors="pt")

    # Get model prediction
    out = model.generate(**inputs)
    try:
        model_answer = int(processor.decode(out[0], skip_special_tokens=True)[0])
        # Evaluate the model's answer
        if model_answer == correct_answer_index:
            correct_answers += 1
        print(model_answer)
    except ValueError:
        print(f"Model output not an integer for image {i}: {processor.decode(out[0], skip_special_tokens=True)[0]}")
        # continue


    # Evaluate the model's answer
    total_questions += 1

# Calculate accuracy
accuracy = correct_answers / total_questions
print(f"Model Accuracy: {accuracy * 100:.5f}%")
