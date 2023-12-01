from openai import OpenAI
import os

client = OpenAI()

def generate_image_prompt(context, current_text):
    """Generate an image prompt based on the current text and previous context."""
    prompt = (
        f"Context: {context}\n"
        "The following text was transcribed from a toddler speaking in romanian. Keeping in mind the above context, generate a concise, 20-word image prompt for a toddler-style illustration based on the below text.:\n"
        f"Text: {current_text}"
    )
    print(f"--------------------------------------")
    print(f"Generating image prompt for: {prompt}")

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": prompt}
        ]
    )

    # Correctly access the response content
    generated_prompt = response.choices[0].message.content

    print(f"Generated image prompt: {generated_prompt}")
    return generated_prompt


def process_subtitles_for_image_prompts(input_file_path, output_file_path):
    with open(input_file_path, 'r') as infile:
        lines = infile.readlines()

    context_prompts = []  # List to store the most recent prompts
    image_prompts = []

    for i in range(0, len(lines), 4):
        print(f"Processing subtitle {i//4 + 1} of {len(lines)//4}")
        if i + 1 < len(lines) and ' --> ' in lines[i+1]:
            print(f"Skipping line {i+1} because it is a timestamp")
            subtitle_text = lines[i+2].strip()
            prompt = generate_image_prompt(context_prompts[-3:], subtitle_text)
            
            # Update context with the new prompt, keeping only the last 3 prompts
            context_prompts.append(prompt)
            if len(context_prompts) > 3:
                context_prompts.pop(0)  # Remove the oldest prompt

            image_prompts.append(f"{lines[i]}{lines[i+1]}{prompt}\n\n")

    # Write the image prompts to the output file
    with open(output_file_path, 'w') as outfile:
        outfile.writelines(image_prompts)

    print(f"Image prompts written to {output_file_path}")
# Example usage
input_file_path = "sources/eric1_accumulated.srt"
output_file_path = "sources/eric1_image_prompts.srt"
process_subtitles_for_image_prompts(input_file_path, output_file_path)
