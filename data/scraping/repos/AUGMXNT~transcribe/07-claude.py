from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

anthropic = Anthropic()


with open('transcript.txt', 'r') as f:
    text = f.read()

def call_claude(content):
    print(f"Calling claude-2... {len(content)}")
    prompt = f'Reformat the following transcript into Markdown, bolding the speakers. Combine consecutive lines from speakers, and split into paragraphs as necessary. Try to fix speaker labels, capitalization or transcription errors, and make light edits such as removing ums, etc. There is some Danish, please italicize the Danish sentences. Reply with only the corrected transcript as we will be using your output programmatically:\n\n{content}'

    completion = anthropic.completions.create(
        model="claude-2",
        max_tokens_to_sample=100000,
        prompt=f"{HUMAN_PROMPT} {prompt}{AI_PROMPT}",
    )
    return completion.completion

line_count = 0
max_lines = 100
input = ""
output = ""
with open("transcript.txt") as file:
    for line in file:
        input += line
        line_count += 1 

        if line_count >= max_lines:
            output += call_claude(input)
            line_count = 0
            input = ""

output += call_claude(input)

with open("transcript-edited-by-claude2-try2.md", "w") as file:
    file.write(output)
