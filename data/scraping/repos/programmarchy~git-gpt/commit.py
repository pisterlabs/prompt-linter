import os
import subprocess
import sys
import openai

default_commit_prompt = "Write a detailed git commit message for the following diff. Explain the changes to the code if possible. Don't include the actual command or any prefixes to the message."

def commit(options):
  output = subprocess.check_output(["git", "diff", "--staged"])
  diff = output.decode()
  if not diff:
    raise Exception("no changes added to commit")

  prompt = os.getenv('GITGPT_COMMIT_PROMPT', default_commit_prompt)
  prompt = prompt + '\n\n' + diff

  res = openai.Completion.create(
    engine="text-davinci-003",
    prompt=prompt,
    temperature=0.7,
    max_tokens=250,
    top_p=1,
    frequency_penalty=0.3,
    presence_penalty=0.3
  )

  if not res.choices:
    raise Exception("GPT could not generate a commit message.")

  commit_message = res.choices[0].text.strip()

  command = ["git", "commit", "-m", commit_message]
  if options.dry_run:
    print(' '.join(command))
  else:
    subprocess.run(command)
