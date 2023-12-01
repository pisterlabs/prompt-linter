import subprocess
import openai


def run(command):
    process = subprocess.run(command, capture_output=True, shell=True, text=True)
    if process.returncode != 0:
        raise Exception(f'{command} failed with exit code {process.returncode}')
    return process.stdout


def checkIfStaged():
    try:
        result = run('git diff --staged')
        if result == '':
            return False
    except Exception:
        return False
    return True


def commitFromPatch(patch):
    prompt = f"""Given the following git patch file:
    {patch}
    ###
    Generate a one-sentence long git commit message.
    Return only the commit message without comments or other text.
    Use conventional commits standard.
    """

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0,
        max_tokens=150)
    # TODO: catch errors
    message = response['choices'][0]['text']
    return message.strip().replace('"', '').replace("\n", '')


if __name__ == '__main__':
    if not checkIfStaged():
        print('No staged commits')
        exit(0)
    diff = run('git diff --staged')
    commit_message = commitFromPatch(diff)
    run(f'git commit -m "{commit_message}"')
    print(f'Committed with message: {commit_message}')
