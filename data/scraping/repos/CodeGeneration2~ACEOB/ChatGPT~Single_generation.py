import openai

import torch

openai.api_key = '                       '



def generate_code(input_code):
    with torch.no_grad():
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user",
                 "content": f"Please optimize the following inefficient code into a more efficient version, while keeping the functionality unchanged (The generated code should start with <s>\n and end with \n</s>, only the generated code is needed):\n{input_code}"}])

    return completion.choices[0].message["content"]


if __name__ == '__main__':
    input_code = """
for _ in range(int(input())):
    n = int(input())
    l = list(map(int, input().split()))
    ans = sum(l)
    if n == 2:
        ans = max(ans, 5 * abs(l[0] - l[1]))
    elif n == 3:
        lx = [l[0], l[2], abs(l[0] - l[1]), abs(l[1] - l[2]), abs(l[5] - l[2])]
        ans = max(ans, max(lx) * 3)
    else:
        ans = max(ans, max(l) * n)
    print(ans)


"""

    output = generate_code(input_code)
    print(output)
