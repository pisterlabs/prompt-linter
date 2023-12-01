import cohere

# Obfuscated key, so that nobody steals my Cohere trial account
KEY_1 = ['n5v36', 'XV6zF', 'FucWL', 'HHQfg']
KEY_2 = ['LEcjw', '9UfJm', 'dcUkA', 'VdNMx']
def gen_key():
    return "".join(KEY_1[i] + KEY_2[i] for i in range(len(KEY_1)))

def generate_prompt(examples, topic, top_text):
    example_string = "\n".join("Top Text: " + example[0] + "\nBottom Text: " + example[1] for example in examples)
    prompt = 'Generate a viral image macro advertising the product \"{topic}\" in a similar format to the examples.\n\n\n{example_string}\nTop Text: {top_text}\nBottom Text: '.format(
        topic=topic,
        example_string=example_string,
        top_text=top_text
    )
    return prompt

# Examples is an array of tuples in format [(top text, bottom text)]
def generate_text(examples, topic, top_text, num_gens=1):
    co = cohere.Client(gen_key())
    response = co.generate(
        model='command',
        prompt=generate_prompt(examples, topic, top_text),
        max_tokens=44,
        temperature=2,
        k=0,
        stop_sequences=[],
        return_likelihoods='NONE',
        num_generations=num_gens
    )
    return [gen.text for gen in response.generations]

def main():
    print(generate_text([('PUSH IN THE SIDES OF YOUR TIN FOIL PACKAGE', 'TWILL KEEP THE BOLL FROM FALLING OUT'), ('THE NIGHT BEFORE, PLACE THINGS YOU DON\'T WANT TO FORGET THE NEXT MORNING...', 'CN TOP OF YOUR SHOES')], 'Lays Chips', 'Eat Lays Chips everyday...'))

if __name__ == '__main__':
    main()
