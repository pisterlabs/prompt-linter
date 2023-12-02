import cohere


apiKey = 'yp4eiloxxgobdmF2LvcxLH9OfkT15RR3tu1cZPs5'


def getKeywords(filename):
    baseContent = open('baseRequest.txt', encoding="utf8")
    newDesc = open(filename, encoding="utf8")
    request = baseContent.read() + newDesc.read() + "\nSkills:"
    co = cohere.Client(apiKey)
    response = co.generate(
        model='medium',
        prompt=request,
        max_tokens=80,
        temperature=0.5,
        k=0,
        p=0.8,
        num_generations=2,
        #preset = 'job-search-ykykn1',
        frequency_penalty=0,
        presence_penalty=0,
        stop_sequences=["--"],
        return_likelihoods='NONE')
    keywords = []
    for i in range(0, 2):
        newline = response.generations[0].text.strip().find('\n')
        output = response.generations[0].text[0:newline+1]
        output = output.replace('--', '')
        output = output.replace('and ', '')
        output = output.strip()
        keywords = keywords + output.split(', ')

    keywordsOut = list(set(keywords))

    print(keywordsOut)
    return keywordsOut


if __name__ == '__main__':
    getKeywords('description3.txt')
