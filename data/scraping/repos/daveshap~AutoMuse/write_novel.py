import openai
import os


with open('openaiapikey.txt', 'r') as infile:
    open_ai_api_key = infile.read()
openai.api_key = open_ai_api_key


novel_dir = 'C:/AutoMuse/novel/'
summary_dir = 'C:/AutoMuse/summaries/'


def load_files(directory):
    result = list()
    for f in os.listdir(directory):
        with open(directory + f, 'r', encoding='utf-8') as infile:
            result.append(infile.read().strip())
    return result


def next_filename():
    last_filename = os.listdir(novel_dir)[-1].strip('.txt')
    number = int(last_filename)
    number += 1
    new_filename = str(number).zfill(4) + '.txt'
    return new_filename


def save_file(fullpath, content):
    with open(fullpath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


def completion(prompt, engine, temp=0.7, top_p=1.0, tokens=400, freq_pen=0.75, pres_pen=0.75, stop=['Summary:', 'Last few lines:', 'Write a long continuation of the above story:', 'Detailed summary:']):
    try:
        print('\n\nPROMPT:', prompt)
        response = openai.Completion.create(
            engine=engine,
            prompt=prompt,
            temperature=temp,
            max_tokens=tokens,
            top_p=top_p,
            frequency_penalty=freq_pen,
            presence_penalty=pres_pen,
            stop=stop)
        response = response['choices'][0]['text'].strip()
        print('\n\nRESPONSE:', response)
        return response
    except Exception as oops:
        print('ERROR in completion function:', oops)
        return None


if __name__ == '__main__':
    counter = 0
    while True:
        # 1) load all summaries (and compress?)
        # 2) load last prose block
        # 3) Populate prompt
        # 4) Generate next prose block, save it
        # 5) Generate summary of new prose block, save it
        counter += 1
        if counter >=10:
            exit(0)
        summaries = load_files(summary_dir)
        summary = " ".join(summaries)
        last_prose = load_files(novel_dir)[-1]
        with open('C:/AutoMuse/mainprompt2.txt', 'r', encoding='utf-8') as infile:
            prompt = infile.read().replace('<<SUMMARY>>', summary).replace('<<NOVEL>>', last_prose)
        next_prose = completion(prompt, 'davinci-instruct-beta')
        with open('C:/AutoMuse/summaryprompt2.txt', 'r', encoding='utf-8') as infile:
            prompt = infile.read().replace('<<CHUNK>>', next_prose)
        new_summary = completion(prompt, 'davinci-instruct-beta')
        filename = next_filename()
        save_file(novel_dir + filename, next_prose)
        save_file(summary_dir + filename, new_summary)
        whole_novel = "\n  ".join(load_files(novel_dir))
        save_file('C:/AutoMuse/novel.txt', whole_novel)