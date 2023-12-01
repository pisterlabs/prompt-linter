import openai
import re
import argparse
import os 
import time 
from stardict import DictCsv

exclude_words=set(
    ['updated', 'marginr', 'timing', 'primarycolour', 'scaley', 'name', 'play', 'type', 'encoding', 'scaledborderandshadow', 'strikeout', 'bold', 'styles', 'fontsize', 'wrapstyle', 'fontname', 'outlinecolour', 'alignment', 'italic', 'original', 'update', 'shadow', 'text', 'end', 'underline', 'borderstyle', 'marginl', 'title', 'details', 'timer', 'collisions', 'normal', 'script', 'by', 'point', 'v4', 'start', 'dialogue', 'editing', 'translation', 'defaultarial', 'format', 'backcolour', 'events', 'marginv', 'resx', 'outline', 'default', 'resy', 'synch', 'style', 'scalex', 'effect', 'secondarycolour', 'spacing', 'angle', 'layer',
     'am','is','are','was','were','been','have','has',
     'did','do','does','done','doing','will','would','shall','should','may','might','must','can','could','ought','need','dare','had','having',
     'go','goes','going','gone','get','gets','getting','got','gotten',
     'went','come','comes','coming','came',
     ]
    )

def query_gpt3(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[{
        "role": "user", 
        "content": prompt}]
        )
    return response.choices[0].message.content.strip()

def word_unknown(word_query, word_judge):
    if word_query is None:
        return False #查不到就算了 
    
    # 是否认识?
    include_tag=word_judge["include_tag"] 
    exclude_tag=word_judge["exclude_tag"]
    collins_threshold=word_judge["collins_threshold"]; collins_default=True
    bnc_threshold=word_judge["bnc_threshold"]; bnc_default=True
    frq_threshold=word_judge["frq_threshold"]; frq_default=True
    
    # check tag
    include_list=include_tag.lower().split()
    exclude_list=exclude_tag.lower().split()
    if word_query['tag']: # 如果该单词有tag标记
        word_tag=word_query['tag'] 
        tag_chk=(not(any(e in word_tag for e in exclude_list)) 
                 and 
                 any(i in word_tag for i in include_list))    
    else:
        tag_chk=True  #如果该单词没有tag标记, 默认为
    # check collins
    collins_chk = (word_query['collins']<=collins_threshold) if word_query['collins']>=0 else collins_default
    # check bnc
    bnc_chk=(word_query['bnc']>=bnc_threshold) if word_query['bnc']>0 else bnc_default
    # check frq
    frq_chk=(word_query['frq']>=frq_threshold) if word_query['bnc']>0 else frq_default
    # check word length
    length_chk=len(word_query['word']) >= word_judge['word_length']
    return ((tag_chk+collins_chk+bnc_chk+frq_chk) >=3 or length_chk)

def identify_rare_words(text, word_judge):
    words = re.findall(r'\b\w+\b', text)
    sdict=DictCsv("ecdict.csv")
    rare_words=[]
    for word in words:
        if ((word.isdigit()) or
            (word.lower() in exclude_words) or
            (len(word)<=2)
            ):
            continue

        word_query=sdict.query(word) if sdict.query(word) else sdict.query('unknown')   
        if word_unknown(word_query, word_judge):
            rare_words.append(word)
    return rare_words

def clean_result(result,N=10):
    # 去除括号内的内容
    result = re.sub(r'\([^)]*\)', '', result)
    # 去掉标点和空格
    result = re.sub(r'[\s.,。，！]+', '', result)
    # 如果是x, 如果长度大于N个汉字，或者显示无法翻译，则返回空字符串
    if ((result.lower() == "x") or 
       (len(result) > N) or 
       ("无法翻译" in result)
       ):
        return ""
    return result

def translate_word(word, context, target_language="Chinese"):
    prompt = f"""
    Please give me the MOST APPROPRIATE {target_language} meaning of the word '{word}' IN this sentence: 
    ----
    {context}
    ----
    The response must be AS CONCISE AS POSSIBLE and NOT include any pinyin. 
    The response must be in the form of a single word or a short phrase.
    The response must be in {target_language}.
    The response must only be the meaning of the word in the sentence, not the entire sentence's meaning.
    If unable to translate, please return 'x'.
    """
    try:
        result = query_gpt3(prompt)
    except:
        print("GPT-3 API error, retrying in 30 seconds...")
        time.sleep(30)
        result = query_gpt3(prompt)
    result=clean_result(result)
    print(f"Translation of '{word}': {result}")
    return result

def process_subtitle(text,word_judge,target_language):
    rare_words = identify_rare_words(text,word_judge)
    # print(f"Rare words: {rare_words}")
    # translations = {}
    annotated_text = text
    for word in rare_words:
        for i in range(5):
            # 一句话里如果有多个生词，则需要处理多次
            sentences_with_word_but_no_translate = re.findall(rf'.*{word}[^\(]+.*', annotated_text, flags=re.IGNORECASE)
            if len(sentences_with_word_but_no_translate) == 0:
                break
            sentence = sentences_with_word_but_no_translate[0]
            translation = translate_word(word, sentence, target_language)
            if translation == "":
                continue
            annotated_sentence = sentence
            annotated_sentence = annotated_sentence.replace(word, f"{word}({translation})")
            annotated_text = annotated_text.replace(sentence, annotated_sentence)

    return annotated_text

def create_parser():
    parser = argparse.ArgumentParser(description="A subtitle processing program that annotates rare English words with their Chinese translations.")
    parser.add_argument('-i', '--input', dest="input_filename", help="需要处理的英文字幕文件")
    parser.add_argument('-o', '--output', dest="output_filename", help='输出文件')
    parser.add_argument('-include', nargs='?', dest="include_tag", type=str,
                        help='生词的定义: 包含哪些标记, 用空格隔开, 例如 cet6 toelf gre ielts',
                       default="cet6 gre ielts")
    parser.add_argument('-exclude', nargs='?', dest='exclude_tag', type=str,
                        help='生词的定义: 除外哪些标记, 用空格隔开, 例如 zk gk cet4',
                       default="zk gk cet4")
    parser.add_argument('-collins', nargs='?',dest='collins_threshold', type=int, 
                        help='collins星级', default=2)
    parser.add_argument('-bnc', nargs='?', dest='bnc_threshold', type=int,
                       help='英国国家语料库词频顺序bnc, 越大越难', default=5000)
    parser.add_argument('-frq', nargs='?', dest='frq_threshold', type=int,
                       help='当代语料库词频顺序frq, 越大越难', default=5000)
    parser.add_argument('-e', '--exclude_word', nargs='?', dest='exclude_word_filename', 
                        type=str, help='需要排除的单词列表, txt文件, 每行一个单词', 
                        default="exclude_word_list.txt")
    parser.add_argument('-l', '--word_length', nargs='?', dest='word_length', 
                        type=int, help='一定长度以上的单词将默认提示', 
                        default=10)
    parser.add_argument('-t', '--target_language', nargs='?', dest='target_language',
                        type=str, help='目标语言, 默认为中文', default="Simplified Chinese")
    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()
    word_judge={}
    word_judge["include_tag"]=args.include_tag 
    word_judge["exclude_tag"]=args.exclude_tag 
    word_judge["collins_threshold"]=args.collins_threshold 
    word_judge["bnc_threshold"]=args.bnc_threshold 
    word_judge['frq_threshold']=args.frq_threshold 
    word_judge['exclude_word_filename']=args.exclude_word_filename
    word_judge['word_length']=args.word_length

    with open(args.input_filename, "r", encoding="utf-8") as f:
        input_subtitle = f.read()

    output_subtitle = process_subtitle(input_subtitle,word_judge,args.target_language)

    with open(args.output_filename, "w", encoding="utf-8") as f:
        f.write(output_subtitle)

    print(f"Processed subtitle saved to {args.output_filename}")

if __name__ == "__main__":
    # Set your GPT-3 API key
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    main()
