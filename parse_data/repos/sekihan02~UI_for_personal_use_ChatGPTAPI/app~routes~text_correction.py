import os
from flask import Blueprint, request, jsonify, render_template
import chardet
import openai
from dotenv import load_dotenv
load_dotenv()

text_correction_bp = Blueprint('text_correction', __name__, url_prefix='/text_correction')

@text_correction_bp.route('/')
def text_correction():
    return render_template('text_correction.html')

@text_correction_bp.route('/detect_encoding', methods=['POST'])
def detect_encoding():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file provided'}), 400
    
    file_bytes = file.read()
    detected = chardet.detect(file_bytes)
    encoding = detected.get('encoding', 'utf-8')  # utf-8 をデフォルトとして使用
    
    # バイト列をデコードしてテキストを得る
    text = file_bytes.decode(encoding, errors='replace')
    
    # テキストもレスポンスに含める
    return jsonify({'encoding': encoding, 'text': text})

@text_correction_bp.route('/get_corrections', methods=['POST'])
def get_corrections():
    data = request.get_json()
    selected_text = data.get('text', '')
    selected_text_len = len(selected_text)
    
    
    # Get the response from the GPT model
    openai.api_key = os.getenv("OPENAI_API_KEY")

    messages = [
        {"role" : "user", "content" : create_prompt(selected_text, selected_text_len)}
    ]
    text_list = []
    for i in range(3):
        # APIにリクエストを送信する
        # token数超過などの例外を返す可能性があるため、try/exceptを設定しておく。
        try:
            # response = openai.Completion.create(
            #     model="gpt-3.5-turbo-instruct",
            #     prompt=messages,

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                # model="gpt-4-0613",
                messages=messages,
                temperature=0.8,
                # max_tokens=800,  # 生成するトークンの最大数（入力と合算して4097以内に収める必要あり）
                # n=1,  # 生成するレスポンスの数
                stop=None,  # 停止トークンの設定
                top_p=1,  # トークン選択時の確率閾値
            )

            # 生成するレスポンス数は1という前提
            # Chain of Densityの結果を一まとめにする
            # Apply the function to the sample incomplete JSON string 
            sample_incomplete_str = response["choices"][0]["message"]["content"]
            # sample_incomplete_str = response["choices"][0]["text"]
            # print(sample_incomplete_str)
            
            messages_res = [
                {"role" : "system",
                "content" : "出力は必ず日本語で出力してください。簡潔な一文章を生成してください。"},
                {"role" : "user", "content" : f"{sample_incomplete_str}, {selected_text_len}" + "\n上記の'selected_text'を簡潔な一つの文章にまとめ、日本語で生成してください。生成する文章の長さは上記の'selected_text_len'の長さ±10文字前後でまとめてください。あなたならできます。頑張って"}
            ]
            response_summary = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                # model="gpt-4-0613",
                messages=messages_res,
                temperature=0.8,
                n=1,  # 生成するレスポンスの数
                stop=None,  # 停止トークンの設定
                top_p=1,  # トークン選択時の確率閾値
            )
            res_incomplete_str = response_summary["choices"][0]["message"]["content"]
            
            text_list.append(res_incomplete_str)
            
        except openai.error.InvalidRequestError as e:
            print("OpenAI API error occurred: " + str(e))
            text_list.append("OpenAI API error occurred: " + str(e))

    # return jsonify({'corrections': corrections})
    return jsonify({'corrections': text_list})


# Chain of Density用プロンプトフォーマット
def create_prompt(text, text_length) :
    return f"""
Article: {text}
text length: {text_length}

You generate more and more concise, substance-rich revised sentences of the above text.

Repeat the following two steps five times.

Identify 1-3 useful entities ("、", "。") from the article that are missing from the previously generated corrected text. delimited) from the above text.
Step 2. Write a new, denser summary of about the same length as the "text length" above, or ±10 characters, covering all entities and details of the previous revised text, plus any missing entities.

The missing entities are
- Relevant: relevant to the main story.
- Specific: descriptive yet concise (5 words or less).
- Novel: not present in the previous summary.
- Faithful: present in the article.
- Everywhere: present anywhere in the article.

Guidelines
- The first revised sentence should be long (the length of the above TEXT LENGTH word), but very nonspecific and contain little information beyond the entity marked as missing. To reach the above TEXT LENGTH word, not use overly verbose expressions and fillers (e.g., "This article discusses").
- Make every word count: rewrite the previous summary to improve flow and make room for additional entities.
- Create space by fusing, compressing, and using phrases with fewer words.
- The summary should be dense and concise, yet self-contained.
- Missing entities may appear anywhere in the new text.
- Entities should not be removed from the previous text. If space is not available, fewer new entities should be added. Remember to use exactly the same number of words in each summary.
最後に出力は必ず日本語で出力してください。
"""
