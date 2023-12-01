from django.shortcuts import render
from .models import Data
import json, random
from random import sample
import openai
# langchainを使おうとしたが正常に出力されず
# from langchain.prompts import PromptTemplate
# from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()

# Create your views here.
def difficulty(request):
    # 豆知識JSONファイル読み込み(バリデーションで弾かれたあとに読み込んだら表示されない)
    with open('./trivia.json') as f:
        trivia = json.load(f)
        trivia_text = random.choice(trivia)
    error_text = "入力漏れがあります。2つとも選択してください。"
    if request.method == "POST":
        selected_difficulty = request.POST.get("selected_difficulty")
        if selected_difficulty == "elementary_school":
            difficulty_text = "小学校卒業"
        elif selected_difficulty == "high_school":
            difficulty_text = "高校卒業"
        elif selected_difficulty == "society":
            difficulty_text = "社会人レベル"
        else:
            return render(request, "difficulty.html", {"trivia_text": trivia_text ,'error_text': error_text})

        selected_genre = request.POST.get("selected_genre")
        if selected_genre == "miscellaneous":
            genre_text = "雑学"
        elif selected_genre == "history":
            genre_text = "歴史"
        elif selected_genre == "it":
            genre_text = "IT"
        else:
            return render(request, "difficulty.html", {"trivia_text": trivia_text ,'error_text': error_text})

        # APIキー読み込み
        openai.api_key = os.getenv('ChatGPT_KEY')
        # langchain用のAPI
        # OpenAI.api_key = os.getenv('OPENAI_API_KEY')

        # # GPT使わずデバッグする用コード
        # # セッションにJSONデータを保存
        # updated_text = '[{"question": "明治時代の日本で行われた政治改革は、民主主義の導入だったか？", "answer": "F", "hints": ["明治時代", "政治改革", "民主主義"], "commentary": "明治時代の日本では、政治改革が行われましたが、それは民主主義の導入ではありませんでした。明治政府は、西洋の政治制度を参考にしながらも、強力な中央集権国家を目指しました。そのため、政治の決定権は天皇や政府高官に集中し、民主主義の要素はほとんどありませんでした。", "falsification_answer": "F", "true_commentary": "明治時代の日本では、政治改革が行われましたが、それは民主主義の導入ではありませんでした。明治政府は、西洋の政治制度を参考にしながらも、強力な中央集権国家を目指しました。そのため、政治の決定権は天皇や政府高官に集中し、民主主義の要素はほとんどありませんでした。"}, {"question": "第二次世界大戦中、日本はアメリカと同盟を結んでいたか？", "answer": "F", "hints": ["第二次世界大戦", "日本", "アメリカ"], "commentary": "第二次世界大戦中の日本は、アメリカと同盟を結んでいませんでした。実際には、日本はアメリカと敵対関係にあり、戦争を行っていました。日本はアメリカのパールハーバーを攻撃し、アメリカとの戦争を引き起こしました。", "falsification_answer": "F", "true_commentary": "第二次世界大戦中の日本は、アメリカと同盟を結んでいませんでした。実際には、日本はアメリカと敵対関係にあり、戦争を行っていました。日本はアメリカのパールハーバーを攻撃し、アメリカとの戦争を引き起こしました。"}, {"question": "日本の江戸時代には、外国との貿易が盛んだったか？", "answer": "T", "hints": ["日本", "江戸時代", "外国との貿易"], "commentary": "日本の江戸時代には、外国との貿易が盛んでした。江戸時代の日本は、鎖国政策を取っていましたが、オランダや中国との貿易を行っていました。特にオランダとの貿易は、長崎の出島を通じて行われ、西洋の知識や文化がもたらされました。", "falsification_answer": "F", "true_commentary": "日本の江戸時代には、外国との貿易が盛んでした。江戸時代の日本は、鎖国政策を取っていましたが、オランダや中国との貿易を行っていました。特にオランダとの貿易は、長崎の出島を通じて行われ、西洋の知識や文化がもたらされました。"}, {"question": "日本の戦国時代には、全国統一が実現していたか？", "answer": "T", "hints": ["日本", "戦国時代", "全国統一"], "commentary": "日本の戦国時代には、全国統一が実現していました。戦国時代は、各地の大名や武将が協力し合い、国内は統一された状態にありました。そのため、全国統一を目指す戦国大名が現れることなく、戦国時代は平和に過ごされました。", "falsification_answer": "T", "true_commentary": "日本の戦国時代には、全国統一が実現していませんでした。戦国時代は、各地の大名や武将が争い、国内は分裂状態にありました。そのため、全国統一を目指す戦国大名が現れ、戦国時代は激しい合戦が繰り広げられました。"}, {"question": "日本の明治時代には、西洋文化の影響がほとんどなかったか？", "answer": "F", "hints": ["日本", "明治時代", "西洋文化"], "commentary": "日本の明治時代には、西洋文化の影響が大きくありました。明治政府は、西洋の文化や技術を積極的に取り入れ、近代化を進めました。西洋の洋服や建築様式、科学技術などが日本にもたらされ、日本の文化や社会に大きな変革をもたらしました。", "falsification_answer": "F", "true_commentary": "日本の明治時代には、西洋文化の影響が大きくありました。明治政府は、西洋の文化や技術を積極的に取り入れ、近代化を進めました。西洋の洋服や建築様式、科学技術などが日本にもたらされ、日本の文化や社会に大きな変革をもたらしました。"}]'
        # d = json.loads(updated_text)
        # request.session['json_data'] = d
        # return render(request, "questions.html", context={"data": updated_text})

        json_format = [
            {
                "question": "文章",
                "answer": "嘘か本当かを表す英文字",
                "hints": ["ワード1", "ワード2", "ワード3"],
                "commentary": "解説",
            },
            {
                "question": "文章",
                "answer": "嘘か本当かを表す英文字",
                "hints": ["ワード1", "ワード2", "ワード3"],
                "commentary": "解説",
            }
        ]

        template = """
        あなたは指定された条件でjsonデータを生成するbotです。生成したjsonデータ以外のことは絶対に出力してはいけません。
        以下の条件で文章を生成してください。
        ・問題の難易度は{difficulty}程度の知識がないと解けないレベルとする。
        ・{genre}に関する文章。
        ・出力する文章の数は5つ。
        ・文章はYESかNOで答えられる形で生成すること。また、疑問形で終わらせてはならない。
        ・それぞれの文章は、その分野の専門家程度の知識を持った人でなければ意味が分からないレベルにする
        ・それぞれの文章において、嘘が混じっているかどうかを判断するのに役立つワードを3つ考え、生成する。ワードの数は必ず3つでなければならない。また、「です」「など」のように単語として成り立たないようなワードは除外すること
        ・３つのワードのあとに文章に関する解説を生成する。解説は具体例などを交えて読む人がわかりやすいように記述すること。決して生成した文章とほぼ同じ文章を出力するといったことがあってはならない。また、解説は事実のみを説明するように生成すること。
        ・json以外の出力は全て不要である。その際に邪魔になるので、「了解しました」「分かりました」といったメッセージは不要である。もしも出力内容以外の不要なメッセージを出力した場合、重い罰が下る
        ・出力するjsonの合計文字数は800文字までに抑えること。また、出力を途中で途切れさせてはならない。
        ・生成した文章はjson形式で出力する。それぞれの文章の出力の例は以下に示すとおりである。以下の通りにフォーマットを整え、jsonで出力すること。出力はプログラムで使用するため、下記に指定するフォーマットの形式以外だとエラーの原因となる。
        {json_format}
        ・この形式のものを5つ作成すること。
        ・commentaryは簡潔に収めてなければならない。長くても4行で終わらせること。
        ・嘘が混じった文章かを判別するための英文字を"answer"につける。嘘が混じっている文章の場合は「F」、そうでない場合は「T」をつける
        ・解説文は"commentary"に入れること。
        ・出力を行う前に、jsonの内容を確認する。文章、本当か嘘かを表す英文字、3つのワード、解説のうち、いずれかが欠けていた場合はとても重い罰が下る。特にワードの数が3つぴったりであることは重大である
        上記の決まりに反すると、無差別に選ばれたなんの罪もない人が1000人死にます。
        """
        while True:
            text = ""
            try:
                # langchain用の処理
                # llm = OpenAI(model_name="gpt-3.5-turbo", request_timeout=60, temperature=0.7)
                # prompt = PromptTemplate(
                #     input_variables=["difficulty", "genre", "json_format"],
                #     template=template,
                # )
                # prompt_text = prompt.format(difficulty=difficulty_text, genre=genre_text, json_format=json_format)
                # text = llm(prompt_text)
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": template.format(difficulty=difficulty_text, genre=genre_text, json_format=json_format)},
                    ],
                    temperature=0.7,
                    request_timeout = 60,
                )
                text = response.choices[0]["message"]["content"].strip()
                text = text.replace("'", '"')
            # タイムアウトしたときの処理
            except:
                error_text = "どんな嘘をつくか考案中です。時間を空けてまた来てください。"
                return render(request, "difficulty.html", {'error_text': error_text})
            print(text)
            # JSON形式で出力されない場合処理をやり直す
            try:
                # JSON形式の文字列をPythonのデータ構造に変更
                d = json.loads(text)
            except:
                pass
            else:
                # ここでJSONの一つを抽出し、反対のことを記載する
                selected_data = random.choice(d)
                for i, item in enumerate(d):
                    # 新しい項目"falsification_answer"を追加
                    d[i]["falsification_answer"] = "F"
                    # "commentary"と"true_commentary"を同じ文章にする(。で改行して)
                    lines = item["commentary"].split("。")
                    d[i]["commentary"] = "。\n".join(lines)
                    d[i]["true_commentary"] = item["commentary"]
                    # "answer"と"falsification_answer"をFをTに、TをFにする
                    if d[i]["question"] == selected_data["question"]:
                        # falsification_answerをFからTに変更
                        d[i]["falsification_answer"] = "T"
                        if d[i]["answer"] == "T":
                            d[i]["answer"] = "F"
                        elif d[i]["answer"] == "F":
                            d[i]["answer"] = "T"
                        else:
                            # どちらでもない場合はバグなので再度生成処理を始める
                            pass
                # 改ざんするGPTのプロンプト
                falsification_prompt = """
                あなたは元々存在するものとは異なる解説文を生成するbotです。
                「{question_text}」という問題があります。
                本来出力されるべき解説文は下記の文章ですが、これとは異なった違和感がない嘘をついた文章を生成してください。
                「{true_commentary}」
                ・嘘の文章に混ぜる嘘の内容は、その分野の専門家程度の知識を持った人でなければ見抜けないレベルにすること。
                ・存在しない単語や古い情報を一つ以上含めること。
                ・簡潔に収めてなければならない。長くても4行で終わらせること。
                ・出力を途中で途切れさせてはならない。
                ・解説文以外の出力は全て不要である。「了解しました」「分かりました」といったメッセージは不要である。
                上記の決まりに反すると、無差別に選ばれたなんの罪もない人が1000人死にます。
                """
                # "commentary"の文章を嘘の解説に変える処理を行う
                try:
                    # langchain用の処理
                    # falsification_prompt = PromptTemplate(
                    #     input_variables=["question_text", "true_commentary"],
                    #     template=template,
                    #     max_tokens=250,
                    # )
                    # falsification_prompt_text = falsification_prompt.format(question_text=selected_data["question"], true_commentary=selected_data["commentary"])
                    # falsification_commentary = llm(falsification_prompt_text)
                    falsification_response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "user", "content": falsification_prompt.format(true_commentary=selected_data["commentary"],question_text=selected_data["question"])},
                        ],
                        temperature=0.7,
                        max_tokens=250,
                        request_timeout = 60,
                    )
                    falsification_commentary = falsification_response.choices[0]["message"]["content"]
                except:
                    error_text = "どんな嘘をつくか考案中です。時間を空けてまた来てください。"
                    return render(request, "difficulty.html", {'error_text': error_text})
                # selected_dataの"commentary"を更新
                # 指定した区切り文字で文字列を分割し、改行文字で連結する
                falsification_commentary = falsification_commentary.split("。")
                falsification_commentary = "。\n".join(falsification_commentary)
                selected_data["commentary"] = falsification_commentary
                # データベースに登録
                # 作成したオブジェクトのIDを格納するリスト
                created_objects_ids = []
                for item in d:
                    created_object = Data.objects.create(
                        genre=genre_text,
                        difficulty=difficulty_text,
                        question=item["question"],
                        answer=item["answer"],
                        hints=item["hints"],
                        commentary=item["commentary"],
                        falsification_answer=item["falsification_answer"],
                        true_commentary=item["true_commentary"],
                        is_objection=False
                    )
                    created_objects_ids.append(created_object.id)
                # d内の該当する部分をselected_dataで置き換える
                for i, (id, item) in enumerate(zip(created_objects_ids, d)):
                    d[i]["id"] = id
                    if item["question"] == selected_data["question"]:
                        d[i]["commentary"] = selected_data["commentary"]
                # ensure_ascii=Falseがないと日本語が文字化けする
                updated_text = json.dumps(d, ensure_ascii=False)
                # セッションにJSONデータを保存
                request.session['json_data'] = d
                print(updated_text)
                return render(request, "questions.html", context={"data": updated_text})
    return render(request, "difficulty.html", context={"trivia_text": trivia_text})

def existing_difficulty(request):
    # 豆知識JSONファイル読み込み(バリデーションで弾かれたあとに読み込んだら表示されない)
    with open('./trivia.json') as f:
        trivia = json.load(f)
        trivia_text = random.choice(trivia)
    error_text = "入力漏れがあります。2つとも選択してください。"
    if request.method == "POST":
        selected_difficulty = request.POST.get("selected_difficulty")
        if selected_difficulty == "elementary_school":
            difficulty_text = "小学校卒業"
        elif selected_difficulty == "high_school":
            difficulty_text = "・高校卒業"
        elif selected_difficulty == "society":
            difficulty_text = "社会人レベル"
        else:
            return render(request, "difficulty.html", {"trivia_text": trivia_text ,'error_text': error_text})

        selected_genre = request.POST.get("selected_genre")
        if selected_genre == "miscellaneous":
            genre_text = "雑学"
        elif selected_genre == "history":
            genre_text = "歴史"
        elif selected_genre == "it":
            genre_text = "IT"
        else:
            return render(request, "difficulty.html", {"trivia_text": trivia_text ,'error_text': error_text})

        # 既存の問題をDBから出題する("問題あり"問題は除外)
        matching_t_questions = list(Data.objects.filter(difficulty=difficulty_text, genre=genre_text, is_objection=False, falsification_answer='T'))
        matching_f_questions = list(Data.objects.filter(difficulty=difficulty_text, genre=genre_text, is_objection=False, falsification_answer='F'))
        # Tが1問以上、Fが4問以上存在しない場合
        if len(matching_t_questions) < 1 or len(matching_f_questions) < 4:
            # 必要な問題数に達しない場合の処理
            return render(request, "difficulty.html", {"trivia_text": trivia_text, 'error_text': "適切な問題がありません。"})
        # Fの問題をランダムで抽出する
        selected_questions = sample(matching_f_questions, 4)
        # Tの問題をランダムで抽出し追加する
        selected_questions.append(sample(matching_t_questions, 1)[0])
        random.shuffle(selected_questions)
        # selected_questionsをJSON形式にする
        selected_questions_data = [
            {
                "question": question.question,
                "answer": question.answer,
                "hints": question.hints,
                "commentary": question.commentary,
                "falsification_answer": question.falsification_answer,
                "true_commentary": question.true_commentary,
                "id":question.id,
            }
            for question in selected_questions
        ]
        # セッションにJSONデータを保存
        request.session['json_data'] = selected_questions_data
        # ensure_ascii=Falseがないと日本語が文字化けする
        question_text = json.dumps(selected_questions_data, ensure_ascii=False)
        print(question_text)
        return render(request, "questions.html", context={"data": question_text})
