import openai
from openai_key import api_key

openai.api_key = api_key

news_content = "神奈川県鎌倉市で「バリアフリービーチ」というイベントがありました。鎌倉市の医者などが、車いすを使っている人などにも海の中に入って楽しんでもらいたいと考えました。車いすの人と家族などの44のグループが参加しました。たくさんのボランティアが手伝いました。89歳の男性は、孫に手伝ってもらって、特別な車いすで海に入って楽しんでいました。男性は「海の水が気持ちよかったです」と話していました。1歳の女の子も家族と参加しました。女の子の病院の先生と看護師も一緒です。女の子は、水や砂に触ったり、お母さんと一緒に車いすで海に入ったりして、初めての海を楽しみました。お母さんは「娘は少しびっくりしていたようですが、夏のいい思い出になりました」と話していました。"

student_answer = "看護師（かんごし）、病院（びういん）、太平洋（たいへいよう）"

response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                        messages=[
                                            {"role": "system",
                                             "content": f"You are a Japanese language expert. Your task is to verify the furigana readings of given kanji words extracted from a news article and give a grade. For each kanji word, check if the provided furigana is correct based on the context of the article. If the kanji word is not from the news article, mark as incorrect. Grade the student based on the number of correct readings they provide out of a total of three. The article is as follows: {news_content} The student's answers are: {student_answer}"
                                             },],
                                        temperature=0,
                                        )

grade = response['choices'][0]['message']['content']  # type: ignore
print(grade)
