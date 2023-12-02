import os

from dotenv import load_dotenv
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

load_dotenv('./.env')


class Summarizer:
    """
    入力された複数の文書のそれぞれのエッセンスとなる要素を
    逐次要約して加味しながら最終的な要約文を作成するクラス
    """
    def __init__(self):
        """
        コンストラクタ。インスタンスを初期化し、inputされた文書の要約を行うテンプレートと
        要約文同士を対照して追加すべきエッセンスを逐次加えていく指示を定義したテンプレートを作成している。
        また、この二つのテンプレートを用いて要約を行うチェーンを定義する。
        """
        # 入力された文章の要約を指示するテンプレート
        self.input_prompt = PromptTemplate(template=self.prompt_template,
                                           input_variables=["text"])
        # input_promptで作成した要約を元に、追加すべきエッセンスを逐次加えていく指示を定義したテンプレート
        # 保持している作成済み要約文章と新たに追加される文章の二つを引数とする
        self.refine_prompt = PromptTemplate(template=self.refine_template,
                                            input_variables=["existing_answer", "text"])
        # 上記を用いて要約を行うチェーンを定義
        # StreamingStdOutCallbackHandlerを用いて標準出力に逐次結果を表示するように設定
        self.chain = load_summarize_chain(llm=ChatOpenAI(temperature=0,
                                                         openai_api_key=os.environ.get("OPENAI_API_KEY"),
                                                         model_name="gpt-3.5-turbo-0613",
                                                         streaming=True,
                                                         callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])),
                                          chain_type="refine",  # 要約モードは３種類用意されている。今回はrefineモードを使用：要約文の逐次改善に重点がある
                                          verbose=None,  # これをTrueにすると実行中の詳細なログが表示される
                                          return_intermediate_steps=False,  # 中間生成物を出力するか否か。Trueにすると、要約の途中経過が出力される
                                          question_prompt=self.input_prompt,  # 要約を指示するテンプレート
                                          refine_prompt=self.refine_prompt)  # 要素追加と要約改善を行うテンプレート

    def summarize_text(self, texts):
        # 入力されたテキストをドキュメント化
        self.text = [Document(page_content=str(t)) for t in texts]
        # ドキュメント化したデータから問題を引き起こす文字列を除外したうえでチェーンに渡す。
        # "input_documents"はrefineモードでは固定の変数名
        response = self.chain(inputs={"input_documents": self.clean_string(self.text)},
                              return_only_outputs=True)
        return response

    def clean_string(self, input):
        """入力された文字列から問題を引き起こす文字列を除外する。

        Args:
            input (str or list or dict): 入力文字列

        Returns:
            str or list or dict: 問題を引き起こす文字列を除外した文字列
        """
        if isinstance(input, list):
            return [self.clean_string(element) for element in input]
        elif isinstance(input, dict):
            return {key: self.clean_string(value) for key, value in input.items()}
        elif isinstance(input, str):
            return input.replace('\u3000', ' ').replace('\n', ' ')
        else:
            return input

    @property
    def prompt_template(self):
        # 入力された文章の要約を指示するテンプレート
        # コード上の見た目は汚いインデントになるが、プロンプトに余計な空白を入れないため左に寄せている。
        return """
Please write a concise summary of the following `INPUT TEXT` in Japanese.
Format Example:```
【活動内容と成果の実績】
・モブプログラミングを実施した
・スクラム運営方針を決定した
・月次レビューの指摘事項について振り返りを行った
【課題と解決策】
・スケジュール遅延が生じている
・気軽にヘルプを求められていない
・評価指標が未作成である
【できごと・気づき】
・モブプロにより他のメンバーとの知見共有がスムーズになった
・輪読会での学びがスプリントの設計に活用できた
・スクラムの精神を学び失敗しても大丈夫な雰囲気の醸成ができた
```

INPUT TEXT:```
{text}
```
CONCISE SUMMARY IN JAPANESE:"""

    @property
    def refine_template(self):
        # 要素追加と要約改善を行うテンプレート
        # コード上の見た目は汚いインデントになるが、プロンプトに余計な空白を入れないため左に寄せている。
        return """
Your job is to produce a final summary.
We have provided an existing summary up to a certain point: {existing_answer}.
We have the opportunity to refine the existing summary (only if needed) with some more context below.
------------
{text}
------------
Given the new context, refine the original summary in Japanese.
In particular, correct the original summary if it is contrary to the facts stated in the new context.
If the context isn't useful, return the original summary.

Format Example of your output:```
【活動内容と成果の実績】
・モブプログラミングを実施した
・スクラム運営方針を決定した
・月次レビューの指摘事項について振り返りを行った
【課題と解決策】
・スケジュール遅延が生じている
・気軽にヘルプを求められていない
・評価指標が未作成である
【できごと・気づき】
・モブプロにより他のメンバーとの知見共有がスムーズになった
・輪読会での学びがスプリントの設計に活用できた
・スクラムの精神を学び失敗しても大丈夫な雰囲気の醸成ができた
```
"""


if __name__ == '__main__':
    # 動作確認用コード
    initial_text = """
- 5月のマンスリーレビュー資料のテーマと目的について
    - スクラムを通じた大規模言語モデルの活用方法とアジャイル開発の思想の学習および習得を主目的として進行する
    - スクラムを用いたプロダクト開発の実践を通じて、アジャイル開発の思想、大規模言語モデルの技術動向の把握および活用方法、ソフトウェアの開発技法を学習・習得する
- 制作予定物について
    - ストーリーを通じて、Botに命令することでNotionに散らばっている情報を抽出・要約し、2023年の5月分のマンスリーレビュー用の資料を作成する
    - 実現したいゴールは、複数のサービス（Notion、Slack、Googleドキュメント等）から当月の情報を抽出・要約し、マンスリーレビュー用の資料を返すSlack Botを作成すること
- PBLメンバーが抱えている課題について
    - 情報がドキュメントや議事録やタスク表に散らばっており整理に手間がかかる
    - 重複した情報が存在することで、先祖帰りが発生する可能性がある
    - Notion関連のドキュメントにまとめきれていない有益な情報を見落とす可能性がある
    - Google Meetで音声で打ち合わせた内容も含めて検索できるようになりたい
- 解決案について
    - Botに日付と簡単な指令を入力することで、その月の活動の概要を整理した内容をBotが返してくれる
- 活動計画と実績について
    - 年間の活動計画と1Qの活動計画と実績について記載されている
    - アジャイル開発思想の学習・習得、大規模言語モデルの活用方法の習得、プロダクト開発、開発
    """

    texts = [
        initial_text,
        # 以下は、上記のテキストに対して、追加でinputされるテキスト。これらを加味した要約が作成される。
        """課題： マンスリーレビューやSAの提出、プロジェクト計画書提出など、様々な締切が高頻度にやってくるため、息の長い取り組みがしにくい状況にある""",
        """活動計画： モブプログラミングの位置付けを適宜行う作業ではなく、開発作業の主軸に位置付けることとした。""",
        """制作予定物： Cloud Run上で動作するようにする。""",
    ]

    smzr = Summarizer()
    response = smzr.summarize_text(texts)
    print('\n\nfinal output:\n', response['output_text'])
