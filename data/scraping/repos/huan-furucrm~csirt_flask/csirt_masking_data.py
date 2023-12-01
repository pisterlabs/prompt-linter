from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from csirt_config_info import get_openai_config
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer.entities import OperatorConfig

import openai

openai.app_key = get_openai_config()
def call_completion_model(prompt: str, model: str = "text-davinci-003", max_tokens: int = 2048) -> str:
    """Creates a request for the OpenAI Completion service and returns the response.

    :param prompt: The prompt for the completion model
    :param model: OpenAI model name
    :param max_tokens: Model's max tokens parameter
    """

    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens
    )

    return response['choices'][0].text

def create_prompt_en(anonymized_text: str) -> str:
    """
    Create the prompt with instructions to GPT-3.

    :param anonymized_text: Text with placeholders instead of PII values, e.g. My name is <PERSON>.
    """

    prompt = f"""
    Your role is to create synthetic text based on de-identified text with placeholders instead of personally identifiable information.
    Replace the placeholders (e.g. , , {{DATE}}, {{ip_address}}) with fake values.

    Instructions:

    Use completely random numbers, so every digit is drawn between 0 and 9.
    Use realistic names that come from diverse genders, ethnicities and countries.
    If there are no placeholders, return the text as is and provide an answer.
    input: How do I change the limit on my credit card {{credit_card_number}}?
    output: How do I change the limit on my credit card 2539 3519 2345 1555?
    input: {anonymized_text}
    output:
    """
    return prompt

def create_prompt_ja(anonymized_text: str) -> str:
    """
    Create the prompt with instructions to GPT-3.

    :param anonymized_text: Text with placeholders instead of PII values, e.g. My name is <PERSON>.
    """

    prompt = f"""
    あなたの役割は、個人を特定できないようにプレースホルダーを使用して合成テキストを作成することです。
    （例：{{DATE}}、{{ip_address}}）プレースホルダーを適切な偽の値で置き換えます。
    
    指示：
    
    完全にランダムな数字を使用します。各数字は0から9の間で抽選します。
    多様な性別、民族、国から来るリアルな名前を使用します。
    プレースホルダーがない場合は、元のテキストをそのまま返し、回答を提供します。
    入力：クレジットカードの制限をどのように変更しますか{{credit_card_number}}？
    出力：クレジットカードの制限をどのように変更しますか2539 3519 2345 1555？
    入力：{anonymized_text}
    出力：
    """
    return prompt

def masking_data_with_star(text):
    operator = {"DEFAULT": OperatorConfig("mask", {"chars_to_mask": 40, "masking_char": "*", "from_end": True})}
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()

    results = analyzer.analyze(text, language="en")
    anonymized = anonymizer.anonymize(text=text, analyzer_results=results, operators=operator)
    anonymized_text = anonymized.text
    # print(anonymized_text)
    return anonymized_text
    
    
def masking_data(text, language="en"):
    
    # # Create configuration containing engine name and models
    # configuration = {
    #     "nlp_engine_name": "spacy",
    #     "models": [{"lang_code": "ja", "model_name": "ja_core_news_lg"}],
    # }

    # # Create NLP engine based on configuration
    # provider = NlpEngineProvider(nlp_configuration=configuration)
    # nlp_engine_with_spanish = provider.create_engine()
    # analyzer = AnalyzerEngine(nlp_engine = nlp_engine_with_spanish, supported_languages=["ja"])
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()
    results = analyzer.analyze(text, language=language)
    print(results)
    anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
    anonymized_text = anonymized.text
    if language == "ja":
        return call_completion_model(create_prompt_ja(anonymized_text))
    else:
        return call_completion_model(create_prompt_en(anonymized_text))
    
if __name__ == '__main__':
    input_text_en = """
    Subject: Urgent: Microsoft Edge Vulnerabilities Found on Remote Windows Host (CVE-2022-4436 to CVE-2022-4440)

    From: John Doe
    Email: john.doe123@emailprovider.com
    Phone: +1 (555) 123-4567
    
    Dear IT Team,
    
    I hope this email finds you well. My name is John Doe, and I am reaching out on behalf of the IT Security Department regarding a critical issue with the remote Windows host and Microsoft Edge.
    
    It has come to our attention that the version of Microsoft Edge installed on the remote host is older than 108.0.1462.54. As a result, it is susceptible to multiple vulnerabilities disclosed in an advisory on December 16, 2022.
    
    Here are the details of the vulnerabilities and their impact on the system:
    
    CVE-2022-4436 (Chromium Severity: High)
    
    Vulnerability: Use After Free in Google Chrome's Blink Media after memory release.
    Potential Risk: Remote attackers could exploit this issue via a crafted HTML page to manipulate heap corruption.
    CVE-2022-4437 (Chromium Severity: High)
    
    Vulnerability: Use After Free in Google Chrome's Mojo IPC after memory release.
    Potential Risk: Remote attackers could exploit this issue via a crafted HTML page to manipulate heap corruption.
    CVE-2022-4438 (Chromium Severity: High)
    
    Vulnerability: Use After Free in Google Chrome's Blink Frames after memory release.
    Potential Risk: Remote attackers could exploit this issue by tricking users into performing specific UI operations on a crafted HTML page, leading to heap corruption.
    CVE-2022-4439 (Chromium Severity: High)
    
    Vulnerability: Use After Free in Chrome OS on Google Chrome's Aura after memory release.
    Potential Risk: Remote attackers could exploit this issue by deceiving users into performing specific UI operations, leading to heap corruption.
    CVE-2022-4440 (Chromium Severity: Medium)
    
    Vulnerability: Use After Free in Google Chrome's Profiles after memory release.
    Potential Risk: Remote attackers could exploit this issue via a crafted HTML page to manipulate heap corruption.
    We would like to highlight that Nessus has not tested these vulnerabilities. Instead, it relies solely on the self-reported version numbers provided by the application.
    
    To ensure the security and integrity of our systems, we urgently request your team to take immediate action and update Microsoft Edge to a version that includes security patches addressing the mentioned vulnerabilities. This step is vital in safeguarding our network from potential remote attacks and protecting sensitive data.
    
    If you require any further information or assistance, please do not hesitate to contact us.
    
    Thank you for your prompt attention to this matter.
    
    Best regards,
    John Doe
    IT Security Department
    """
    input_text = """
    件名: 緊急：リモートの Windows ホストにおける Microsoft Edge の脆弱性 (CVE-2022-4436 から CVE-2022-4440)

    差出人: 山田太郎
    メールアドレス: yamada.taro123@emailprovider.com
    電話番号: +81 (0)12-3456-7890

    拝啓、ITチームの皆様

    お世話になっております。山田太郎と申します。ITセキュリティ部門を代表して、リモートのWindowsホストとMicrosoft Edgeにおける重要な問題について連絡いたします。

    私どもの確認によりますと、リモートのホストにインストールされているMicrosoft Edgeのバージョンが108.0.1462.54より古いものとなっております。そのため、2022年12月16日のアドバイザリに記載されている複数の脆弱性の影響を受ける可能性があるとのことです。

    以下に、脆弱性とそのシステムへの影響について詳細を記載いたします：

    CVE-2022-4436 (Chromiumセキュリティ深刻度: 高)

    脆弱性内容：Google ChromeのBlink Mediaにおけるメモリ解放後のUse After Free。
    潜在的なリスク：リモート攻撃者が細工されたHTMLページを介してヒープ破損を悪用する可能性があります。
    CVE-2022-4437 (Chromiumセキュリティ深刻度: 高)

    脆弱性内容：Google ChromeのMojo IPCにおけるメモリ解放後のUse After Free。
    潜在的なリスク：リモート攻撃者が細工されたHTMLページを介してヒープ破損を悪用する可能性があります。
    CVE-2022-4438 (Chromiumセキュリティ深刻度: 高)

    脆弱性内容：Google ChromeのBlink Framesにおけるメモリ解放後のUse After Free。
    潜在的なリスク：リモート攻撃者がユーザーに特定のUI操作を行わせ、細工したHTMLページを経由してヒープ破壊を悪用する可能性があります。
    CVE-2022-4439 (Chromiumセキュリティ深刻度: 高)

    脆弱性内容：Chrome OS上のGoogle ChromeのAuraにおけるメモリ解放後のUse After Free。
    潜在的なリスク：リモート攻撃者が特定のUI操作を行うようユーザーを騙し、特定のUI操作を介してヒープ破壊を悪用する可能性があります。
    CVE-2022-4440 (Chromiumセキュリティ深刻度: 中)

    脆弱性内容：Google ChromeのProfilesにおけるメモリ解放後のUse After Free。
    潜在的なリスク：リモート攻撃者が細工されたHTMLページを介してヒープ破損を悪用する可能性があります。
    Nessusはこれらの問題をテストしておらず、代わりにアプリケーションの自己報告されたバージョン番号にのみ依存しています。

    システムのセキュリティと機密性を確保するために、緊急の対応をお願いいたします。Microsoft Edgeをセキュリティパッチが適用された最新バージョンに更新していただくことが重要です。これにより、潜在的なリモート攻撃からネットワークを保護し、重要なデータを守ることができます。

    ご不明点やお手伝いが必要な場合は、どうぞお気軽にお問い合わせください。

    お手数をおかけいたしますが、ご対応をお願いいたします。

    敬具、

    山田太郎
    ITセキュリティ部門
    """
    # text = masking_data(input_text_en, "en")
    text = masking_data_with_star(input_text_en)
    print(text)