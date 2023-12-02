import openai
import streamlit as st

with st.echo(code_location="below"):
    st.title("Azureでの利用方法")
    openai.api_base = "https://oai.langcore.org/v1"
    openai.api_type = "azure"
    if "AZURE_OPENAI_API_KEY" not in st.secrets:
        openai.api_key = st.text_input("AzureのOpenAI APIキーを入力してください", type="password")
    else:
        openai.api_key = st.secrets["AZURE_OPENAI_API_KEY"]

    if "AZURE_OPENAI_API_VERSION" not in st.secrets:
        openai.api_version = st.text_input(
            "AzureのOpenAI APIバージョンを入力してください", placeholder="2023-07-01"
        )
    else:
        openai.api_version = st.secrets["AZURE_OPENAI_API_VERSION"]

    if "AZURE_DEPLOYMENT_ID" not in st.secrets:
        deployment_id = st.text_input(
            "AzureのデプロイメントIDを入力してください", placeholder="your-deployment-id"
        )
    else:
        deployment_id = st.secrets["AZURE_DEPLOYMENT_ID"]

    if "AZURE_OPENAI_API_HOST" not in st.secrets:
        host = st.text_input(
            "AzureのOpenAI APIホストを入力してください",
            placeholder="https://xxxxxxx.openai.azure.com",
        )
    else:
        host = st.secrets["AZURE_OPENAI_API_HOST"]

    if st.button("キャッチコピー生成"):
        with st.spinner("AIが考え中..."):
            request_body = {
                "deployment_id": deployment_id,
                "headers": {
                    "LangCore-OpenAI-Api-Base": host,
                },
                "messages": [
                    {"role": "user", "content": "最高なキャッチコピーを考えてください。"},
                ],
                "user": "山田太郎",
            }
            response_body = openai.ChatCompletion.create(**request_body)
            result = response_body.choices[0].message.content

            st.subheader("結果:")
            st.write(result)
