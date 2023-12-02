from vsslite import LangChainVSSLiteClient

vss = LangChainVSSLiteClient("http://127.0.0.1:8000")
# 0. Delete all data
vss.delete_all()

# 1. Open OpenAI terms of use: https://openai.com/policies/terms-of-use
# 2. Copy from "1. Registration and Access" to the end of contents
# 3. Save as openai_terms.txt

# 4. Upload
vss.upload("openai_terms.txt", namespace="openai")

# 5. Test
print(vss.search("Is there an age restriction for using ChatGPT?", namespace="openai"))
