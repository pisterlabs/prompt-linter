import cohere

txt_lst = ["hello world", "안녕하세요"]
co = cohere.Client('')
response = co.detect_language(texts=txt_lst)

language_names = [lang.language_name for lang in response.results]
print(language_names)