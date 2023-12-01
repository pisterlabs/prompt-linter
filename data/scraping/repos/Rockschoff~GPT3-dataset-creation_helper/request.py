
def ApiCall():
    file1 = open("input_prompt.txt" , "r")

    content = file1.readlines()
    print(len(content))
    # print(content)
    final_content = []
    for i in range(len(content)):
        if content[i]==" \n":
            # print("empty line")
            pass
        else:
            final_content.append(content[i])

    prompt_content = ""
    for s in final_content:
        prompt_content = prompt_content + s

    # print(len(final_content))

    total_words = 0
    for string in final_content:
        total_words= total_words + len(string.split())

    print(f'The words count for the scrapped content is {total_words} and maximum allowable count is 1500 words')

    import pandas as pd

    df = pd.read_excel("Questions.xlsx")
    prompt_list = []
    for index , row in df.iterrows():
        # print(row[0])
        part1 = "Website Content :" + prompt_content + "\n###\n"
        part2 = "Question : " + row[0]
        total = part1 + part2 + "\n Answer:"
        prompt_list.append(total)

    import sys
    if(total_words > 1500):
        print("warninig: word limit exceeded")
        sys.exit()

    import openai;

    api_key = "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"

    openai.api_key = api_key

    #model : curie:ft-storyprocess-2021-09-12-11-59-52
    answers = []
    for i in range(len(prompt_list)):
        print(f"reqeusting of question{i+1}")
        res = openai.Completion.create(
            model = "curie:ft-storyprocess-2021-09-12-11-59-52",
            prompt=prompt_list[i],
            stop="#END#",
            max_tokens=100
        )
        answers.append(res["choices"][0]["text"])

    print(answers[:5])

    questions = list(df["Question"])
    # answers = [x for x in range(len(questions))]
    ans = pd.DataFrame({"question" : questions , "answer" : answers})

    ans.to_excel("Response.xlsx")

# file1 = open("input_prompt.txt" , "r")

# content = file1.readlines()
# print(len(content))
# # print(content)
# final_content = []
# for i in range(len(content)):
#     if content[i]==" \n":
#         # print("empty line")
#         pass
#     else:
#         final_content.append(content[i])

# prompt_content = ""
# for s in final_content:
#     prompt_content = prompt_content + s

# # print(len(final_content))

# total_words = 0
# for string in final_content:
#     total_words= total_words + len(string.split())

# print(f'The words count for the scrapped content is {total_words} and maximum allowable count is 1500 words')

# import pandas as pd

# df = pd.read_excel("Questions.xlsx")
# prompt_list = []
# for index , row in df.iterrows():
#     # print(row[0])
#     part1 = "Website Content :" + prompt_content + "\n###\n"
#     part2 = "Question : " + row[0]
#     total = part1 + part2 + "\n Answer:"
#     prompt_list.append(total)

# import sys
# if(total_words > 1500):
#     print("warninig: word limit exceeded")
#     sys.exit()

# import openai;

# api_key = "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"

# openai.api_key = api_key

# #model : curie:ft-storyprocess-2021-09-12-11-59-52
# answers = []
# for i in range(len(prompt_list)):
#     print(f"reqeusting of question{i+1}")
#     res = openai.Completion.create(
#         model = "curie:ft-storyprocess-2021-09-12-11-59-52",
#         prompt=prompt_list[i],
#         stop="#END#",
#         max_tokens=100
#     )
#     answers.append(res["choices"][0]["text"])

# print(answers[:5])

# questions = list(df["Question"])
# # answers = [x for x in range(len(questions))]
# ans = pd.DataFrame({"question" : questions , "answer" : answers})

# ans.to_excel("Response.xlsx")


# res = openai.Completion.create(
#   model = "curie:ft-storyprocess-2021-09-12-11-59-52",
#   prompt=prompt_list[5],
#   stop="#END#"
# )
# print(res)
# print(res["choices"][0]["text"])




