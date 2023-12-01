# %% Activate API
from __future__ import annotations
import leetcode
from time import sleep
import json
import leetcode.auth
import openai
from bs4 import BeautifulSoup

def setup():

    with open('accounts_info.json') as f:
        info = json.load(f)

    leetcode_session = info["leetcode_session"]
    # csrf_token = info["csrf_token"]
    openai.api_key = info["openai_key"]
    
    csrf_token = leetcode.auth.get_csrf_cookie(leetcode_session)

    configuration = leetcode.Configuration()

    configuration.api_key["x-csrftoken"] = csrf_token
    configuration.api_key["csrftoken"] = csrf_token
    configuration.api_key["LEETCODE_SESSION"] = leetcode_session
    configuration.api_key["Referer"] = "https://leetcode.com"
    configuration.debug = False

    api_instance = leetcode.DefaultApi(leetcode.ApiClient(configuration))
    
    return api_instance


# %% Status Checking
def status_check(api_instance):
    """
    Check leetcode API status
    :return:
    """
    graphql_request = leetcode.GraphqlQuery(
        query="""
    {
        user {
        username
        isCurrentUserPremium
        }
    }
    """,
        variables=leetcode.GraphqlQueryVariables(),
    )
    print(api_instance.graphql_post(body=graphql_request))

# %% Only Test given test case
def test_submission(api_instance, id: int, code: str, test_case: str, lang="python"):
    """
    Test submission, will not be recorded
    :param id: question id
    :param code: code
    :param lang: code language
    :return: status
    """
    test_data = leetcode.TestSubmission(
        data_input=test_case,
        typed_code=code,
        question_id=id,
        test_mode=False,
        lang=lang,
    )

    interpretation_id = api_instance.problems_problem_interpret_solution_post(
        problem="two-sum", body=test_data
    )

    print("Test has been queued. ID:")
    print(interpretation_id)

    result = None
    # 5 Second While loop waiting for respond
    while not result or result['state'] == "STARTED" or result['state'] == "PENDING":
        sleep(5)
        result = api_instance.submissions_detail_id_check_get(
        id=interpretation_id.interpret_id
    )
    return result


#%% Real submission to LeetCode, submission with be record
def submission(api_instance, id: int, code: str, lang="python"):
    """
    Real submission, will be recorded to leetcode account
    :param id: question id
    :param code: code
    :param lang: code language
    :return: status
    """
    submission = leetcode.Submission(
        judge_type="large",
        typed_code=code,
        question_id=id,
        test_mode=False,
        lang=lang,
    )

    interpretation_id = api_instance.problems_problem_submit_post(
        problem="two-sum", body=submission
    )

    print("Submission has been queued. ID:")
    print(interpretation_id)
    result = None
    # 5 Second While loop waiting for respond
    while not result or result['state'] == "STARTED" or result['state'] == "PENDING":
        sleep(5)
        result = api_instance.submissions_detail_id_check_get(
            id=interpretation_id.submission_id
        )
    return result

#%% Get Question Detail
def get_problem_list(api_instance, problem="algorithms"):
    """
    get the list of problem
    :param id: question id
    :param code: code
    :param lang: code language
    :return: status
    """
    # query = leetcode.GraphqlQueryGetQuestionDetailVariables(title_slug=problem)
    api_response = api_instance.api_problems_topic_get(topic=problem)

    return api_response

#%% Get Question Detail
def get_problem(api_instance, problem="two-sum", lang="Python"):
    graphql_request = leetcode.GraphqlQuery(
        query="""
                query getQuestionDetail($titleSlug: String!) {
                question(titleSlug: $titleSlug) {
                    questionId
                    questionFrontendId
                    boundTopicId
                    title
                    content
                    translatedTitle
                    isPaidOnly
                    difficulty
                    likes
                    dislikes
                    isLiked
                    similarQuestions
                    contributors {
                    username
                    profileUrl
                    avatarUrl
                    __typename
                    }
                    langToValidPlayground
                    topicTags {
                    name
                    slug
                    translatedName
                    __typename
                    }
                    codeSnippets {
                    lang
                    langSlug
                    code
                    __typename
                    }
                    stats
                    codeDefinition
                    hints
                    solution {
                    id
                    canSeeDetail
                    __typename
                    }
                    status
                    sampleTestCase
                    enableRunCode
                    metaData
                    translatedContent
                    judgerAvailable
                    judgeType
                    mysqlSchemas
                    enableTestMode
                    envInfo
                    __typename
                }
                }
            """,
        variables=leetcode.GraphqlQueryGetQuestionDetailVariables(title_slug=problem),
        operation_name="getQuestionDetail",
    )

    api_response = api_instance.graphql_post(body=graphql_request)
    
    question_str = "Question ----------------------------\n"
    soup = BeautifulSoup(api_response.data.question.content, "html.parser")
    question_str += soup.get_text()
    question_str += "\nInitial Code ------------------------\n"
    question_str += api_response.data.question.code_snippets[2].code
    
    hint = ""
    if hasattr(api_response.data.question, "hint"):
        question_str += '\nHint\n'
        question_str += api_response.data.question.hint
        hint = api_response.data.question.hint
    
    return str(soup), api_response.data.question.code_snippets[2].code, hint, api_response.data.question.stats

#%%
def chatgpt_response(problem_content, question_code, question_hint, messages):
    message = "Just write python codes to answer the following question without any explaination or example cases:\n" + problem_content
    message += ("\nWrite all under the following code module:\n" + question_code)
    if question_hint:
        message += ("\n The hints are shown below:\n" + question_hint)
        
    if message:
        messages.append(
            {"role": "user", "content": message},
        )
        chat_completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )

    reply = chat_completion.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})
    return reply

#%%

def main():
    print("Check leetcode API status ------------------------\n")
    # Setup all configurations
    api_instance = setup()
    status_check(api_instance)
    
    # Get Question List
    problem_list = get_problem_list(api_instance)
    
    i = 1 # Selected problem index in the list
    # Select one of question (can use for loop to loop all questions)
    problem_slug = problem_list.stat_status_pairs[i].stat.question__title_slug
    problem_id = problem_list.stat_status_pairs[i].stat.question_id
    problem_content, question_code, question_hint, question_status = get_problem(api_instance, problem_slug)
    print("\nCode Problem ------------------------------------\n")
    print("Problem Name:", problem_list.stat_status_pairs[i].stat.question__title)
    print("Problem id in Leetcode Website:", problem_list.stat_status_pairs[i].stat.frontend_question_id)
    print("Problem id in our code:", problem_id)
    print("Problem Status:", question_status)
    
    # Get ChatGPT response
    # system message first, it helps set the behavior of the assistant
    all_problems_and_responses = [{"role": "system", "content": "Let's do some coding questions!"}]
    chatgpt_reply = chatgpt_response(problem_content, question_code, question_hint, all_problems_and_responses)
    print("\nChatGPT Response --------------------------------\n")
    print(chatgpt_reply)
    
    # Submit the response code to leetcode
    print("\nSubmission Result -------------------------------\n")
    submission_result = submission(api_instance, problem_id, chatgpt_reply, lang="python")
    print(submission_result)

if __name__ == '__main__': 
    main()




#%%
code = """
class Solution:
    def twoSum(self, nums, target):
        record = {}
        for i, n in enumerate(nums):
            if target - n in record.keys():
                # return [1]
                return [record[target - n], i]
            record[n] = i
"""
test_case = "[2,7,11,15]\n9"
lang = "python"
api_instance = setup()
status_check(api_instance)
submission_result = submission(api_instance, 1, code, lang="python")
#%%
