from helper import Helper
import openai

# c0104.blogger - use OpenAPI chat create blog paragraphs from given titles
class Blogger:
    @staticmethod
    def generate_answer(test_data):
        words_for_titles = str(test_data.json()["blog"])
        answers = []
        # call openai moderation api
        openai.api_key = Helper().get_openapi_key()

        ai_resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 # "content": "pizza margherita  blog entry 4 titles from given words, no deviation"},
                 # "content": "pizza margherita  blog 4 short paragraphs  (to 50 words) containing exact given titles"},
                 "content": "write separate paragraphs for each title, one short paragraph for each title"},
                {"role": "user",
                 "content": words_for_titles}
            ])
        json_result = ai_resp.choices[0].message.content

        # return array of answers like ["1","0","0","1"]
        result_list = json_result.splitlines()
        result_list = [i for i in result_list if i] # remove''
        print(result_list)
        return result_list[1:8:2]  # wycinanie co 2., ponieważ format odpowiedzi: ["title", "paragraph", "title", "paragraph" itd.]


if __name__ == '__main__':
    test_data = Helper.create_simulated_response(
        b'{"blog":["Wstep: kilka slow na temat historii pizzy", "Niezbdne skladniki na pizze", "Robienie pizzy", "Pieczenie pizzy w piekarniku"]}')

    ans = Blogger().generate_answer(test_data)
    print(ans)


# """blogger here
# ###
# rules:
# ###
# -i create a blog entry about pizza margherita containing 4 titles
# -list titles no deviation"""
