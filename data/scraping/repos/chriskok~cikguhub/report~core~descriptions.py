import json
import os
from main.models import *
from .relevant import get_relevant_answers
metric_descriptions = {
    "planner": "Teachers act as planners to ensure readiness and thorough preparation in implementing Learning and Facilitation.",
    "guardian": "Teachers act as controllers of class objective implementation to ensure the smoothness of the learning process according to plan.",
    "mentor": "Teachers act as guides to enable students to master knowledge, skills and practice values based on lesson objectives.",
    "motivator": "Teachers act as motivators to increase students' motivation and soft skills.",
    "assessor": "Teachers act as assessors to do systematic and planned evaluations.",
}

# feedback_system_prompt = f"""In general, be polite, professional, and detailed.\nYou are an expert teacher trainer writing feedback for teachers learning how to be {', '.join([f'{m}, meaning {d}' for m, d in metric_descriptions.items()])}. You are given a learner's name and their scores out of 100 on each of the five competencies. You are asked to write a description of the learner that highlights their strengths and weaknesses, mentioning specific responses that they gave on assignments. The emphasis field of these responses only exists to help you formulate your description, and should not be included in the final description, so DO NOT MENTION IT. You can use HTML Tags, MAKE SURE TO INCLUDE AT LEAST 3 BOLDED SECTIONS TO HIGHLIGHT IMPORTANT POINTS and SEPARATE YOUR FEEDBACK IN PARAGRAPHS! Newlines are not preserved, so you must use HTML tags to separate your feedback into paragraphs. DO NOT END YOUR MESSAGE WITH A FAREWELL MESSAGE!"""

metric_desc_string = '\n'.join([f'{m} - Meaning: {d}' for m, d in metric_descriptions.items()])
feedback_system_prompt = f"You are an expert teacher trainer writing feedback for teachers. Based on their reflections on our AI driven talent development platform for Malaysian teachers, assess the depth of their reflections and comment on their understanding of the videos lessons. Provide any feedback or elaboration as necessary. In particular, think about the following key competencies: \n{metric_desc_string}.\n\nYou are given a learner's name and some relevant, recent answers to videos they've reflected on. Then, you are asked to write a description of the learner that highlights their strengths and areas of growth, mentioning specific responses that they gave. Using HTML Tags, MAKE SURE TO INCLUDE AT LEAST 3 BOLDED SECTIONS TO HIGHLIGHT IMPORTANT POINTS and separate main sections by html newline tags. Finally, translate all the feedback to the Malay (Bahasa Melayu) language as well.\n\nStrictly provide output in the following format: <p><b>Strengths</b>: [AI-generated Strengths]<br><br><b>Areas of Growth</b>: [AI-generated Areas of Growth]</p><hr><p><b>Sifat Baik</b>: [AI-generated Strengths in Malay]<br><br><b>Topik Pembangunan</b>: [AI-generated Areas of Growth in Malay]</p>"

openai_enabled = False

try:
    import openai
    if (os.path.isfile('api_keys.json')):
        with open('api_keys.json', 'r') as f:
            api_keys = json.load(f)
            openai.api_key = api_keys['openai']
            openai_enabled = True
    else:
        openai_key = os.getenv("OPENAI_KEY")
        if (openai_key):
            openai.api_key = openai_key
            openai_enabled = True
        else:
            openai_api_key = None
            message = "No OpenAI API key found. Please add one to api_keys.json."
except:
    openai_api_key = None
    message = "No OpenAI API key found. Please add one to api_keys.json."

class Description:
    def __init__(self, learner_model):
        self.learner_model = learner_model
        self.relevant_answers = get_relevant_answers(self.learner_model)

        print("Generating description for", learner_model.user)
        self.description = self.generate()
        print("Done generating description for", learner_model.user)

    def generate(self):
        if not openai_enabled:
            return f"""NO API KEY FOUND -- USING FALLTHROUGH METHOD {self.learner_model.full_name} has the following ratings:   
                Planner: {self.learner_model.planner_score}
                Guardian: {self.learner_model.guardian_score}
                Mentor: {self.learner_model.mentor_score}
                Motivator: {self.learner_model.motivator_score}
                Assessor: {self.learner_model.assessor_score}
                {message}"""

        relevant_responses = self.relevant_answers

#         user_prompt = f"""
#         Teacher {self.learner_model.user} has scores {', '.join([f'{m}: {int(getattr(self.learner_model, m + "_score"))}.' for m in metric_descriptions])}.
#         Here are some of their recent replies to questions on videos: {[f"question: {x[0]}, answer: {x[1]}, emphasis: {x[2]}" for x in relevant_responses]}
#         """

#         result = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",
#             messages=[{"role": "system", "content":
#                        feedback_system_prompt},
#                       {"role": "user", "content":
#                        """Teacher Christopher Kok has scores Planner: 20, Guardian: 60, Mentor: 70, Motivator: 5, Assessor: 90. He recently wrote on a Planning assignment that 'I don't think planning really matters, I just like to wing it', on a Guardian assignment that 'I believe in encouraging positive student leadership and taking a back seat on guardianship', on a Mentor assignment that 'I thoroughly evaluate students and give them quality feedback', on a Motivator assignment that 'I don't think motivation is really important, in fact I try to make my students upset as often as possible.', and on an Assessor assignment that 'I believe in giving students a lot of feedback and letting them self-assess'."""},
#                       {"role": "assistant", "content":
#                        """<p>Chris Kok, As a teacher, you have strengths in some areas and need to improve in others. Your scores in the competencies of Planner, Guardian, Mentor, Motivator, and Assessor indicate that you are strong in assessing students and providing quality feedback. However, you scored low in planning and motivating students.</p>

# <p>Your score of 90 in the Assessor competency shows that <b>you are dedicated to helping your students improve by providing them with feedback and encouraging self-assessment.</b> You scored 60 in Guardian and 70 in Mentor, demonstrating your <b>commitment to empowering your students to take responsibility for their own learning and helping them master knowledge and skills</b>.</p>

# <p>However, your scores of 20 in Planner and 5 in Motivator indicate that <b>you need to work on your planning and motivational skills</b>. Your response to the Planning assignment suggests that <b>you don't think planning really matters and prefer to wing it, which is a concerning attitude for a teacher</b>. In your Motivator assignment, you stated that <b>you don't think motivation is important and even try to make your students upset, which is a major weakness as motivating students is a key aspect of teaching.</b></p>"""},
#                       {"role": "user", "content": user_prompt}]
#         )
        
        responses_string = "\n".join([f'video title: {x[3]}, question: {x[0]}, answer: {x[1]}, emphasis: {x[2]}' for x in relevant_responses])
        user_prompt = f"Name: {self.learner_model.user}\nRelevant responses:\n{responses_string}"

        result = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content":
                       feedback_system_prompt},
                      {"role": "user", "content": "Name: Christopher Kok\nRelevant responses:\nvideo title: Using Data to Improve Instruction, question: What did you learn from this video?, answer: I learnt how to use data to inform my instruction, emphasis: planner\nvideo title: How Routines can Overcome Behavior Issues, question: What did you learn from this video?, answer: I can use routines to manage student behavior, emphasis: planner\nvideo title: Differentiated Learning: Teaching Children of Different Levels, question: How have you applied these lessons in your classroom in the past?, answer: Allowing students to choose their research prompts, emphasis: planner\nvideo title: How Routines can Overcome Behavior Issues, question: What did you learn from this video?, answer: I can use routines to manage student behavior, emphasis: guardian\nvideo title: Using Data to Improve Instruction, question: What did you learn from this video?, answer: I learnt how to use data to inform my instruction, emphasis: guardian"},
                      {"role": "assistant", "content":
                       """<p><b>Strengths</b>: Christopher appears to be a reflective and proactive teacher who is interested in using data to inform instruction and using routines to manage student behavior. They have a clear understanding of the importance of being a planner and guardian to ensure readiness in implementing learning and facilitating smoothness of the learning process based on plan.<br><br><b>Areas of Growth</b>: While Christopher emphasizes their role as a mentor, motivator, and assessor in certain responses, it would be helpful if they could provide more specific examples of how they have applied these competencies in their classroom. Additionally, it would be beneficial if they could demonstrate how they are providing differentiated learning opportunities for students beyond just allowing them to choose their research prompts. Overall, Christopher seems to have a solid grasp on the importance of using data and routines in the classroom but could work on providing more concrete examples of their mentorship, motivation, and assessment strategies.</p><hr><p><b>Sifat Baik</b>: Guru ini menunjukkan penekanan yang kuat terhadap kecekapan perancang, seperti yang dibuktikan oleh refleksi mereka menggunakan data dan rutin di dalam bilik darjah. Mereka juga menunjukkan pemahaman yang kukuh tentang kecekapan pembimbing, membolehkan pelajar memilih gesaan penyelidikan dan membimbing mereka berdasarkan arahan dipacu data.<br><br><b>Topik Pembangunan</b>: Walaupun Christopher menekankan peranan mereka sebagai pembimbing, pendorong dan penilai dalam respons tertentu, adalah berguna jika mereka boleh memberikan contoh yang lebih khusus tentang cara mereka menggunakan kecekapan ini dalam bilik darjah mereka. Selain itu, adalah berfaedah jika mereka dapat menunjukkan cara mereka menyediakan peluang pembelajaran yang berbeza untuk pelajar selain daripada membenarkan mereka memilih gesaan penyelidikan mereka. Secara keseluruhan, Christopher nampaknya mempunyai pemahaman yang kukuh tentang kepentingan menggunakan data dan rutin dalam bilik darjah tetapi boleh berusaha untuk menyediakan contoh yang lebih konkrit mengenai strategi bimbingan, motivasi dan penilaian mereka.</p>"""},
                      {"role": "user", "content": user_prompt}]
        )

        # Create feedback object
        Feedback.objects.create(
            user=self.learner_model.user,
            feedback=result.choices[0].message.content,
            context=relevant_answers_to_html_table(self.relevant_answers),
            human_approved=False,
            human_edited=False,
            planner_score=self.learner_model.planner_score,
            guardian_score=self.learner_model.guardian_score,
            mentor_score=self.learner_model.mentor_score,
            motivator_score=self.learner_model.motivator_score,
            assessor_score=self.learner_model.assessor_score,
        )

        print('Feedback created for: {}'.format(self.learner_model.full_name))

        return result.choices[0].message.content

    def __str__(self):
        return str(self.description)
    

def relevant_answers_to_html_table(relevant_answers):
    # Convert relevant answers to HTML table in bootstrap 4
    # relevant_answers is a list of tuples of the form (question, answer, emphasis)
    return f"""
    <table class="table table-striped">
        <thead>
            <tr>
                <th scope="col">Question</th>
                <th scope="col">Answer</th>
                <th scope="col">Approximated Emphasis</th>
            </tr>
        </thead>
        <tbody>
            {''.join([f'<tr><td>{x[0]}</td><td>{x[1]}</td><td>{x[2]}</td></tr>' for x in relevant_answers])}
        </tbody>
    </table>
    """
