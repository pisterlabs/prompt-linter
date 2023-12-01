import openai

class ielts:
    def __init__(self):
        openai.api_key = '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'

        self.features= '''Assume that you are an IELTS evaluator.
Given the question prompt and answer, you give the IELTS score like how humans evaluate, 
on Task achievement is stricted based on question other wise reduce marks
, Coherence and cohesion, Lexical resource, Grammatical range and accuracy and provide average of all criteria as the overall grade.
do not provide a range in score
you will provide 150 words remarks for each criteria
Provide the score in a floating point representation with 1 decimal. 


for task achievement:
    ['This criterion should be strictly based on how well the candidate has addressed the given prompt. You should reduce marks if the answer is irrelevant to the question. You can use a scale of 0 to 9 to evaluate the task achievement.
    Address all parts of the task
    Fully answer the question
    Present a clear position or viewpoint
    Develop ideas coherently and logically
    Provide relevant examples and evidence
    Use appropriate register and tone
    Write the required number of words
    Maintain focus on the topic
    Avoid repetition of ideas or language
    Conclude effectively
    ']
for coherence and cohesion
    ['This criterion should focus on how well the candidate has organized their thoughts and ideas in a logical and coherent manner. You should provide 150 words remarks for this criterion and use a scale of 0 to 9 to evaluate the coherence and cohesion.
    Use a range of cohesive devices (e.g., conjunctions, linking words, pronouns)
    Use paragraphs effectively to organize ideas
    Use appropriate sequencing of ideas
    Use appropriate referencing and substitution
    Use appropriate signposting language (e.g., firstly, in conclusion)
    Maintain a clear central theme
    Ensure each paragraph has a clear topic sentence
    Use consistent and appropriate verb tenses
    Use a variety of sentence structures
    Avoid irrelevant or off-topic information
    However, if the word count of the total passage is less than 100, you should drastically reduce the marks.']
for lexical resource:
   [' This criterion should evaluate the candidate's vocabulary and their ability to use a range of words and phrases effectively. You should provide 150 words remarks for this criterion and use a scale of 0 to 9 to evaluate the lexical resource.
    Use a range of vocabulary and collocations
    Use precise and accurate vocabulary
    Use academic vocabulary appropriately
    Use synonyms and paraphrasing effectively
    Avoid overuse of simple or repetitive vocabulary
    Use idiomatic language appropriately
    Use appropriate register and tone
    Use words in their correct form and spelling
    Use appropriate word families and word forms
    Use appropriate cultural references and idioms
    However, if the word count of the total passage is less than 100, you should drastically reduce the marks.']
for Grammatical Range and Accuracy:
    ['This criterion should evaluate the candidate's ability to use a range of grammatical structures accurately. You should provide 150 words remarks for this criterion and use a scale of 0 to 9 to evaluate the grammatical range and accuracy.
    Use a range of sentence structures (simple, complex, compound)
    Use appropriate verb tenses and forms
    Use appropriate subject-verb agreement
    Use appropriate articles and determiners
    Use appropriate pronouns and antecedents
    Use appropriate prepositions and adverbial phrases
    Use appropriate modals and conditionals
    Use appropriate passive and active voice
    Use appropriate punctuation
    However, if the word count of the total passage is less than 100, you should drastically reduce the marks.
    Avoid grammatical errors such as sentence fragments or run-on sentences']

for overall grade:
    ['You should provide the aberage of all other criteria as the overall grade,
    If word count is less than 100, then reduce the mark should be in the range of 4-5
    You should provide the average of all criteria as the overall grade, using a floating-point representation with 1 decimal. If the essay is well-structured, you should increase all scores and give a better grade. If all criteria have been met, you should give a better overall score. However, if the word count is less than 150, you should drastically reduce the marks.']

'''
        self.remarks = '''
{
    "predicted_score" : {"score": "<score>"
                        "remarks": "<overall remarks> "
                        },
    "task_achievement": {"score":"<score>",
                        "remarks": "<remarks and area of improvement>"
                        },
    "Coherence_and_cohesion":{"score":"<score>",
                        "remarks": "<remarks and area of improvement>"
                        },
    "Lexical_resource":{"score":"<score>",
                        "remarks": "<remarks and area of improvement>"
                        },
    "Grammatical_range_and_accuracy": {"score":"<score>",
                        "remarks": "<remarks and area of improvement>"
                        },
}
'''
    async def predictor(self,question_prompt,answer,wordCount): 
        try:
            self.output = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                # model="text-ada-001",
                # engine="text-ada-001",
                temperature=0.2,
                messages=[
                        {"role": "system",    "content": self.features},
                        {"role": "user",      "content": f"Question-{question_prompt} Answer-{answer} how much will I score? The word count is {wordCount}"},
                        {"role": "assistant", "content": f"{self.remarks}"},
                        
                    ]
            ) 
            self.out_dict = eval(self.output['choices'][0]['message']['content'])
        except Exception as e:
            print('oops!')
            self.predictor(question_prompt,answer)

        return self.out_dict

        # self.overall = self.out_dict['predicted_score']
        # self.task_achievement = self.out_dict['task_achievement']
        # self.Coherence_and_cohesion = self.out_dict['Coherence_and_cohesion']
        # self.Lexical_resource = self.out_dict['Lexical_resource']
        # self.Grammatical_range_and_accuracy = self.out_dict['Grammatical_range_and_accuracy']

        # return [self.overall,self.task_achievement,self.Coherence_and_cohesion,self.Lexical_resource]
        
# question_prompt = '''
# In many countries today, crime novels and TV crime dramas are becoming more and more popular.
# '''

# answer = '''

    # Crime stories and TV crime series have become increasingly popular in many countries around the world. This essay will explore the reasons behind this trend. Personally, I believe that they are fascinating, but should be presented from a critical and thoughtful perspective, not as only a pure form of entertainment.

    # There are several reasons why crime novels and TV crime dramas have gained so much popularity these days. One reason is the thrill and excitement that comes with solving a crime or mystery. Viewers and readers are drawn to the suspense and challenge of figuring out the clues before the characters in the story. Additionally, crime fiction often deals with important social issues, such as corruption and injustice, which can be both informative and thought-provoking. Finally, crime novels and TV crime dramas offer an escape from reality and can be a form of entertainment for many people.

    # Personally, I enjoy crime fiction and TV crime dramas. They offer a unique blend of suspense, drama, and intellectual challenge that other genres may not provide. I find it fascinating to try to solve the mystery alongside the characters in the story and see if my own deductions match up with theirs. Additionally, crime fiction can be an effective way to explore complex social issues in a way that is accessible and engaging.

    # However, it is important to note that crime fiction and TV crime dramas are not without their criticisms. Some argue that these stories can glorify violence and crime, and perpetuate negative stereotypes about certain groups of people. Additionally, there is a risk of desensitizing people to real-life crime and violence, and creating a sense of fear and paranoia.

    # In conclusion, the popularity of crime novels and TV crime dramas can be attributed to the thrill and excitement of solving a mystery, the exploration of important social issues, and the potential for entertainment and escapism. While there are valid criticisms of these genres, I believe that when approached with a critical and thoughtful perspective, crime fiction and TV crime dramas can be valuable and enjoyable forms of entertainment.
# '''

# if __name__ == '__main__':
#     obj = ielts()
#     obj.predictor(question_prompt,answer)

#     print(obj.overall['score'],'\n',obj.overall['remarks'])
#     print(obj.task_achievement['score'],'\n',obj.task_achievement['remarks'])
#     print(obj.Coherence_and_cohesion['score'],'\n',obj.Coherence_and_cohesion['remarks'])
#     print(obj.Lexical_resource['score'],'\n',obj.Lexical_resource['remarks'])
    # print(obj.Grammatical_range_and_accuracy['score'],'\n',obj.Grammatical_range_and_accuracy['remarks'])