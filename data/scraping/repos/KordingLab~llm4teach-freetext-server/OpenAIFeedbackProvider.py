from typing import List, Optional
import guidance

from ..config import OpenAIConfig
from ..feedback_providers.FeedbackProvider import FeedbackProvider
from ..llm4text_types import Assignment, Feedback, Submission


class OpenAIChatBasedFeedbackProvider(FeedbackProvider):

    """
    A feedback provider that transacts with the OpenAI API for student
    responses. Uses Microsoft's `guidance` tool for feedback curation. This
    version, unline OpenAICompletionBasedFeedbackProvider, uses the chat API
    instead of the completion API, which gives more conversational feedback and
    more cost-effective API use, at the cost of a more constrained prompt.
    """

    def __init__(self, config_override: Optional[OpenAIConfig] = None):
        if config_override is not None:
            self.config = config_override
        else:
            self.config = OpenAIConfig()

    async def get_feedback(
        self, submission: Submission, assignment: Assignment
    ) -> list[Feedback]:
        """
        Returns the feedback.
        Arguments:
            submission: The submission to provide feedback for.
            assignment: The assignment to provide feedback for.
        Returns:
            A list of feedback objects.
        """

        # set the default language model used to execute guidance programs
        try:
            openai_kwargs = self.config.dict()
            guidance.llm = guidance.llms.OpenAI("gpt-3.5-turbo", **openai_kwargs)

            grader = guidance.Program(
                """
            {{#system~}}
            You are a helpful instructor, who knows that students need precise and terse feedback. Students are most motivated if you are engaging and remain positive, but it is more important to be honest and accurate than cheerful.
            {{~/system}}

            {{#user~}}
            The student has been given the following prompt by the instructor:

            ----
            {{prompt}}
            ----

            The secret, grader-only criteria for grading are:
            ----
            {{criteria}}
            ----

            Please give your OWN answer to the prompt:

            {{~/user}}

            {{#assistant~}}
            {{gen '_machine_answer'}}
            {{~/assistant}}

            {{#user~}}
            The complete student response is as follows:
            ----

            {{response}}

            ----

            Thinking about the differences between your answer and the student's, provide your feedback to the student as a bulleted list indicating both what the student got right and what they got wrong. Give details about what they are missing or misunderstood, and mention points they overlooked, if any.

            Do not instruct the student to review the criteria, as this is not provided to the student. Write to the student using "you" in the second person. The student will not see your answer to the prompt, so do not refer to it.

            Be particularly mindful of scientific rigor issues including confusing correlation with causation, biases, and logical fallacies. You must also correct factual errors using your extensive domain knowledge, even if the errors are subtle or minor.

            Do not say "Keep up the good work" or other encouragement "fluff." Write only the response to the student; do not write any other text.

            {{audience_caveat}}

            {{fact_check_caveat}}
            {{~/user}}

            {{#assistant~}}
            {{gen 'feedback'}}
            {{~/assistant}}
            """
            )

            response = submission.submission_string
            feedback = grader(
                response=response,
                prompt=assignment.student_prompt,
                criteria="\n".join(
                    [f"     * {f}" for f in assignment.feedback_requirements]
                ),
                audience_caveat="",  # You should provide feedback keeping in mind that the student is a Graduate Student and should be graded accordingly.
                fact_check_caveat="You should also fact-check the student's response. If the student's response is factually incorrect, you should provide feedback on the incorrect statements.",
            )

            return [
                Feedback(
                    feedback_string="\n" + feedback["feedback"],
                    source="OpenAIFeedbackProvider",
                    location=(0, len(submission.submission_string)),
                )
            ]
        except Exception as e:
            print(e)
            return []

    async def suggest_criteria(self, assignment: Assignment) -> List[str]:
        """
        Uses the OpenAI ChatGPT 3.5 Turbo model to suggest grading criteria
        for a question. If criteria are already provided, they may be edited
        or modified by this function.

        Arguments:
            assignment (Assignment): The assignment to suggest criteria for.

        Returns:
            List[str]: A list of criteria.

        """
        try:
            openai_kwargs = self.config.dict()
            guidance.llm = guidance.llms.OpenAI("gpt-3.5-turbo", **openai_kwargs)

            grader = guidance.Program(
                """
            {{#system~}}
            You are a helpful instructor, who knows that students need precise and terse feedback.
            {{~/system}}

            {{#user~}}
            The student has been given the following prompt by the instructor:

            ----
            {{prompt}}
            ----

            The secret, grader-only criteria for grading are:
            ----
            {{criteria}}
            ----

            Please give your OWN answer to the prompt:

            {{~/user}}

            {{#assistant~}}
            {{gen '_machine_answer'}}
            {{~/assistant}}

            {{#user~}}
            Thinking about the important points that must be addressed in this question, provide a bulleted list of criteria that should be used to grade the student's response. These criteria should be specific and precise, and should be able to be applied to the student's response to determine a grade. You may include the criteria that were provided to the student if you agree with them, or you may modify them or replace them entirely.

            In general, you should provide 3-5 criteria. You can provide fewer if you think that is appropriate.

            {{audience_caveat}}
            {{~/user}}

            {{#assistant~}}
            {{gen 'criteria'}}
            {{~/assistant}}
            """
            )

            response = assignment.student_prompt
            criteria = grader(
                response=response,
                prompt=assignment.student_prompt,
                criteria="\n".join(
                    [f"     * {f}" for f in assignment.feedback_requirements]
                ),
                audience_caveat="",
            )

            return criteria["criteria"].split("\n")
        except Exception as e:
            print(e)
            return []

    async def suggest_question(self, assignment: Assignment) -> str:
        """
        Generate a suggestion for a student-facing question, given a current
        question and a set of criteria.

        The current implementation is a bit heavy, and may be unadvised for
        high-throughput applications as it requires several responses from a
        LLM to arrive at an improved question.

        The current approach is the following:
        * Take a question "draft" and have the machine answer it.
        * Grade that response with the current criteria.
        * Identify what properties (if any) SHOULD have been more clear in the
            question text.
        * Generate a new question that includes those properties.

        """
        try:
            openai_kwargs = self.config.dict()
            guidance.llm = guidance.llms.OpenAI("gpt-3.5-turbo", **openai_kwargs)

            draft_response = guidance.Program(
                """
            {{#system~}}
            You are a knowledgeable assistant who is working to develop a course.
            {{~/system}}

            {{#user~}}
            You must answer the following question to the best of your ability.

            ----
            {{prompt}}
            ----

            Please give your OWN answer to this question:

            {{~/user}}

            {{#assistant~}}
            {{gen '_machine_answer'}}
            {{~/assistant}}
            """
            )
            criteria = draft_response(prompt=assignment.student_prompt)

            feedback = await self.get_feedback(
                Submission(
                    submission_string=criteria["_machine_answer"],
                    assignment_id="_DRAFT_",
                ),
                assignment,
            )

            question_improver = guidance.Program(
                """
            {{#system~}}
            You are a knowledgeable instructor who is working to develop a course.
            {{~/system}}

            {{#user~}}
            A student has been given the following prompt by the instructor:

            ----
            {{prompt}}
            ----

            The student has received the following feedback from the grader:

            ----
            {{feedback}}
            ----

            You are concerned that the student may have been confused by the question. You want to improve the question so that students are less likely to be confused. You should not change the meaning of the question, but you may clarify the question so that the requirements of the grader are more clear. Do not explicitly refer to the feedback in your question. Your question should take the form of a question that a student would be asked.

            {{~/user}}

            {{#assistant~}}
            {{gen 'improved_question'}}
            {{~/assistant}}
            """
            )

            improved_question = question_improver(
                prompt=assignment.student_prompt,
                feedback="\n".join([f.feedback_string for f in feedback]),
            )

            return improved_question["improved_question"].lstrip('"').rstrip('"')
        except Exception as e:
            print(e)
            return assignment.student_prompt


__all__ = [
    "OpenAIChatBasedFeedbackProvider",
]
