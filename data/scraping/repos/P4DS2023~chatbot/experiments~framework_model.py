from langchain.llms import OpenAI

class FrameworkModel():
    def __init__(self):
        self.pre_prompt = """We are in the situation of a case interview for applying to a job in consulting. You are taking the role of the interviewer in this case. You are there to practice solving case studies with the interviewee. You are provided with a reference solution to the case. You are not tasked to solve the case yourself but rather guide the candidate throughout the case. Important things for you to remember
            \n - You are supposed to help the candidate but not provide with with the solutions. The candidate must be able to solve the case on its own.
            \n - The candidate must not match the reference solution one to one but should provide most information.
            \n - after each step wait for the candidate to answer the question. You are never to take the role of the candidate and answer questions yourself!
            \n - Never automatically add something to the responses of the candidate. Only react towards what the candidate is writing.
            \n - You are provided with the full history of the conversation after the # Case Interview tag. Tags are used to show who said what in the conversation. Possible tags are 1) Candidate: 2) Interviewer: 3) Command: 4) State:
            \n - You are only supposed to use the Interviewer or System tag. Use the Interviewer tag whenever you are talking to the Candidate. Ocasionally a Command will be used to ask you about the current interview state (example: Command: Which section of the interview are we currenlty in?). Use the State tag to respond to these commands.
            \n - Command: and State: tags are not shown to the candidate
            \n - The command tag is used to provide additional commands to you. Pay attention to these commands when continuing the conversation. 
            """
        
        self.task_specific_prompt = """
            Your task is to guide the candidate through one section of the case interview process. The specific section you are evaluating is the development of a framework for the case. Your tasks are:
            \n - Guide the candidate through developing a framework for the case
            \n - First the candidate should take some time to come up with a framework on their own. 
            \n - A good framework should consists of 2-4 buckets. The buckets should be mutually exclusive but in total cover all important aspects of the case.
            \n - The candidate should not just provide the buckets but more information what exactly he wants to tackle exactly within each bucket
            \n - You are given a reference solution below. The candidate must not match the reference solution one to one but should provide most information. It is your task to evaluate if the candidate has given enough information.
            \n - If the candidate misses some points in his framework, ask him if he wants to add something to his framework. You can provide tips to the candidate if he is stuck.
            \n - It is your task to check if the candidate has finished this section and came up with a good enough framework. Do not let the candidate leave before this is achieved."""

        self.case_information = """# Reference Information about the case
            \n ## Problem Statement: A leading biotech company is developing a treatment for Alzheimer's disease. This ground-breaking treatment is different from any other Alzheimer's treatment on the market because it actually slows the progression of the disease, rather than just treating its symptoms. The company’s executive team is concerned about the results of a high-level internal feasibility study that flagged a potential risk to the launch of this treatment – a rumored shortage of infusion capacity in the US. Given that the Alzheimer's treatment is designed to be administered via infusion, such a shortage would severely hamper the market acceptance and hence the financial rewards from the treatment. In preparation for the launch of this treatment, the company has hired you to help them figure out the extent of the expected shortfall, and how they should respond.
            \n ## Additional Information:  [
                "Infusion refers to inserting the medicine directly into a patients bloodstream via IV (intravenous) application, ideally through the patient’s arm.",
                "The treatment will be launched in the US alone.",
                "The client has not yet estimated how big the infusion shortfall will be.",
                "The client does not have any strategies to mitigate the shortfall.",
                "Most other Alzheimers medications are delivered as oral pills.",
                "The treatment (if approved by the FDA) would come to market in about 2 years."
            ]"""
        
        self.reference_solution = """# Reference Solution for good framework
            \n - Bucket 1: What is the expected shortfall in infusion capacity?
            \n - Bucket 2: Why is there a shortfall in infusion capacity?
            \n - Bucket 3: How can the shortfall in infusion capacity be mitigated?"""
        
        self.conversation_history = [self.pre_prompt, self.task_specific_prompt, self.case_information, self.reference_solution]

        self.llm = OpenAI()
    
    def run_llm(self):
        return self.llm("\n\n".join(self.conversation_history))
    
    def add_candidate_interaction(self, candidate_response, on_finished_callback):
        self.conversation_history.append(f"Candidate: {candidate_response}")
        # check whether we continued
        self.conversation_history.append("Command: Did the candidate provide enough information to solve the case. Be critical. Especially check if enough buckets are provided and if detailed information per bucket is provided, it is not enough to just state the buckets. Answer with either State: Enough information provided or State: Not enough information provided")

        result = self.run_llm()
        print(f"Check to continue result: {result}")

        # check how to continue
        if "State: Not enough information provided".lower() in result.lower():
            print("Not enough information provided")
            self.conversation_history.append("Command: The candidate has not provided enough information. Ask the candidate to add more information to his framework. You can provide tips to the candidate if he is stuck.")
            
            result = self.run_llm()

            print(result)
        
        elif "State: Enough information provided".lower() in result.lower():
            print("Enough information provided")
            self.conversation_history.append("Command: The candidate has provided enough information. You can now ask the candidate to move on to the next section, ask him where he wants to start")
            result = self.run_llm()
            on_finished_callback()