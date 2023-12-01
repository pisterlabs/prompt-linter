# -*- coding: utf-8 -*-

from datetime import datetime
from langchain import HuggingFaceHub, OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate

class ShroomClassifier:
    """Represents a classifier for the SHROOM evaluation dataset."""

    TASK = {
        "DM": "The given task is Definition Modeling, where the goal is to generate a definition for the term between the '<define>' and '</define>' delimiters in the input.",
        "PG": "The given task is Paraphrase Generation, where the goal is to generate a paraphrase of the input.",
        "MT": "The given task is Machine Translation, where the goal is to generate a natural language translation of the input.",
        "TS": "The given task is Text Simplification, where the goal is to generate a simplified version of the input.",
    }

    REFERENCE = {
        "src": "the input",
        "tgt": "the target",
        "either": "either the input or the target",
    }

    DEMOS = {
        'DM': '##\nInput: <define> trollcards </define> for my trollsona : v :\nTarget: (Homestuck, _, fandom slang) A graphic featuring a portrait and biography of a fantroll.\nOutput: (fandom slang) A card game in which the player character is a troll.\nRationale: The output is a hallucination. The target defines "trollcards" in the context of Homestuck fandom slang as a graphic featuring a portrait and biography of a fantroll, which is essentially a fan-created character in the Homestuck universe. In contrast, the output describes "trollcards" as a card game where the player character is a troll. This description does not match the target reference, and there is no evidence provided that "trollcards" is indeed a card game. Therefore, the output contains information that is not supported by the reference, making it a hallucination.\n##\nInput: I would sit at the end of the bar , far from the <define> jukie </define> , near the door .\nTarget: (slang) A jukebox (machine for playing music).\nOutput: (Australia, New Zealand, slang) A bar.\nRationale: The output is a hallucination. The term "jukie" in the context provided by the target refers to a jukebox, which is a machine for playing music. The output incorrectly defines "jukie" as a bar in Australia and New Zealand slang. This definition is not supported by the target reference, which clearly states that "jukie" is slang for a jukebox, not a location. Thus, the information in the output is invented and not based on the provided reference.\n##\nInput: Now , happy bridging over a <define> chinese snooker </define> .\nTarget: (snooker) A reverse snooker position where the cue ball is in front of, rather than behind, a ball that is not on, making the shot very difficult because the bridge is hampered and the cueing angle is unnaturally high.\nOutput: (snooker) A type of snooker played in China.\nRationale: The output is a hallucination. The target defines "Chinese snooker" as a specific situation in the game of snooker that involves a reverse snooker position, where the difficulty arises due to the position of the cue ball. The output incorrectly defines it as a type of snooker played in China, which is not supported by the reference provided in the target. The term "Chinese snooker" does not refer to a regional variation of the game but rather to a particular challenging shot or position within the game.\n##\nInput: Seahawks Coach Mike Holmgren was not happy when NBC showed a close - up of his <define> playsheet </define> during last week ’s loss to the Saints , but it was probably time for a new one — the Seahawks ( 3 - 3 ) have scored only 17 points in their past two games .\nTarget: (sports) A sheet of paper outlining a team\'s game plan.\nOutput: (sports) A list of all the plays that have been played in a game.\nRationale: The output is a hallucination. The target definition of a "playsheet" is a sheet of paper outlining a team\'s game plan. The output incorrectly defines a playsheet as "a list of all the plays that have been played in a game," which suggests it is a record of past events rather than a pre-planned strategy. The playsheet is used for planning future plays in a game, not documenting ones that have already occurred.\n##\nInput: We show that these groups act <define> acylindrically </define> on the CAT(-1 ) polygonal complex naturally associated to their presentation .\nTarget: In an acylindrical manner.\nOutput: In an acylindrical manner.\nRationale: The output is not a hallucination. It directly matches the target by providing an equivalent phrase, "In an acylindrical manner," which is an accurate definition of "acylindrically" as used within the given context. There is no additional information in the output that goes beyond the reference provided by the target.\n',
        'PG': '##\nInput: What did she do to you?\nTarget: \nOutput: What\'d she do to ya?\nRationale: The output "What\'d she do to ya?" is not a hallucination. It is a paraphrased version of the input "What did she do to you?" that uses contractions and a more colloquial or informal form of "you" ("ya"). The meaning of the sentence is preserved, and no new information is introduced. The output maintains the original question\'s intent and content without adding, removing, or altering the information. Therefore, the assertion that the output is a hallucination is incorrect.\n##\nInput: They\'re fake.\nTarget: \nOutput: They\'re phony.\nRationale: The output "They\'re phony" is not a hallucination. It is a valid paraphrase of the input "They\'re fake." Both "fake" and "phony" are synonyms that convey the idea of something being inauthentic or not genuine. The output maintains the original meaning without introducing any new information that is unsupported by the reference.\n##\nInput: Hello, Smitty.\nTarget: \nOutput: Hiya, Smitty.\nRationale: The output "Hiya, Smitty." is not a hallucination. It is a valid paraphrase of the input "Hello, Smitty." Both the input and the output serve as greetings directed at someone named Smitty. The word "Hiya" is an informal way of saying "Hello," thus maintaining the meaning of the original input without introducing any unsupported information.\n##\nInput: Where\'s Chuck?\nTarget: \nOutput: Where\'s chuck?\nRationale: Output is not a hallucination. The output is simply a direct repetition of the input with only the difference in capitalization of the name "Chuck." There is no new information introduced in the output that is unsupported by the reference input.\n##\nInput: Are you gonna go?\nTarget: \nOutput: Are you going to go?\nRationale: The output is not a hallucination. It is a direct paraphrase of the input "Are you gonna go?" where "gonna" is simply replaced with its standard English equivalent "going to." There is no new information added or implied that is unsupported by the input; therefore, the output remains faithful to the original meaning.\n',
        'MT': '##\nInput: Даже Том не смог бы помочь.\nTarget: Even Tom couldn\'t help.\nOutput: Even Tom couldn\'t help.\nRationale: The output "Even Tom couldn\'t help." is not a hallucination. It directly corresponds to the input "Даже Том не смог бы помочь." and conveys the same meaning as the target translation. There is no additional information in the output that is not supported by the reference input or target. Therefore, the output is an accurate translation and not a hallucination.\n##\nInput: Положите это всё в грузовик.\nTarget: Put it all in the truck.\nOutput: Put it all in the truck.\nRationale: The output is not a hallucination. It directly corresponds to the provided input and accurately reflects the target. The English translation "Put it all in the truck." is an appropriate and faithful translation of the Russian input "Положите это всё в грузовик." with no additional information added. Hence, the output contains no information that is not supported by the reference.\n##\nInput: Это зависит от того, куда мы решим поехать.\nTarget: That depends on where we decide to go.\nOutput: It depends on where we decide to go.\nRationale: The output is not a hallucination. The output "It depends on where we decide to go." is a faithful translation of the input "Это зависит от того, куда мы решим поехать." The meaning is preserved, and there is no additional information introduced that is not supported by the input or the target. Both the input and the output express the same conditional dependency on a decision about a destination.\n##\nInput: Даже Том не смог бы помочь.\nTarget: Even Tom couldn\'t help.\nOutput: Even Tom couldn\'t help.\nRationale: The output is not a hallucination. The output "Even Tom couldn\'t help." directly corresponds to the target translation and accurately reflects the information provided in the input "Даже Том не смог бы помочь." There is no additional information introduced in the output that is not supported by the input or the target.\n##\nInput: Даже Том не смог бы помочь.\nTarget: Even Tom couldn\'t help.\nOutput: Even Tom couldn\'t help.\nRationale: The output is not a hallucination. The translation "Even Tom couldn\'t help." accurately reflects the meaning of the input "Даже Том не смог бы помочь." There is no additional information introduced in the output that is unsupported by the input, and it aligns well with the provided target. Therefore, the output is a correct translation of the input with no hallucinated content.\n',
        'TS': '',
    }

    RATIONALE_GENERATION_PROMPT = """{task}  
You have been provided with the below inputs, outputs and targets. Your goal is to determine if the output is 
a hallucination, defined as an output that contains information that is not supported by the reference. Using {ref} 
as the reference, provide a succinct rationale arguing for or against the assertion that the output is a hallucination.
##
Input: {src}
Target: {tgt} 
Output: {hyp}
Rationale:
"""

    STAGE_2_RATIONALE_GENERATION_PROMPT = """{task}  
You have been provided with the below inputs, outputs and targets. Your goal is to determine if the output is 
a hallucination, defined as an output that contains information that is not supported by the reference. Using {ref} 
as the reference, provide a succinct rationale arguing for or against the assertion that the output is a hallucination.
{demos}
##
Input: {src}
Target: {tgt} 
Output: {hyp}
Rationale:
"""

    ANSWER_GENERATION_PROMPT = """Using the argument provided in the below rationale, answer the question: 
is the output a hallucination? Answer 'Hallucination' if the output is a hallucination, or 'Not Hallucination'  
if the output is not a hallucination. Only answer 'Hallucination' or 'Not Hallucination'. 
##
Input: {src}
Target: {tgt} 
Output: {hyp}
Rationale: {rationale}
Answer:
"""
    
    def __init__(self, model_name="gpt-4", temperature=0.1):
        """
        Initializes a classifier for the SemEval 2024 Task 6, "SHROOM, a Shared-task on Hallucinations and Related Observable Overgeneration Mistakes".
        
        Parameters:
            model_name: The name of the model to be used for zero shot CoT classification (default "gpt-4").
            temperature: The temperature parameter for the model (default 0.1).
         """
        self.model_name = model_name
        self.temperature = temperature
        self.llm = self._llm(model_name, temperature)
        self._classify_chain = self._zero_shot_chain_of_thought()
        self._stage_2_classify_chain = self._few_shot_chain_of_thought()

    def _llm(self, model_name, temperature):
        if model_name in [
            "gpt-4",
            "gpt-3.5-turbo",
            "gpt-4-1106-preview",
            ]:
            return ChatOpenAI(model_name=model_name, temperature=temperature, request_timeout=50)
        elif model_name in [
            "text-curie-001"
            ]:
            return OpenAI(model_name=model_name, temperature=temperature, request_timeout=50)
        elif model_name in [
            "meta-llama/Llama-2-70b-chat-hf", 
            "google/flan-t5-xxl",
            ]:
            return HuggingFaceHub(repo_id=model_name, model_kwargs={ "temperature": temperature })
        else:
            raise Exception(f'Model {model_name} not supported')

    def _zero_shot_chain_of_thought(self):
        """
        Creates a langchain.SequentialChain that implements a zero-shot
        chain of thought (CoT) using a specification. 
        """
        rationale_generation = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=[ "task", "src", "tgt", "hyp", "ref"], 
                template=self.RATIONALE_GENERATION_PROMPT
            ), 
            output_key="rationale"
        )
        answer_generation = LLMChain(
            llm=self.llm, 
            prompt=PromptTemplate(
                input_variables=["src", "tgt", "hyp", "rationale"], 
                template=self.ANSWER_GENERATION_PROMPT
            ), 
            output_key="label"
        )
        return SequentialChain(
            chains=[rationale_generation, answer_generation],
            input_variables=["task", "src", "tgt", "hyp", "ref"],
            output_variables=["rationale", "label"]
        )
    
    def _few_shot_chain_of_thought(self):
        """
        Creates a langchain.SequentialChain that implements a few-shot
        chain of thought (CoT) using a specification. 
        """
        rationale_generation = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=[ "task", "src", "tgt", "hyp", "ref", "demos"], 
                template=self.STAGE_2_RATIONALE_GENERATION_PROMPT
            ), 
            output_key="rationale"
        )
        answer_generation = LLMChain(
            llm=self.llm, 
            prompt=PromptTemplate(
                input_variables=["src", "tgt", "hyp", "rationale"], 
                template=self.ANSWER_GENERATION_PROMPT
            ), 
            output_key="predicted"
        )
        return SequentialChain(
            chains=[rationale_generation, answer_generation],
            input_variables=["task", "src", "tgt", "hyp", "ref", "demos"],
            output_variables=["rationale", "predicted"]
        )
    
    def classify(self, task, src, tgt, hyp, ref):
        """
        Determines whether or not the output (hyp) is a hallucination.
        
        Parameters:
            task: The task associated with a datapoint. One of "DM", "PG", "MT", or "TS".
            src: The input passed to a model.
            tgt: The intended reference "gold" text that the model ought to generate.
            hyp: The output the model generated.
            ref: The field(s) containing the semantic information used to determine if the input is a hallucination. One of "src", "tgt", or "either".
       
        Returns:
            A dict containing a classification of the output based on the task, reference, input, output and target.
        """
        classifications = [ 
            self._classify_chain({ 
                "task": self.TASK[task], 
                "src": src, 
                "tgt": tgt, 
                "hyp": hyp,
                "ref": self.REFERENCE[ref],
            }) for i in range(5) 
        ]
        predictions = [ classification["label"] for classification in classifications ]
        weight = 1./float(len(predictions))
        predicted_p = float(sum([ weight for prediction in predictions if prediction == 'Hallucination' ]))
        if predicted_p > 0.5:
            predicted = "Hallucination"
        else:
            predicted = "Not Hallucination"
        output = {
            "predictions": predictions,
            "predicted": predicted,
            "predicted_p": predicted_p,
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + 'Z'
        }
        for i, classification in enumerate(classifications):
            output[f'rationale_{i}'] = classification["rationale"]
        return output

    def stage_1_classify(self, task, src, tgt, hyp, ref):
        """
        Determines whether or not the output (hyp) is a hallucination.
        
        Parameters:
            task: The task associated with a datapoint. One of "DM", "PG", "MT", or "TS".
            src: The input passed to a model.
            tgt: The intended reference "gold" text that the model ought to generate.
            hyp: The output the model generated.
            ref: The field(s) containing the semantic information used to determine if the input is a hallucination. One of "src", "tgt", or "either".
       
        Returns:
            A dict containing a classification of the output based on the task, reference, input, output and target.
        """
        classifications = [ 
            self._classify_chain({ 
                "task": self.TASK[task], 
                "src": src, 
                "tgt": tgt, 
                "hyp": hyp,
                "ref": self.REFERENCE[ref],
            }) for i in range(5) 
        ]
        predictions = [ classification["label"] for classification in classifications ]
        rationales = [ classification["rationale"] for classification in classifications ]
        weight = 1./float(len(predictions))
        predicted_p = float(sum([ weight for prediction in predictions if prediction == 'Hallucination' ]))
        if predicted_p > 0.5:
            predicted = "Hallucination"
        else:
            predicted = "Not Hallucination"
        output = {
            "predictions": predictions,
            "predicted": predicted,
            "predicted_p": predicted_p,
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + 'Z',
            "rationales": rationales,
        }
        return output

    def stage_2_classify(self, task, src, tgt, hyp, ref):
        """
        Determines whether or not the output (hyp) is a hallucination.
        
        Parameters:
            task: The task associated with a datapoint. One of "DM", "PG", "MT", or "TS".
            src: The input passed to a model.
            tgt: The intended reference "gold" text that the model ought to generate.
            hyp: The output the model generated.
            ref: The field(s) containing the semantic information used to determine if the input is a hallucination. One of "src", "tgt", or "either".
       
        Returns:
            A dict containing a classification of the output based on the task, reference, input, output and target.
        """
        classification = self._stage_2_classify_chain({ 
                "task": self.TASK[task], 
                "src": src, 
                "tgt": tgt, 
                "hyp": hyp,
                "ref": self.REFERENCE[ref],
                "demos": self.DEMOS[task],
        })
        output = {
            "predicted": classification["predicted"],
            "predicted_p": (0.0 if classification["predicted"] == "Not Hallucination" else 1.0),
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + 'Z',
            "rationale": classification["rationale"],
        }
        return output
