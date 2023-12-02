import json
import os
from typing import *

from promptwatch.constants import EnvVariables
from promptwatch.promptwatch_context import PromptWatch
from promptwatch.data_model import ActivityFeedback, ChatMessage, NamedPromptTemplateDescription,ActivityBase, LlmPrompt, ChainSequence
from .schema import *
from .evaluation import DEFAULT_COSINE_SCORE_EVALUATION, TestCaseEvaluationWrapper, EvaluationStrategyBase
import logging
from tqdm import tqdm
from datetime import datetime, timezone


        

class UnitTest:

    def __init__(self, test_name:str, evaluation:EvaluationStrategyBase=DEFAULT_COSINE_SCORE_EVALUATION, api_key:str=None) -> None:
        """ Initialize new test definition. If called with the same parameters, it will be reused """
        self.test_name=test_name
        self.evaluation_strategy=evaluation
        self._entered=False
        #avoiding circular imports
        
        # it ok to create new instance... PromptWatch is a singleton and if someone want's to initialize extra params, he can wrapping it int  PromptWatch context 
        self.prompt_watch=None 
        self.conditions=None
        self.test_cases_generator=None
        from promptwatch.client import Client
        self.client=Client(api_key=api_key)
        self.evaluator=None
        
        self.simulation_definition=None
        self.simulation_type=None
       
    def for_simulated_conversations(self,  definition:ConversationSimulationDefinition=None, simulation_type:Type=None):
        """ Initiate the test run for simulated conversation
        Args:
            definition (ConversationSimulationDefinition): optional definition, to override the saved one
        """
       
        persona_names = ",".join([p.name for p in definition.personas])
        self.conditions = PromptUnitTestConditions(for_simulated_conversations=persona_names)
        self.simulation_definition=definition
        self.simulation_type= simulation_type or Simulation
        return self

    def for_test_cases(self, examples:List["TestCase"])->"UnitTest":
        self.test_cases_generator = (e for e in examples)
        return self

    def for_test_cases_in_file(self, file_path:str)->"UnitTest":
        def iterate(file_path):
            with open(file_path) as f:
                
                for line in f:
                    try:
                        data = json.loads(line)
                    except Exception as e:
                        raise Exception(f"Failed to parse line {line} in file {file_path}. Data are expected to be in json line format.\nError: {e}") 
                    
                    test_case = TestCase(**data)
                        
                    yield test_case

        self.test_cases_generator = iterate(file_path)
        self.conditions = PromptUnitTestConditions(for_test_cases_in_file=os.path.basename(file_path))
        return self
        
    
    def for_prompt_template(self,registered_template:Union[str, NamedPromptTemplateDescription]):
        """ This will pull and iterate all labeled runs for the registered template
        Args:
            param registered_template (str): registered_template name or instance of NamedPromptTemplateDescription (result of register_prompt_template)
         
        """
        # if isinstance(registered_template, NamedPromptTemplateDescription):
        #     for_template_name =  registered_template.template_name
        if isinstance(registered_template, str):
            for_template_name = registered_template
        else:
            raise ValueError("registered_template parameter must be either str or NamedPromptTemplateDescription")
        
        def iterate():
            results=None
            skip=0
            page_size=500
            while skip==0 or results:
                results= self.client.get_test_cases(for_template_name=for_template_name, skip=skip, limit=page_size)
                for res in results:
                    yield (res)
                skip+=page_size
                if len(results)<page_size:
                    break
                results=None
        
        self.conditions = PromptUnitTestConditions(for_template_name=for_template_name)
        self.test_cases_generator = iterate()
        return self
    
    def for_project_sessions(self,for_tracking_project:str=None):
        """  This will pull and iterate all top level chains in sessions that have been created with the context of the given tracking project
        
        """
        for_tracking_project=for_tracking_project or os.environ.get(EnvVariables.PROMPTWATCH_TRACKING_PROJECT)
        if not for_tracking_project:
            
            raise ValueError("tracking_project parameter is required")
        def iterate():
            results=None
            skip=0
            page_size=500
            while skip==0 or results:
                results= self.client.get_test_cases(for_tracking_project=for_tracking_project, skip=skip, limit=page_size)
                for res in results:
                    yield (res)
                skip+=page_size
                if len(results)<page_size:
                    break
                results=None
        self.conditions = PromptUnitTestConditions(for_tracking_project=for_tracking_project)
        self.test_cases_generator = iterate()
        return self


    def get_all_test_cases(self)->List[TestCase]:
        """ Useful when you just want to retrieve the test cases without running the test"""
        return list(self.test_cases_generator)
        

    
    def __enter__(self):
        
        self._entered=True
        if not self.test_cases_generator and not self.simulation_definition:
            raise Exception("Invalid use. Please define the scope of test by calling one of for_test_cases, for_test_cases_in_file, for_prompt_template, for_project_sessions or for_simulated_conversations methods before entering the context")
        
        self.prompt_watch=PromptWatch(tracking_project=self.conditions.for_tracking_project if self.conditions else None, api_key=self.client.api_key)
        self.prompt_watch.__enter__()
        
        self.unit_test_run = PromptUnitTestRun(
            test_name=self.test_name,
            session_id=self.prompt_watch.current_session.id, 
            start_time=datetime.now(tz=timezone.utc), 
            conditions=self.conditions,
            evaluation_method=self.evaluation_strategy.evaluation_method
            )
        self._session= UnitTestSession( 
            unit_test=self, 
            unit_test_run=self.unit_test_run, 
            evaluation_strategy=self.evaluation_strategy, 
            test_cases_generator=self.test_cases_generator,
            simulation_type=self.simulation_type,
            simulation_definition=self.simulation_definition
            )
        self._session.__enter__()
        self.prompt_watch.set_session_name(self.test_name)
        return self._session

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._entered=False
        self.prompt_watch.__exit__(exc_type, exc_val, exc_tb)
        self._session.__exit__(exc_type, exc_val, exc_tb)
        pass
    
class ConversationSimulationAgent(ABC):

    @abstractmethod
    def next_conversation_topic(self, continue_conversation:bool)->"TestConversationSession":
        raise NotImplementedError()


class UnitTestSession:

    def __init__(self,
                 unit_test:UnitTest, 
                 unit_test_run:PromptUnitTestRun,  
                 evaluation_strategy: EvaluationStrategyBase,
                 test_cases_generator:Optional[Iterator[TestCase]], 
                 simulation_type:Type,
                 simulation_definition:ConversationSimulationDefinition
                 ) -> None:
        """ Handle for running UnitTestSession """

        
        self.test_name = unit_test.test_name
        self.prompt_watch=unit_test.prompt_watch
        
        # entering the PromptWatch context to start the session
        
        # hooking up the activity callback to capture the LlmPrompt
        self.prompt_watch.add_on_activity_callback(self._add_llm_prompt_activity_to_result)
        self.client=self.prompt_watch.client

        self.unit_test_run=unit_test_run
        self.client.save_unit_test_run(self.unit_test_run)
        self.test_results=self.unit_test_run.results
        
        self.evaluation_strategy=evaluation_strategy
        self._test_cases_generator=test_cases_generator
        if simulation_type:
            self._simulation=simulation_type(simulation_definition,self)
        else:
            self._simulation=None
        
        #init fields       
        self._pending_test_case:TestCaseEvaluationWrapper=None
        self.outbound_stack=[]
        self.order_counter=0
        
        
    


    def __enter__(self):
        return self
   

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._pending_test_case:
            self.add_failed_result(self._pending_test_case.inner_test_case, str(exc_val))
        self.unit_test_run.end_time=datetime.now(tz=timezone.utc)
        self.print_results()
        try:
            self.client.save_unit_test_run(self.unit_test_run)
            if self.outbound_stack:
                self.client.save_unit_test_cases_result(self.test_name, self.unit_test_run.run_id,test_case_results=self.outbound_stack)
        except Exception as e:
            raise Exception(f"Failed to save unit test results. Error: {e}")

    @property
    def langchain(self):
        if not hasattr(self, "langchain_tools"):
            try:
                import langchain
                from ..langchain.unit_tests import LangChainTools
                self.langchain_tools = LangChainTools(self)
            except ImportError as e:
                raise ImportError("Failed to import langchain. Please install langchain to use this features")
        return self.langchain_tools

    def add_evaluation_result(self, test_case_result:"TestCaseResult"):
        self.order_counter+=1
        test_case_result.order=self.order_counter
        if self._pending_test_case is not None and self._pending_test_case.inner_test_case!=test_case_result.test_case:
            raise Exception("Invalid call, you can't set evaluation result for an example while another example is already pending.")
        self.outbound_stack.append(test_case_result)
        score = test_case_result.score  if test_case_result.score is not None else 0
        if self.test_results.total_processed>=1:
            self.test_results.total_processed += 1
            # Calculate the new overall score ... -> de-average the old score and add the new score .. and average again
            # new_total_score = ((total_score * N-1) + new_score) / (N)
            self.test_results.overall_score =((self.test_results.overall_score*(self.test_results.total_processed-1)) + score) /self.test_results.total_processed 
        else:
            self.test_results.total_processed += 1
            self.test_results.overall_score = score
        
        if test_case_result.passed:
            self.test_results.passed+=1
        else:
            self.test_results.failed+=1
        self._pending_test_case=None
        try:
            self.client.save_unit_test_cases_result(self.test_name, self.unit_test_run.run_id,test_case_results= self.outbound_stack)
            self.client.save_unit_test_run(self.unit_test_run)
            self.outbound_stack.clear()
        except Exception as e:
            logging.info(f"Failed to send test case result: {e}. Will retry later")

    @property
    def simulation(self)->"Simulation":
        return self._simulation
            
    
        
    def add_failed_result(self, example:"TestCase", error_description:str):
        self.add_evaluation_result(TestCaseResult(
                test_case=example, 
                score=0,
                conversation_session_id=self.prompt_watch.session_id,
                error_description=str(error_description), 
                passed=False
              ))
           
        
    def test_cases(self)->Iterator["TestCaseEvaluationWrapper"]:
        for example in tqdm(self._test_cases_generator, desc=f"Running test {self.test_name}"):
            if self._pending_test_case:
                raise Exception("Invalid call, you can't get next example until you set evaluation result for the previous one. Use `test_case.evaluate_result(generated_str)` method before iterating to next example. ")
            self._pending_test_case=TestCaseEvaluationWrapper(test_session=self, test_case=example)
            yield self._pending_test_case

    def skip(self, n=1):
        """ Useful when you want to skip some examples, without evaluating them.
        Args:
            n (int, optional): Number of examples to skip. Defaults to 1.
        """
        print("skipping")
        if self._pending_test_case:
            _skipped=[self._pending_test_case]
            self._pending_test_case=None
        else:
            _skipped=[]
        try:
            for _ in range(n-1):
                _skipped.append(next(self._test_cases_generator))
        except StopIteration:
            pass
        return _skipped


    
    def _add_llm_prompt_activity_to_result(self, activity:ActivityBase):
        """ Callback to capture the LlmPrompt """
        if self.unit_test_run.conditions and self.unit_test_run.conditions.for_template_name and self._pending_test_case and isinstance(activity,LlmPrompt):
            self._pending_test_case.llm_prompt =activity
            self._pending_test_case.activity_id=activity.id
        elif self.unit_test_run.conditions and self.unit_test_run.conditions.for_tracking_project and isinstance(activity,ChainSequence):
            if not self._pending_test_case.activity_id and activity.parent_activity_id is None:
                self._pending_test_case.activity_id=activity.id

        
        

    def print_results(self):
        # Define some color codes
        RED = '\033[31m'
        GREEN = '\033[32m'
        MAGENTA = '\033[35m'

        # Define reset code to restore the default text color
        RESET = '\033[0m'

        print("--------------")
        print(f"Unit test {MAGENTA}{self.unit_test_run.test_name}{RESET} finished")
        if self.unit_test_run.end_time is not None and self.unit_test_run.start_time is not None:
            total_duration_sec=(self.unit_test_run.end_time-self.unit_test_run.start_time).total_seconds()
        print(f"  Total Duration: {MAGENTA}{total_duration_sec:.2f}{RESET} seconds")
        print("--------------")
        print("Results:")

        if self.unit_test_run.results is not None:
            
            print(f"  Overall Score: {MAGENTA}{round((self.unit_test_run.results.overall_score or 0)*100 )}%{RESET}")

            print(f"  Total Processed: {MAGENTA}{self.unit_test_run.results.total_processed}{RESET}")
            print(f"  Passed: {GREEN if self.unit_test_run.results.passed else MAGENTA }{self.unit_test_run.results.passed}{RESET}")
            print(f"  Failed: {RED  if self.unit_test_run.results.failed else MAGENTA }{self.unit_test_run.results.failed}{RESET}")
        print("--------------")
        link_url = f"https://www.promptwatch.io/unit-tests?unit-test-run={self.unit_test_run.run_id}"
        print(f"{MAGENTA}You can see the full results here: {link_url}{RESET}")




    

class TestConversationSession:
    def __init__(self, session:UnitTestSession, simulation:"Simulation", persona:SimulationPersona, topic: ConversationTopic, evaluate_conversation:bool=True) -> None:
        self.session=session
        self._simulation=simulation
        self.conversation_id = None
        self.persona=persona
        self.topic=topic
        self._last_evaluation={}
        self.conversation = None
        
        
    
    def __enter__(self):
        self._simulation._start_conversation( self.conversation_id, self.persona, self.topic)
        self.session.prompt_watch.start_new_session()
        self.conversation_id = self.session.prompt_watch.session_id
        session_name=f"Simulated conversation with {self.persona.name}: {self.topic.name}"
        self.session.prompt_watch.set_session_name(session_name)
        
        self._is_active=True
        return self
    
    
        

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._is_active=False
        
        if not exc_val or exc_type == ExitConversationException:
            conversation_evaluation = self._simulation._evaluate_conversation(self.conversation_id)
            conversation_evaluation.conversation_session_id=self.session.prompt_watch.session_id
            self.session.add_evaluation_result(conversation_evaluation)
            
        else:
            test_case = self._simulation._get_test_case(self.conversation_id)
            logging.exception("Exception during simulated conversation")
            self.session.add_failed_result(example=test_case, 
                                       error_description=str(exc_val))
          
        self._simulation._end_conversation(self.conversation_id)
        
        self.conversation_id=None
        return True if exc_type!=KeyboardInterrupt else None


    @property
    def is_active(self)->bool:
        return self._is_active
    
    def _test_conversation_id(self):
        if not self.conversation_id:
            raise Exception("Invalid call. You haven't entered the conversation yet. Please use `with` statement to enter the conversation")

    def reset_conversation(self):
        self._test_conversation_id()
        self._simulation._reset_conversation(self.conversation_id)
    
    def evaluate_response(self, answer:str, sources:List[str]=None, reevaluate:bool=False)->ActivityFeedback:
        self._test_conversation_id()
        if not reevaluate and self._last_evaluation.get(answer):
            logging.info(f"Skipping evaluation of response {answer} as it has been already evaluated. Use reevaluate=True to force reevaluation")
            return self._last_evaluation[answer]
        evaluation = self._simulation._evaluate_response(self.conversation_id,answer ,sources)
        self._last_evaluation[answer]=evaluation
        
        return evaluation

    
    def get_next_message(self, response_to_previous:str=None, sources:List[str]=None, evaluate_response:bool=True)->str:
        self._test_conversation_id()
        if evaluate_response and response_to_previous:
            if self._last_evaluation.get(response_to_previous):
                last_evaluation = self._last_evaluation.get(response_to_previous)
            else:
                last_evaluation = self.evaluate_response(response_to_previous, sources)
            
        if response_to_previous:
            
            self.session.prompt_watch.log_assistant_answer(response_to_previous, 
                                                            feedback_label=last_evaluation.feedback_label,
                                                            feedback_rating=last_evaluation.feedback_rating,
                                                            feedback_notes=last_evaluation.feedback_notes,
                                                            metadata={"evaluation_metadata":last_evaluation.metadata}
                                                           )
        
        next_msg, is_active = self._simulation._get_next_message(self.conversation_id, response_to_previous)
        self._is_active=is_active
        if next_msg:
            self.session.prompt_watch.log_user_question(next_msg)
        else:
            self._is_active=False
            raise ExitConversationException()
        
        return next_msg
    
    def change_topic(self, new_topic:Union[ConversationTopic, "PersonaTopicConversationHandle"]):
        self._test_conversation_id()
        if isinstance(new_topic, PersonaTopicConversationHandle):
            if new_topic.persona!=self.persona:
                logging.warning("Changing topic mismatch: You shouldn't change topic to a topic of different persona")
            new_topic=new_topic.topic
        raise NotImplementedError()

class ExitConversationException(Exception):
    """ Used to quit the conversation """
    pass

class PersonaTopicConversationHandle:
    def __init__(self, session:TestConversationSession, simulation, persona:"SimulationPersona", topic:ConversationTopic) -> None:
        self.session=session
        self.persona=persona
        self.simulation=simulation
        self.topic=topic
    
    def start_conversation(self, evaluate_conversation:bool=True)->TestConversationSession:
        return TestConversationSession(session=self.session, simulation=self.simulation, persona=self.persona, topic=self.topic, evaluate_conversation=evaluate_conversation)

class PersonaHandle:
    def __init__(self, persona:SimulationPersona, simulation:"Simulation") -> None:
        self.persona=persona
        self.simulation=simulation

    def conversation_topics(self)->Iterable[PersonaTopicConversationHandle]:
        """ topics for all personas"""
        for topic in self.persona.conversation_topics:
            yield PersonaTopicConversationHandle(self.simulation.session, self.simulation,  self.persona, topic)

    def get_next_topic(self, previous: PersonaTopicConversationHandle=None):
        """ get next topic for the persona"""
        for topic in self.conversation_topics():
            if previous is None:
                return topic
            if topic==previous:
                previous=None

    def first(self):
        return self.get_next_topic(None)
        
    @property
    def name(self)->str:
        return self.persona.name
    
    def __repr__(self) -> str:
        return f"<PersonaHandle {self.name}>"


class SimulatedPersonasList:
    def __init__(self, simulation:"Simulation", personas_definition:List[SimulationPersona]):
        self.simulation=simulation
        self.personas_definition=personas_definition
    
    def conversation_topics(self,skip:int=0)->Iterable[PersonaTopicConversationHandle]:
        """ topics for all personas"""
        i=0
        for persona in self.personas_definition:
            for topic in persona.conversation_topics:
                i+=1
                if i>skip:
                    yield PersonaTopicConversationHandle(self.simulation.session, self.simulation,  persona, topic)

    def __getitem__(self, index:Union[str,int]):
        if isinstance(index, str):
            return self.get_by_name(index)
        elif isinstance(index, int):
            return PersonaHandle(self.personas_definition[index], self.simulation)
        else:
            raise TypeError(f"Index type {type(index)} not supported")
        
    def get_by_name(self, name:str)->PersonaHandle:
        for persona in self.personas_definition:
            if persona.name==name:
                return PersonaHandle(persona, self.simulation)
        raise KeyError(f"Persona with name {name} not found")

    def __len__(self):
        return len(self.personas_definition)

    def __iter__(self)->Iterator[PersonaHandle]:
        for persona in self.personas_definition:
            yield PersonaHandle(persona, self.simulation)
    
class Simulation:

    def __init__(self, simulation_definition:ConversationSimulationDefinition):
        self.simulation_definition=simulation_definition
        self.session = None
        self._personas_list = SimulatedPersonasList(simulation=self, personas_definition=simulation_definition.personas)
       
    @property
    def personas(self)->SimulatedPersonasList:
        return self._personas_list
    
    def _start_conversation(self,conversation_id:str, persona:SimulationPersona, topic:ConversationTopic)->None:
        self.session=session
        #raise NotImplementedError()
    
    def _end_conversation(self,conversation_id:str)->None:
        raise NotImplementedError()
    
    def _get_next_message(self,conversation_id:str, response:str)->Tuple[str,bool]:
        raise NotImplementedError()
    
    def _get_test_case(self,conversation_id:str)->TestCase:
        raise NotImplementedError()
    
    def _evaluate_response(self,conversation_id:str, response:str, sources:List[str]=None)->ActivityFeedback:
        raise NotImplementedError()

    def _evaluate_conversation(self, conversation_id)->TestCaseResult:
        raise NotImplementedError()
    
    def _reset_conversation(self,conversation_id:str)->None:
        raise NotImplementedError()