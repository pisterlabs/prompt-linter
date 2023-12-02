import hashlib
import os
import glob
import json
import copy
import regex as re
from collections import OrderedDict

import lib.utils.util as du
import lib.utils.util_stopwatch as sw
import lib.utils.util_code_runner as ucr

from lib.agents.agent    import Agent

import openai
import numpy as np

class SolutionSnapshot( Agent ):
    
    @staticmethod
    def get_timestamp():
        
        return du.get_current_datetime()
    
    @staticmethod
    def clean_question( question ):
        
        regex = re.compile( "[^a-zA-Z ]" )
        cleaned_question = regex.sub( '', question ).lower()
        
        return cleaned_question
    
    @staticmethod
    def generate_embedding( text ):
        
        msg = f"Generating embedding for [{du.truncate_string( text )}]..."
        timer = sw.Stopwatch( msg=msg )
        openai.api_key = os.getenv( "FALSE_POSITIVE_API_KEY" )
        
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        timer.print( "Done!", use_millis=True )
        
        return response[ "data" ][ 0 ][ "embedding" ]
    
    @staticmethod
    def generate_id_hash( push_counter, run_date ):
        
        return hashlib.sha256( (str( push_counter ) + run_date).encode() ).hexdigest()
    
    @staticmethod
    def get_default_stats_dict():
        
        return {
            "run_count"  : 0,
            "total_ms"   : 0,
            "mean_run_ms": 0,
            "last_run_ms": 0
        }
    
    def __init__( self, push_counter=-1, question="", synonymous_questions=OrderedDict(),
                  last_question_asked="", answer="", answer_short="", answer_conversational="", error="",
                  created_date=get_timestamp(), updated_date=get_timestamp(), run_date=get_timestamp(),
                  runtime_stats=get_default_stats_dict(),
                  id_hash="", solution_summary="", code=[], code_type="raw", thoughts="",
                  programming_language="Python", language_version="3.10",
                  question_embedding=[ ], solution_embedding=[ ], code_embedding=[ ], thoughts_embedding=[ ],
                  solution_directory="/src/conf/long-term-memory/solutions/", solution_file=None, debug=False, verbose=False
        ):
        
        super().__init__( debug=debug, verbose=verbose )
        
        dirty                      = False
        
        self.push_counter          = push_counter
        self.question              = SolutionSnapshot.clean_question( question )
        self.thoughts              = thoughts
        
        self.answer                = answer
        self.answer_short          = answer_short
        self.answer_conversational = answer_conversational
        self.error                 = error
        
        # Is there is no synonymous questions to be found then just recycle the current question
        if len( synonymous_questions ) == 0:
            self.synonymous_questions = synonymous_questions[ question ] = 100.0
        else:
            self.synonymous_questions = synonymous_questions
            
        self.last_question_asked   = last_question_asked
        
        self.solution_summary      = solution_summary
        
        self.code                  = code
        self.code_type             = code_type
        
        # metadata surrounding the question and the solution
        self.updated_date          = updated_date
        self.created_date          = created_date
        self.run_date              = run_date
        self.runtime_stats         = runtime_stats
        
        if id_hash == "":
            self.id_hash           = SolutionSnapshot.generate_id_hash( self.push_counter, self.run_date )
        else:
            self.id_hash           = id_hash
        self.programming_language  = programming_language
        self.language_version      = language_version
        self.solution_directory    = solution_directory
        self.solution_file         = solution_file
        
        # If the question embedding is empty, generate it
        if question != "" and not question_embedding:
            self.question_embedding = self.generate_embedding( question )
            dirty = True
        else:
            self.question_embedding = question_embedding
        
        # If the code embedding is empty, generate it
        if code and not code_embedding:
            self.code_embedding     = self.generate_embedding( " ".join( code ) )
            dirty = True
        else:
            self.code_embedding     = code_embedding
    
        # If the solution embedding is empty, generate it
        if solution_summary and not solution_embedding:
            self.solution_embedding = self.generate_embedding( solution_summary )
            dirty = True
        else:
            self.solution_embedding = solution_embedding

        # If the thoughts embedding is empty, generate it
        if thoughts and not thoughts_embedding:
            self.thoughts_embedding = self.generate_embedding( thoughts )
            dirty = True
        else:
            self.thoughts_embedding = thoughts_embedding
            
        # Save changes if we've made any change as while loading
        if dirty: self.write_current_state_to_file()
        
    @classmethod
    def from_json_file( cls, filename, debug=False ):
        
        if debug: print( f"Reading {filename}..." )
        with open( filename, "r" ) as f:
            data = json.load( f )
        
        return cls( **data )
    
    @classmethod
    def create( cls, agent ):
        
        print( "(create_solution_snapshot) TODO: Reconcile how we're going to get a dynamic path to the solution file's directory" )
        
        # Instantiate a new SolutionSnapshot object using the contents of the calendaring or function mapping agent
        return SolutionSnapshot(
                         question=agent.question,
              last_question_asked=agent.last_question_asked,
             synonymous_questions=OrderedDict( { agent.question: 100.0 } ),
                            error=agent.prompt_response_dict.get( "error", "" ),
                 solution_summary=agent.prompt_response_dict.get( "explanation", "No explanation available" ),
                             code=agent.prompt_response_dict.get( "code", "No code provided" ),
                         thoughts=agent.prompt_response_dict.get( "thoughts", "No thoughts recorded" ),
                           answer=agent.code_response_dict.get( "output", "No answer available" ),
            answer_conversational=agent.answer_conversational
               
               # TODO: Reconcile how we're going to get a dynamic path to the solution file's directory
               # solution_directory=calendaring_agent.solution_directory
        )
    
    
    def add_synonymous_question( self, question, score=100.0 ):
        
        question = SolutionSnapshot.clean_question( question )
        
        self.last_question_asked = question
        
        if question not in self.synonymous_questions:
            self.synonymous_questions[ question ] = score
            self.updated_date = self.get_timestamp()
        else:
            print( f"Question [{question}] is already listed as a synonymous question." )
        
    def get_last_synonymous_question( self ):
        
        return list( self.synonymous_questions.keys() )[ -1 ]
        
    def complete( self, answer, code=[ ], solution_summary="" ):
        
        self.answer = answer
        self.set_code( code )
        self.set_solution_summary( solution_summary )
        # self.completion_date  = SolutionSnapshot.get_current_datetime()
        
    def set_solution_summary( self, solution_summary ):

        self.solution_summary = solution_summary
        self.solution_embedding = self.generate_embedding( solution_summary )
        self.updated_date = self.get_timestamp()

    def set_code( self, code ):

        # ¡OJO! code is a list of strings, not a string!
        self.code           = code
        self.code_embedding = self.generate_embedding( " ".join( code ) )
        self.updated_date   = self.get_timestamp()
    
    def get_question_similarity( self, other_snapshot ):
        
        if not self.question_embedding or not other_snapshot.question_embedding:
            raise ValueError( "Both snapshots must have a question embedding to compare." )
        return np.dot( self.question_embedding, other_snapshot.question_embedding ) * 100
    
    def get_solution_summary_similarity( self, other_snapshot ):
        
        if not self.solution_embedding or not other_snapshot.solution_embedding:
            raise ValueError( "Both snapshots must have a solution summary embedding to compare." )
        
        return np.dot( self.solution_embedding, other_snapshot.solution_embedding ) * 100
    
    def get_code_similarity( self, other_snapshot ):
        
        if not self.code_embedding or not other_snapshot.code_embedding:
            raise ValueError( "Both snapshots must have a code embedding to compare." )
        
        return np.dot( self.code_embedding, other_snapshot.code_embedding ) * 100
    
    def to_jsons( self, verbose=True ):
        
        # TODO: decide what we're going to exclude from serialization, and why or why not!
        # Right now I'm just doing this for the sake of expediency as I'm playing with class inheritance for agents
        fields_to_exclude = [ "prompt_response", "prompt_response_dict", "code_response_dict" ]
        data = { field: value for field, value in self.__dict__.items() if field not in fields_to_exclude }
        return json.dumps( data )
        
    def get_copy( self ):
        return copy.copy( self )
    
    def get_html( self ):
        
        return f"<li id='{self.id_hash}'><span class='play'>{self.run_date} Q: {self.last_question_asked}</span> <span class='delete'></span></li>"
     
    def write_current_state_to_file( self ):
        
        # Get the project root directory
        project_root = du.get_project_root()
        # Define the directory where the file will be saved
        directory = f"{project_root}{self.solution_directory}"
        
        if self.solution_file is None:
            
            print( "NO solution_file value provided (Must be a new object). Generating a unique file name..." )
            # Generate filename based on first 64 characters of the question
            filename_base = du.truncate_string( self.question, max_len=64 ).replace( " ", "-" )
            # Get a list of all files that start with the filename base
            existing_files = glob.glob( f"{directory}{filename_base}-*.json" )
            # The count of existing files will be used to make the filename unique
            file_count = len( existing_files )
            # generate the file name
            filename = f"{filename_base}-{file_count}.json"
            self.solution_file = filename
        
        else:
            
            print( f"solution_file value provided: [{self.solution_file}]..." )
        
        # Generate the full file path
        file_path = f"{directory}{self.solution_file}"
        # Print the file path for debugging purposes
        print( f"File path: {file_path}", end="\n\n" )
        # Write the JSON string to the file
        with open( file_path, "w" ) as f:
            f.write( self.to_jsons() )
        
        # Set the file permissions to world-readable and writable
        os.chmod( file_path, 0o666 )
        
    # Add a method to delete this snapshot file from the local file system if it exists
    def delete_file( self ):
        
        file_path = f"{du.get_project_root()}{self.solution_directory}{self.solution_file}"
        
        if os.path.isfile( file_path ):
            os.remove( file_path )
            print( f"Deleted file [{file_path}]" )
        else:
            print( f"File [{file_path}] does not exist" )
            

    def update_runtime_stats( self, timer ) -> None:
        """
        Updates the runtime stats for this object
        
        Uses the timer object to calculate the delta between now and when this object was instantiated

        :return: None
        """
        delta_ms = timer.get_delta_ms()
        
        self.runtime_stats[ "run_count"   ] += 1
        self.runtime_stats[ "total_ms"    ] += delta_ms
        self.runtime_stats[ "mean_run_ms" ]  = int( self.runtime_stats[ "total_ms" ] / self.runtime_stats[ "run_count" ] )
        self.runtime_stats[ "last_run_ms" ]  = delta_ms
    
    def run_code( self, debug=False, verbose=False ):
        
        self.code_response_dict = ucr.assemble_and_run_solution( self.code, path="/src/conf/long-term-memory/events.csv", debug=debug )
        self.answer             = self.code_response_dict[ "output" ]
        
        if debug and verbose:
            du.print_banner( "Code output", prepend_nl=True )
            for line in self.code_response_dict[ "output" ].split( "\n" ):
                print( line )
        
        return self.code_response_dict
    
    def format_output( self, format_model=Agent.GPT_3_5 ):
        
        preamble = self._get_formatting_preamble()
        instructions = self._get_formatting_instructions()
        
        self._print_token_count( preamble, message_name="formatting preamble", model=format_model )
        self._print_token_count( instructions, message_name="formatting instructions", model=format_model )
        
        self.answer_conversational = self._query_llm( preamble, instructions, model=format_model, debug=self.debug )
        
        return self.answer_conversational
    
    def _get_user_message( self ):
        
        msg = "SolutionSnapshot._get_user_message() NOT IMPLEMENTED"
        du.print_banner( msg, expletive=True )
        raise NotImplementedError( msg )
    
    def _get_system_message( self ):
        
        msg = "SolutionSnapshot._get_system_message() NOT IMPLEMENTED"
        du.print_banner( msg, expletive=True )
        raise NotImplementedError( msg )
    
    def is_runnable( self ):
        
        if hasattr( self, 'code' ) and self.code != [ ]:
            return True
        else:
            return False
    
    def run_prompt( self, question="" ):
        
        msg = "SolutionSnapshot.run_prompt() NOT IMPLEMENTED"
        du.print_banner( msg, expletive=True )
        raise NotImplementedError( msg )
    
    def is_promptable( self ):
        
        return False
    
    # @staticmethod
    # def get_timestamp():
    #
    #     return du.get_current_datetime()
    #
    # @staticmethod
    # def clean_question( question ):
    #
    #     regex = re.compile( "[^a-zA-Z ]" )
    #     cleaned_question = regex.sub( '', question ).lower()
    #
    #     return cleaned_question
    #
    # @staticmethod
    # def generate_embedding( text ):
    #
    #     msg = f"Generating embedding for [{du.truncate_string( text )}]..."
    #     timer = sw.Stopwatch( msg=msg )
    #     openai.api_key = os.getenv( "FALSE_POSITIVE_API_KEY" )
    #
    #     response = openai.Embedding.create(
    #         input=text,
    #         model="text-embedding-ada-002"
    #     )
    #     timer.print( "Done!", use_millis=True )
    #
    #     return response[ "data" ][ 0 ][ "embedding" ]
    #
    # @staticmethod
    # def generate_id_hash( push_counter, run_date ):
    #
    #     return hashlib.sha256( (str( push_counter ) + run_date).encode() ).hexdigest()
    #
    # @staticmethod
    # def get_default_stats_dict():
    #
    #     return {
    #         "run_count"  : 0,
    #         "total_ms"   : 0,
    #         "mean_run_ms": 0,
    #         "last_run_ms": 0
    #     }


# Add main method
if __name__ == "__main__":
    
    embedding = SolutionSnapshot.generate_embedding( "what time is it" )
    print( embedding )
    # today = SolutionSnapshot( question="what day is today" )
    # # tomorrow = SolutionSnapshot( question="what day is tomorrow" )
    # # blah = SolutionSnapshot( question="i feel so blah today" )
    # # color = SolutionSnapshot( question="what color is the sky" )
    # # date = SolutionSnapshot( question="what is today's date" )
    #
    # # snapshots = [ today, tomorrow, blah, color, date ]
    # snapshots = [ today ]
    #
    # for snapshot in snapshots:
    #     score = today.get_question_similarity( snapshot )
    #     print( f"Score: [{score}] for [{snapshot.question}] == [{today.question}]" )
    #     snapshot.write_to_file()
    
    # foo = SolutionSnapshot.from_json_file( du.get_project_root() + "/src/conf/long-term-memory/solutions/what-day-is-today-0.json" )
    # print( foo.to_json() )
    # pass
