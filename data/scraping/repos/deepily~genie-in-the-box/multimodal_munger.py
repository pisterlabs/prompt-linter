import json
import re
import os
import ast
from collections import defaultdict

import openai
import numpy as np

from lib.utils import util_stopwatch as sw, util as du

# Currently, all transcription mode descriptors are three words long.
# This will become important or more important in the future?
trans_mode_text_raw           = "multimodal text raw"
trans_mode_text_email         = "multimodal text email"
trans_mode_text_punctuation   = "multimodal text punctuation"
trans_mode_text_proofread     = "multimodal text proofread"
trans_mode_text_contact       = "multimodal contact information"
trans_mode_python_punctuation = "multimodal python punctuation"
trans_mode_python_proofread   = "multimodal python proofread"
trans_mode_server_search      = "multimodal server search"
trans_mode_run_prompt         = "multimodal run prompt"
trans_mode_vox_cmd_browser    = "multimodal browser"
trans_mode_vox_cmd_agent      = "multimodal agent"
trans_mode_default            = trans_mode_text_punctuation

modes_to_methods_dict = {
    
    trans_mode_vox_cmd_agent     : "munge_vox_cmd_agent",
    trans_mode_vox_cmd_browser   : "munge_vox_cmd_browser",
    trans_mode_text_raw          : "munge_text_raw",
    trans_mode_text_email        : "munge_text_email",
    trans_mode_text_punctuation  : "munge_text_punctuation",
    trans_mode_text_proofread    : "munge_text_proofread",
    trans_mode_text_contact      : "munge_text_contact",
    trans_mode_python_punctuation: "munge_python_punctuation",
    trans_mode_python_proofread  : "munge_python_proofread",
    # trans_mode_server_search     : "do_ddg_search",
    # trans_mode_run_prompt        : "do_run_prompt",
}
class MultiModalMunger:

    def __init__( self, raw_transcription, prefix="", prompt_key="generic", config_path="conf/modes-vox.json",
                  use_string_matching=True, use_ai_matching=True, vox_command_model="ada:ft-deepily-2023-07-12-00-02-27",
                  debug=False, verbose=False, last_response=None ):

        self.debug                  = debug
        self.verbose                = verbose
        self.config_path            = config_path
        self.raw_transcription      = raw_transcription
        self.prefix                 = prefix
        self.use_ai_matching        = use_ai_matching
        self.use_string_matching    = use_string_matching
        self.vox_command_model      = vox_command_model
        self.vox_command_threshold  = 50.0
        
        self.domain_name_model      = "ada:ft-deepily:domain-names-2023-07-12-17-15-41"
        self.search_terms_model     = "ada:ft-deepily:search-terms-2023-07-12-19-27-22"
        
        self.punctuation            = du.get_file_as_dictionary( du.get_project_root() + "/src/conf/translation-dictionary.map", lower_case=True, debug=self.debug )
        self.domain_names           = du.get_file_as_dictionary( du.get_project_root() + "/src/conf/domain-names.map",           lower_case=True )
        self.numbers                = du.get_file_as_dictionary( du.get_project_root() + "/src/conf/numbers.map",                lower_case=True )
        self.contact_info           = du.get_file_as_dictionary( du.get_project_root() + "/src/conf/contact-information.map",    lower_case=True )
        self.prompt_dictionary      = du.get_file_as_dictionary( du.get_project_root() + "/src/conf/prompt-dictionary.map",      lower_case=True )
        self.prompt                 = du.get_file_as_string( du.get_project_root() + self.prompt_dictionary.get( prompt_key, "generic" ) )
        self.command_strings        = self._get_command_strings()
        self.class_dictionary       = self._get_class_dictionary()
        
        print( "prompt_key:", prompt_key )
        if self.debug and self.verbose:
            print( "prompt:", self.prompt )
        
        self.modes_to_methods_dict  = modes_to_methods_dict
        self.methods_to_modes_dict  = self._get_methods_to_modes_dict( modes_to_methods_dict )
        
        if self.debug and self.verbose:
            print( "modes_to_methods_dict", self.modes_to_methods_dict, end="\n\n" )
            print( "methods_to_modes_dict", self.methods_to_modes_dict, end="\n\n" )
        
        # This field added to hold the Results of a calculation, e.g.: ddg search, contact information, or eventually proofreading
        # When all processing is refactored and consistent across all functionality. ¡TODO!
        self.results       = ""
        self.last_response = last_response
        
        if self.debug and self.verbose:
            print( "self.last_response:", self.last_response )
        
        parsed_fields      = self.parse( raw_transcription )
        self.transcription = parsed_fields[ 0 ]
        self.mode          = parsed_fields[ 1 ]
        
        
    def __str__( self ):

        summary = """
                       Mode: [{}]
                     Prefix: [{}]
          Raw transcription: [{}]
        Final transcription: [{}]
                    Results: [{}]""".format( self.mode, self.prefix, self.raw_transcription, self.transcription, self.results )
        return summary

    def get_jsons( self ):
        
        # instantiate dictionary and convert to string
        munger_dict = { "mode": self.mode, "prefix": self.prefix, "raw_transcription": self.raw_transcription, "transcription": self.transcription, "results": self.results }
        json_str    = json.dumps( munger_dict )
        
        return json_str
        
    def _get_methods_to_modes_dict( self, modes_to_methods_dict ):
        
        methods_to_modes_dict = { }
        
        for mode, method in modes_to_methods_dict.items():
            methods_to_modes_dict[ method ] = mode
        
        return methods_to_modes_dict
    def parse( self, raw_transcription ):
        
        # ¡OJO! super special ad hoc prefix cleanup due to the use of 'multi'... please don't do this often!
        raw_transcription = self._adhoc_prefix_cleanup( raw_transcription )
        
        # knock everything down to alphabetic characters and spaces so that we can analyze what transcription mode we're in.
        regex = re.compile( '[^a-zA-Z ]' )
        transcription = regex.sub( '', raw_transcription ).replace( "-", " " ).lower()
        
        # First and foremost: Are we in multi-modal editor/command mode?
        print( "  self.prefix:", self.prefix )
        print( "transcription:", transcription )
        
        if self.prefix == trans_mode_vox_cmd_browser or transcription.startswith( trans_mode_vox_cmd_browser ):
            
            # ad hoc short circuit for the 'repeat' command
            if (transcription == "repeat" or transcription == trans_mode_vox_cmd_browser + " repeat") and self.last_response is not None:
                
                print( self.last_response )
                
                self.prefix            = self.last_response[ "prefix" ]
                self.results           = self.last_response[ "results" ]
                self.raw_transcription = self.last_response[ "raw_transcription" ]
                
                return self.last_response[ "transcription" ], self.last_response[ "mode" ]
            
            elif (transcription == "repeat" or transcription == trans_mode_vox_cmd_browser + " repeat") and self.last_response is None:
                print( "No previous response to repeat" )

            # strip out the prefix and move it into its own field before continuing
            if transcription.startswith( trans_mode_vox_cmd_browser ):
                
                # Strip out the first two words, regardless of punctuation
                # words_sans_prefix = "multimodal? editor! search, new tab, y tu mamá, también!".split( " " )[ 2: ]
                words_sans_prefix = raw_transcription.split( " " )[ 2: ]
                raw_transcription = " ".join( words_sans_prefix )
                self.prefix = trans_mode_vox_cmd_browser

            du.print_banner( "START MODE: [{}] for [{}]".format( self.prefix, raw_transcription ), end="\n" )
            transcription, mode = self._handle_vox_command_parsing( raw_transcription )
            print( "commmand_dict: {}".format( self.results ) )
            du.print_banner( "  END MODE: [{}] for [{}]".format( self.prefix, raw_transcription ), end="\n\n" )
            
            return transcription, mode
        
        print( "transcription [{}]".format( transcription ) )
        words = transcription.split()

        prefix_count = len( trans_mode_default.split() )
        
        # If we have fewer than 'prefix_count' words, just assign default transcription mode.
        if len( words ) < prefix_count and ( self.prefix == "" or self.prefix not in self.modes_to_methods_dict ):
            method_name = self.modes_to_methods_dict[ trans_mode_default ]
        else:
            
            first_words = " ".join( words[ 0:prefix_count ] )
            print( "first_words:", first_words )
            default_method = self.modes_to_methods_dict[ trans_mode_default ]
            method_name    = self.modes_to_methods_dict.get( first_words, default_method )
            
            # Conditionally pull the first n words before we send them to be transcribed.
            if first_words in self.modes_to_methods_dict:
                raw_words = raw_transcription.split()
                raw_transcription = " ".join( raw_words[ prefix_count: ] )
            else:
                print( "first_words [{}] of raw_transcription not found in modes_to_methods_dict".format( first_words ) )
                # If we have a prefix, try to use it to determine the transcription mode.
                if self.prefix in self.modes_to_methods_dict:
                    print( "prefix [{}] in modes_to_methods_dict".format( self.prefix ) )
                    method_name = self.modes_to_methods_dict[ self.prefix ]
                else:
                    print( "prefix [{}] not found in modes_to_methods_dict either".format( self.prefix ) )
                
        mode = self.methods_to_modes_dict[ method_name ]
        
        if self.debug:
            print( "Calling [{}] w/ mode [{}]...".format( method_name, mode ) )
            print( "raw_transcription [{}]".format( raw_transcription ) )
            
        transcription, mode = getattr( self, method_name )( raw_transcription, mode )
        if self.debug:
            print( "result after:", transcription )
            print( "mode:", mode )
        
        return transcription, mode
        
    def _handle_vox_command_parsing( self, raw_transcription ):
    
        transcription, mode = self.munge_vox_cmd_browser( raw_transcription, trans_mode_vox_cmd_browser )

        # Try exact match first, then AI match if no exact match is found.
        if self.use_string_matching:

            is_match, command_dict = self._is_match( transcription )
            
            if is_match:

                self.results = command_dict
                return transcription, mode

            else:

                # Set results to something just in case we're not using AI matching below.
                self.results = command_dict

        if self.use_ai_matching:

            print( "Attempting AI match..." )
            self.results = self._get_ai_command( transcription )
            print( "Attempting AI match... done!")

        return transcription, mode
    
    def _adhoc_prefix_cleanup( self, raw_transcription ):
        
        # Find the first instance of "multi________" and replace it with "multimodal".
        multimodal_regex  = re.compile( "multi([ -]){0,1}mod[ae]l", re.IGNORECASE )
        raw_transcription = multimodal_regex.sub( "multimodal", raw_transcription, 1 )

        multimodal_regex = re.compile( "t[ao]ggle", re.IGNORECASE )
        raw_transcription = multimodal_regex.sub( "toggle", raw_transcription, 1 )
        
        return raw_transcription
    
    def _remove_protocols( self, words ):
        
        multimodal_regex = re.compile( "http([s]){0,1}://", re.IGNORECASE )
        words = multimodal_regex.sub( "", words, 1 )
        
        return words
    
    def _remove_spaces_around_punctuation( self, prose ):
    
        # Remove extra spaces.
        prose = prose.replace( " / ", "/" )
        prose = prose.replace( "[ ", "[" )
        prose = prose.replace( " ]", "]" )
    
        prose = prose.replace( "< ", "<" )
        prose = prose.replace( " >", ">" )
    
        prose = prose.replace( " )", ")" )
        prose = prose.replace( "( ", "(" )
        prose = prose.replace( " .", "." )
        prose = prose.replace( " ,", "," )
        prose = prose.replace( "??", "?" )
        prose = prose.replace( " ?", "?" )
        prose = prose.replace( "!!", "!" )
        prose = prose.replace( " !", "!" )
        prose = prose.replace( " :", ":" )
        prose = prose.replace( " ;", ";" )
        prose = prose.replace( ' "', '"' )
        
        return prose
    def munge_text_raw( self, raw_transcription, mode ):
        
        return raw_transcription, mode
    
    def munge_text_email( self, raw_transcription, mode ):
    
        # Add special considerations for the erratic nature of email transcriptions when received raw from the whisper.
        # prose = raw_transcription.replace( ".", " dot " )
        print( "BEFORE raw_transcription:", raw_transcription )
        email = raw_transcription.lower()
        
        # Decode domain names
        for key, value in self.domain_names.items():
            email = email.replace( key, value )
        
        # Decode numbers
        for key, value in self.numbers.items():
            email = email.replace( key, value )

        # Remove space between individual numbers.
        regex = re.compile( '(?<=[0-9]) (?=[0-9])' )
        email = regex.sub( "", email )
        
        print( " AFTER raw_transcription:", raw_transcription )
        
        email = re.sub( r'[,]', '', email )
        
        # phonetic spellings often contain -'s
        regex = re.compile( "(?<=[a-z])([-])(?=[a-z])", re.IGNORECASE )
        email = regex.sub( "", email )

        # Translate punctuation mark words into single characters.
        for key, value in self.punctuation.items():
            email = email.replace( key, value )

        # Remove extra spaces around punctuation.
        email = self._remove_spaces_around_punctuation( email )
        
        # Remove extra spaces
        email = email.replace( " ", "" )
        
        # Add back in the dot: yet another ad hoc fix up
        # if not prose.endswith( ".com" ) and prose.endswith( "com" ):
        #     prose = prose.replace( "com", ".com" )
        
        # Remove trailing periods.
        email = email.rstrip( "." )
        
        return email, mode
    
    def munge_vox_cmd_browser( self, raw_transcription, mode ):
        
        command = raw_transcription.lower()

        # Remove the protocol from URLs
        command = self._remove_protocols( command )
        
        # Encode domain names as plain text before removing dots below
        for key, value in self.domain_names.items():
            command = command.replace( key, value )
            
        command = re.sub( r'[,?!]', '', command )
        
        # Remove periods between words + trailing periods
        command = command.replace( ". ", " " ).rstrip( "." )
        
        # Translate punctuation mark words into single characters.
        for key, value in self.punctuation.items():
            command = command.replace( key, value )
        
        # Remove extra spaces.
        command = self._remove_spaces_around_punctuation( command )
        
        return command, mode
    
    def munge_vox_cmd_agent( self, raw_transcription, mode ):
        
        du.print_banner( "AGENT MODE for [{}]".format( raw_transcription ), end="\n" )
        print( "TODO: Implement munge_vox_cmd_agent()... For now this is just a simple passthrough..." )
        
        return raw_transcription, mode
    
    def munge_text_punctuation( self, raw_transcription, mode ):
    
        # print( "BEFORE raw_transcription:", raw_transcription )
        prose = raw_transcription.lower()
        
        # Remove the protocol from URLs
        prose = self._remove_protocols( prose )
    
        # Encode domain names as plain text before removing dots below
        for key, value in self.domain_names.items():
            prose = prose.replace( key, value )

        prose = re.sub( r'[,.]', '', prose )

        # Translate punctuation mark words into single characters.
        for key, value in self.punctuation.items():
            prose = prose.replace( key, value )
            
        # Remove extra spaces.
        prose = self._remove_spaces_around_punctuation( prose )
        
        return prose, mode
    
    def munge_text_contact( self, raw_transcription, mode, extra_words="" ):
        
        # multimodal contact information ___________
        raw_transcription = raw_transcription.lower()
        regex = re.compile( '[^a-zA-Z ]' )
        raw_transcription = regex.sub( '', raw_transcription ).replace( "-", " " )
        # # There could be more words included here, but they're superfluous, we're only looking for the 1st word After three have been stripped out already.
        # contact_info_key = raw_transcription.split()[ 0 ]
        contact_info_key = raw_transcription
        contact_info     = self.contact_info.get( contact_info_key, "N/A" )
        
        print( "contact_info_key:", contact_info_key )
        print( "    contact_info:", contact_info )
        
        if contact_info_key in "full all":
            
            contact_info = "{}\n{}\n{} {}, {}\n{}\n{}".format(
                self.contact_info[ "name" ].title(),
                self.contact_info[ "address" ].title(),
                self.contact_info[ "city" ].title(), self.contact_info[ "state" ].upper(), self.contact_info[ "zip" ],
                self.contact_info[ "email" ],
                self.contact_info[ "telephone" ]
            )
        elif contact_info_key == "city state zip":
        
            contact_info = "{} {}, {}".format(
                self.contact_info[ "city" ].title(), self.contact_info[ "state" ].upper(), self.contact_info[ "zip" ]
            )
        
        elif contact_info_key == "state":
            contact_info = contact_info.upper()
        elif contact_info_key != "email":
            contact_info = contact_info.title()
        
        self.results     = contact_info
        print( "    self.results:", self.results )
        
        return raw_transcription, mode
    def munge_text_proofread( self, raw_transcription, mode ):
    
        transcription, mode = self.munge_text_punctuation( raw_transcription, mode )
        
        return transcription, mode
    
    def munge_python_punctuation( self, raw_transcription, mode ):
    
        code = raw_transcription.lower()
        print( "BEFORE code:", code )

        # Remove "space, ", commas, and periods.
        code = re.sub( r'space, |[,-]', '', code.lower() )

        # Translate punctuation mark words into single characters.
        for key, value in self.punctuation.items():
            code = code.replace( key, value )

            # Decode numbers
        for key, value in self.numbers.items():
            code = code.replace( key, value )

        # Remove space between any two single digits
        code = re.sub( r"(?<=\d) (?=\d)", "", code )
        
        # remove exactly one space between individual letters too
        code = re.sub( r'(?<=\w) (?=\w)', '', code )
        
        # Remove extra spaces.
        code = code.replace( " _ ", "_" )
        code = code.replace( " ,", ", " )
        code = code.replace( "self . ", "self." )
        code = code.replace( " . ", "." )
        code = code.replace( "[ { } ]", "[{}]" )
        code = code.replace( " [", "[" )
        code = code.replace( " ( )", "()" )
        code = code.replace( ") :", "):" )
        code = code.replace( " ( ", "( " )
        code = code.replace( ") ;", ");" )
        # code = code.replace( " ) ", " ) " )

        # Remove extra spaces.
        code = " ".join( code.split() )
        
        print( "AFTER code:", code )
        
        return code, mode
    
    def munge_python_proofread( self, raw_transcription, mode ):
        
        return raw_transcription, mode
    
    # def do_ddg_search( self, raw_transcription, mode ):
    #
    #     transcription, mode = self.munge_text_punctuation( raw_transcription, mode )
    #
    #     if "this information" in transcription:
    #         print( "KLUDGE: 'THIS information' munged into 'DISinformation'" )
    #         transcription = transcription.replace( "this information", "disinformation" )
    #
    #     return transcription, mode
    
    # def do_run_prompt( self, raw_transcription, mode ):
    #
    #     # Not doing much preparation work here. For the moment.
    #     raw_transcription = raw_transcription.lower()
    #
    #     return raw_transcription, mode
    
    def is_text_proofread( self ):
        
        return self.mode == trans_mode_text_proofread
    
    def is_ddg_search( self ):
        
        return self.mode == trans_mode_server_search
    
    def is_run_prompt( self ):
        
        return self.mode == trans_mode_run_prompt
    
    def is_agent( self ):
        
        return self.mode == trans_mode_vox_cmd_agent
    
    def _is_match( self, transcription ):
        
        # set default values
        command_dict = self._get_command_dict( confidence=100.0 )
        
        for command in self.command_strings:
            
            if transcription == command:
                
                print( "EXACT MATCH: Transcription [{}] == command [{}]".format( transcription, command ) )
                command_dict[ "command"    ] = command
                command_dict[ "match_type" ] = "string_matching_exact"
                
                return True, command_dict
                
            elif transcription.startswith( command ):
                
                print( "Transcription [{}] STARTS WITH command [{}]".format( transcription, command ) )
                
                # Grab the metadata associated with this command: Strip out the command and use the remainder as the arguments
                command_dict[ "args"       ] = [ transcription.replace( command, "" ).strip() ]
                command_dict[ "command"    ] = command
                command_dict[ "match_type" ] = "string_matching_startswith"
                
                return True, command_dict
            
            # elif command.startswith( transcription ):
            #     print( "Command [{}] STARTS WITH transcription [{}]".format( command, transcription ) )
            #     print( "TODO: Make sure we are handling startswith() properly" )
            #     return True, "startswith" "args"
        
        print( "NO exact match        [{}]".format( transcription ) )
        print( "NO startswith() match [{}]".format( transcription ) )
        
        return False, command_dict
    
    def _get_command_dict( self, command="none of the above", confidence=0.0, args=[ "" ], match_type="none" ):
        
        command_dict = {
            "match_type": match_type,
               "command": command,
            "confidence": confidence,
                  "args": args,
        }
        return command_dict
    
    def _get_ai_command( self, transcription ):
        
        command_dict = self._get_best_guess( transcription )
        
        # Setting a threshold allows us to return a raw transcription when an utterance represents a command that's doesn't currently
        # an associated command within the fine-tuned model.
        if command_dict[ "confidence" ] >= self.vox_command_threshold:
        
            print( "Best guess is GREATER than threshold [{}]".format( self.vox_command_threshold ) )
            
            # Extract domain names & search terms
            if command_dict[ "command" ] in [ "go to new tab", "go to current tab" ]:
                
                command_dict[ "args" ] = self.extract_args( transcription, model=self.domain_name_model )
                return command_dict
            
            elif command_dict[ "command" ].startswith( "search" ):
            
                command_dict[ "args" ] = self.extract_args( transcription, model=self.search_terms_model )
                return command_dict
        
        else:
            
            print( "Best guess is LESS than threshold [{}]".format( self.vox_command_threshold ) )
            
        return command_dict
    
    def _log_odds_to_probabilities( self, log_odds ):
        
        # Convert dictionary to a sorted list of tuples ( class_name, log_odds_value )
        log_odds = sorted( log_odds.items(), key=lambda tup: tup[ 1 ], reverse=True )
        
        # Create list comprehension & get the length of the longest class name allows us to right-justify the class names when printing.
        max_class_len = max( [ len( self.class_dictionary[ item[ 0 ].strip() ] ) for item in log_odds ] )
        
        probabilities = [ ]
        
        for item in log_odds:
            
            class_name = self.class_dictionary[ item[ 0 ].strip() ]
            percent    = np.exp( float( item[ 1 ] ) ) * 100.0
            probabilities.append( ( class_name, percent ) )
            
            print( "{}: {:2.4f}%".format( class_name.rjust( max_class_len, ' ' ), percent ) )
            
        return probabilities
    
    def _get_best_guess( self, command_str ):
        
        openai.api_key = os.getenv( "FALSE_POSITIVE_API_KEY" )
        
        timer = sw.Stopwatch()
        print( "Calling [{}]...".format( self.vox_command_model ), end="" )
        response = openai.Completion.create(
            model=self.vox_command_model,
            prompt=command_str + "\n\n###\n\n",
            max_tokens=1,
            temperature=0,
            logprobs=len( self.class_dictionary.keys() ),
            stop="\n"
        )
        timer.print( "Calling [{}]... Done!".format( self.vox_command_model ), use_millis=True, end="\n" )
        
        # convert OPENAI object into a native Python dictionary... ugly!
        log_odds = ast.literal_eval( str( response[ "choices" ][ 0 ][ "logprobs" ][ "top_logprobs" ][ 0 ] ) )
        
        best_guesses = self._log_odds_to_probabilities( log_odds )
        
        # Return the first tuple in the sorted list of tuples.
        command_dict = self._get_command_dict( command=best_guesses[ 0 ][ 0 ], confidence=best_guesses[ 0 ][ 1 ], match_type="ai_matching" )
        
        return command_dict

    def extract_args( self, raw_text, model="NO_MODEL_SPECIFIED" ):
        
        openai.api_key = os.getenv( "FALSE_POSITIVE_API_KEY" )
        
        if self.debug:
            print( " raw_text [{}]".format( raw_text ) )
            print( "Using FALSE_POSITIVE_API_KEY [{}]".format( os.getenv( "FALSE_POSITIVE_API_KEY" ) ) )
        
        timer = sw.Stopwatch()
        print( "Calling [{}]...".format( model ), end="" )
        response = openai.Completion.create(
            model=model,
            prompt=raw_text + "\n\n###\n\n",
            # From: https://community.openai.com/t/cheat-sheet-mastering-temperature-and-top-p-in-chatgpt-api-a-few-tips-and-tricks-on-controlling-the-creativity-deterministic-output-of-prompt-responses/172683
            temperature=0.0,
            top_p=0.0,
            max_tokens=12,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["\n"]
        )
        timer.print( "Calling [{}]... Done!".format( model ), use_millis=True, end="\n" )
        
        if self.debug: print( response )
        
        response = response.choices[ 0 ].text.strip()
        print( "extract_args response [{}]".format( response ) )
        
        return [ response ]
        
    def _get_command_strings( self ):
    
        command_strings = du.get_file_as_list( du.get_project_root() + "/src/conf/constants.js", lower_case=True, clean=True )
        vox_commands = [ ]
        
        for command in command_strings :
            
            # Skip comments and other lines that don't split into two pieces.
            if len( command.split( " = " ) ) == 1:
                continue
            
            match = command.split( " = " )[ 1 ].strip()
            
            if match.startswith( '"' ) and match.endswith( '";' ) and not match.startswith( '"http' ):
                # Remove quotes and semicolon.
                match = match[ 1 : -2 ]
                vox_commands.append( match )
            else:
                if self.debug and self.verbose: print( "SKIPPING [{}]...".format( match ) )
                
        
        if self.debug and self.verbose:
            # Sort keys alphabetically before printing them out.
            vox_commands.sort()
            for vox_command in vox_commands: print( vox_command )
        
        # Sort the sending order by length of string, longest first.  From: https://stackoverflow.com/questions/60718330/sort-list-of-strings-in-decreasing-order-according-to-length
        vox_commands = sorted( vox_commands, key=lambda command: ( -len( command ), command) )
        
        return vox_commands
    
    def _get_class_dictionary( self ):
        
        class_dictionary = defaultdict( lambda: "unknown command" )
        # class_dictionary = { }
        class_dictionary[ "0" ] =                    "go to current tab"
        class_dictionary[ "1" ] =                        "go to new tab"
        class_dictionary[ "2" ] =                    "none of the above"
        class_dictionary[ "3" ] =            "search google current tab"
        class_dictionary[ "4" ] =                "search google new tab"
        class_dictionary[ "5" ] =    "search google scholar current tab"
        class_dictionary[ "6" ] =        "search google scholar new tab"
        class_dictionary[ "7" ] =                   "search current tab"
        class_dictionary[ "8" ] =                       "search new tab"
    
        return class_dictionary
    
if __name__ == "__main__":

    prefix = ""
    # transcription = "DOM fully loaded and parsed, Checking permissions.... Done!"
    # transcription = "multi-mode text raw Less then, Robert at somewhere.com greater than. DOM fully loaded and parsed comma Checking permissions.... Done exclamation point."
    # transcription = "multi-mode text proofread Less then, Robert at somewhere.com greater than. DOM fully loaded and parsed comma Checking permissions.... Done exclamation point."
    # transcription = "Multi-mode text punctuation Less then, Robert at somewhere.com greater than. DOM fully loaded and parsed comma Checking permissions.... Done exclamation point."
    # transcription = "multi-modal text email r-i-c-a-r-d-o dot f-e-l-i-p-e dot r-u-i-z at gmail.com"
    # transcription = "multi model text email r-i-c-a-r-d-o dot f-e-l-i-p-e dot r-u-i-z six two at sign gmail. com."
    # transcription = "multi-mode text punctuation Here's my email address. r-i-c-a-r-d-o.f-e-l-i-p-e-.r-u-i-z at gmail.com."
    # transcription = "blah blah blah"
    # transcription = "multimodal text proofread i go to market yesterday comma Tonight i go to the dance, comma, and im very happy that exclamation point."
    
    # transcription = "multimodal python punctuation Deaf, Munch, Underscore Python, Underscore Punctuation, Open Parenthesis, Space, Self, Comma, Raw Underscore transcription, Comma, Space, Mode, Space, Close Parenthesis, Colon, newline newline foo equals six divided by four newline newline bar equals brackets"
    
    # transcription = "multimodal contact information name"
    # transcription = "multimodal contact information address"
    # transcription = "City, State, Zip."
    # prefix        = "multimodal contact information"
    # prefix = ""
    
    # transcription = "full"
    # transcription = "multimodal ai fetch this information: Large, language models."
    
    # transcription = "Take Me To https://NPR.org!"
    # transcription = "fetch https://NPR.org for me in another tab!"
    # transcription = "Zoom, In!"
    # transcription = "Go ZoomInG!"
    # transcription = "Open a new tab and go to blahblah"
    # transcription = "search for fabulous blue dinner plates this tab"
    # transcription = "I'm looking for the best cobalt blue China set in another tab."
    # transcription = "Look up the best cobalt blue China set in another tab."
    # transcription = "In a new tab, search for this that and the other."
    # transcription = "Get search results for Google Scholar blah blah blah."
    # transcription = "Head to stage.magnificentrainbow.com in a new tab"
    transcription = "Head to stagemagnificentrainbowcom in a new tab"
    # transcription = "search new tab"
    # transcription = "search current tab for blah blah blah"
    # transcription = "multimodal python punctuation console.log. open parenthesis quote foober. close quote. close parenthesis. semicolon."
    transcription = "multimodal python punctuation console dot log open parentheses one plus one equals two closed parentheses"
    
    # prefix        = "multimodal browser"
    prefix        = ""
    # prefix          = "multimodal python punctuation"
    # munger = MultiModalMunger( transcription, prefix=prefix, debug=False )
    # print( munger.get_json() )
    # print( munger.extract_args( transcription, model=munger.domain_name_model ) )
    
    # print( "munger.use_exact_matching [{}]".format( munger.use_exact_matching ) )
    # print( "munger.use_ai_matching    [{}]".format( munger.use_ai_matching ) )
    # print( "munger.is_ddg_search()", munger.is_ddg_search() )
    # print( "munger.is_run_prompt()", munger.is_run_prompt(), end="\n\n" )
    # print( munger, end="\n\n" )
    # print( munger.get_json(), end="\n\n" )
    # exact_matches = munger._get_exact_matching_strings()
    #
    # for match in exact_matches: print( match )
    
    # transcription = "http://npr.org"
    
    # multimodal_regex = re.compile( "http([s]){0,1}://", re.IGNORECASE )
    # transcription = multimodal_regex.sub( "", transcription, 1 )
    # transcription = transcription.replace( ("https://")|("http://"), "" )
    # print( transcription )
    
    
    # raw_prompt = """
    # Your task is to generate a short summary of a product review from an ecommerce site.
    # Summarize the review below, delimited by triple backticks, in at most 30 words.
    # Review: ```Got this panda plush toy for my daughter's birthday,
    # who loves it and takes it everywhere. It's soft and
    # super cute, and its face has a friendly look. It's
    # a bit small for what I paid though. I think there
    # might be other options that are bigger for the same price. It arrived a day
    # earlier than expected, so I got to play with it myself before I gave it to her. ```
    # """
    # preamble = raw_prompt.split( "```" )[ 0 ].strip()
    # content = raw_prompt.split( "```" )[ 1 ].strip()
    # print( "preamble [{}]".format( preamble ) )
    # print( "  review [{}]".format( content ) )
    
    # regex = re.compile( "[.]$", re.IGNORECASE )
    # foo = regex.sub( "", "foo.", 1 )
    # print( foo )
    #
    # bar = "1 2 3 4"
    # regex = re.compile( "((?P<before>[0-9])([ ]{0,1})(?P<after>[0-9]))", re.IGNORECASE )
    # # bar = regex.sub( "multimodal", bar, 1 )
    # bar = regex.sub( "\g<before>\g<after>", bar )
    # print( bar )
    #
    # foo = " 1 2 3ab4 5 6 7 "
    # regex = re.compile( '(?<=[0-9]) (?=[0-9])' )
    # foo = regex.sub( "", foo )
    # print( "[{}]".format( foo ) )
    #
    # blah = "a-b-c-d-e-f-g-h-i-j-k-l -- -members-only- -- n-o-p-q-r-s-t-u-v-w-x-y-z"
    # regex = re.compile( "(?<=[a-z])([-])(?=[a-z])", re.IGNORECASE )
    # blah = regex.sub( "", blah )
    # print( blah )
    
    
    
    # print( "munger.get_json()", munger.get_json() )
    # print( "type( munger.get_json() )", type( munger.get_json() ) )
    # print( munger.get_json()[ "transcription" ] )
    # print( munger.is_text_proofread() )
    
    # munger = MultiModalMunger( transcription, prefix=prefix, debug=False )
    # print( munger.get_json() )
    
    # genie_client = gc.GenieClient( debug=True )
    # timer = sw.Stopwatch()
    # preamble = "You are an expert proofreader. Correct grammar. Correct tense. Correct spelling. Correct contractions. Correct punctuation. Correct capitalization. Correct word choice. Correct sentence structure. Correct paragraph structure. Correct paragraph length. Correct paragraph flow. Correct paragraph topic. Correct paragraph tone. Correct paragraph style. Correct paragraph voice. Correct paragraph mood. Correct paragraph theme."
    # transcription = "Yesterday I go to work, Tonight I go dancing friends. Tomorrow I go to work again."
    # response = genie_client.ask_chat_gpt_text( transcription, preamble=preamble )
    # print( response )
    # timer.print( "Proofread", use_millis=True )
    
    