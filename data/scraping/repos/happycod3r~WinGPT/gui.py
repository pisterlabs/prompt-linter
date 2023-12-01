import paths
import tkinter as tk
import customtkinter as ctk
from tkinter import simpledialog
from tkinter import colorchooser
from tkinter import filedialog
import tkinter.messagebox
from tkinter import *
import modules.normaltime as normaltime
import modules.persistence as persistence
import modules.stdops as stdops
from cli import OpenAIInterface
from ctrls import ctktextbox
from ctrls import translation_form as tf
from ctrls import qa_form as qaf
from ctrls import image_form as imgf
from ctrls import embeddings_view as ev
from ctrls import edit_view as edv
from ctrls import audio_transcription_view as atv
from PIL import Image
import sys
import os 
import modules.logger as logger

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class WinGTPGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.stdops = stdops.StdOps()
        self.log = logger.LogManager()
        self._config = persistence.Persistence()
        self._config.openConfig()
        
        self.CURRENT_PATH = paths.CURRENT_PATH
        self.CONFIG_DIR = paths.CONFIG_DIR
        self.GUI_SHOWN_FLAG_FILE = paths.GUI_SHOWN_FLAG_FILE
        
        self.NEW_USER = self._config.getOption("system", "new_user")
        self.USER = self._config.getOption("user", "username")
        self.API_KEY = self._config.getOption("user", "api_key")
        
        self.KEY_CONFIG_FILE = paths.KEY_CONFIG_FILE

        #//////////// SETTINGS ////////////
        self._OUTPUT_COLOR = f"{self._config.getOption('ui', 'color')}"
        self.UI_SCALING = self._config.getOption('ui', "ui_scaling")
        self.THEME = self._config.getOption('ui', "theme")
        self.SAVE_CHAT = self._config.getOption("chat", "chat_to_file")
        self.CHAT_LOG_PATH = self._config.getOption("chat", "chat_log_path")
        self.ECHO_CHAT = self._config.getOption("chat", "echo_chat")
        self.STREAM_CHAT = self._config.getOption("chat", "stream_chat")
        self.USE_STOP_LIST = self._config.getOption("chat", "use_stop_list")
        self.CHAT_TEMP = self._config.getOption("chat", "chat_temperature")
        self.CHAT_ENGINE = self._config.getOption("chat", "chat_engine")
        self.RESPONSE_TOKEN_LIMIT = self._config.getOption("chat", "response_token_limit")
        self.RESPONSE_COUNT = self._config.getOption("chat", "response_count")
        self.API_BASE = self._config.getOption("chat", "api_base")
        self.API_VERSION = self._config.getOption("chat", "api_version")
        self.API_TYPE = self._config.getOption("chat", "api_type")
        self.ORGANIZATION = self._config.getOption("user", "organization")
        self.REQUEST_TYPE = self._config.getOption("chat", "request_type")
        self.BEST_OF = self._config.getOption("chat", "best_of")
        self.FREQUENCY_PENALTY = self._config.getOption("chat", "frequency_penalty")
        self.PRESENCE_PENALTY = self._config.getOption("chat", "presence_penalty")
        self.TIMEOUT = self._config.getOption("chat", "timeout")
        
        self.nt = normaltime.NormalTime()
        self.cli = OpenAIInterface()
        self.cli.setAPIKeyPath(self.KEY_CONFIG_FILE)
        
        self.commands = [
            'exit', # exit the chat session.
            '-l',   # Set the response token limit.
            '-e',   # Set the engine. 
            '-r',   # Set the number of reponses
            '-b',   # Set the API base.
            '-t',   # Set the API type.
            '-v',   # Set the api version.
            '-o',   # Set the organization name.
            '-f',   # Set the user defined file name.
            '-j',   # Set the JSONL data file path.
            'help', # Prints this message. 
            'clear', # Clear the output box.
            'theme', # Change the theme. Requires theme as 1st argument.
            'color', # Change the output color
            'temp',  # Set the output temperature.
        ]
        
        self.width = 1350  # 1100
        self.height = 580
                        
        #//////////// WINDOW ////////////
        self.title("WinGTP Powered by Python & OpenAI")
        self.geometry(f"{self.width}x{self.height}")
        #//////////// GRID LAYOUT (4x4) ////////////
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)
        
        #//////////// SIDEBAR ////////////
        self.sidebar = ctk.CTkFrame(self, width=140, corner_radius=0)                            
        self.sidebar.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar.grid_rowconfigure(6, weight=1)
        self.sidebar.configure(corner_radius=4)
        
        #//////////// SIDEBAR LOGO ////////////
        

        self.sidebar_logo = ctk.CTkLabel(self.sidebar, text="WinGTP")
        self.sidebar_logo.grid(row=0, column=0, padx=0, pady=(20, 10))
        
        #//////////// USERNAME BUTTON ////////////
        self.sidebar_username_btn = ctk.CTkButton(self.sidebar, command=self.sidebar_username_btn_event)
        self.sidebar_username_btn.grid(row=1, column=0, padx=20, pady=10)
        
        #//////////// SET API KEY BUTTON ////////////
        self.sidebar_set_key_btn = ctk.CTkButton(self.sidebar, command=self.sidebar_set_key_btn_event)
        self.sidebar_set_key_btn.grid(row=2, column=0, padx=20, pady=10)
        
        #//////////// EXIT BUTTON ////////////
        self.sidebar_exit_btn = ctk.CTkButton(self.sidebar, command=self.sidebar_exit_btn_event)
        self.sidebar_exit_btn.grid(row=3, column=0, padx=20, pady=10)
        self.sidebar.configure()
        
        #//////////// CHANGE OUTPUT COLOR BUTTON ////////////
        self.change_color_btn_label = ctk.CTkLabel(self.sidebar, text="Output Color", anchor="s")
        self.change_color_btn_label.grid(row=4, column=0, padx=20, pady=(10, 0))
        self.sidebar_change_color_btn = ctk.CTkButton(self.sidebar, command=self.change_output_color_event)
        self.sidebar_change_color_btn.grid(row=5, column=0, padx=20, pady=10)
        
        #//////////// THEME SELECT ////////////
        self.appearance_mode_label = ctk.CTkLabel(self.sidebar, text="Appearance Mode:", anchor="sw")
        self.appearance_mode_label.grid(row=6, column=0, padx=20, pady=(10, 10), sticky="s")
        self.appearance_mode_option_menu = ctk.CTkOptionMenu(self.sidebar, values=["Light", "Dark", "System"], command=self.change_appearance_mode_event)
        self.appearance_mode_option_menu.grid(row=7, column=0, padx=20, pady=(0, 10))
        
        #//////////// UI SIZE SCALING SELECT ////////////
        self.scaling_label = ctk.CTkLabel(self.sidebar, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=8, column=0, padx=20, pady=(10, 0))
        self.scaling_option_menu = ctk.CTkOptionMenu(self.sidebar, values=["80%", "90%", "100%", "110%", "120%"], command=self.change_scaling_event)
        self.scaling_option_menu.grid(row=9, column=0, padx=20, pady=(10, 20))

        #//////////// COMMAND ENTRY ////////////
        self.command_entry = ctk.CTkEntry(self, placeholder_text="Enter a command. Try 'help' for a list of commands.")
        self.command_entry.grid(row=3, column=1, columnspan=1, padx=(20, 0), pady=(20, 20), sticky="nsew")
        #//////////// CLEAR BUTTON ////////////
        self.send_btn = ctk.CTkButton(self, fg_color="transparent", border_width=2, command=(self.process_input), text="Send Query")
        self.send_btn.grid(row=3, column=2, padx=(20, 20), pady=(20, 20), sticky="nsew")
        #//////////// SEND BUTTON ////////////
        self.clear_btn = ctk.CTkButton(master=self, fg_color="transparent", border_width=2, command=(self.clearAll), text="Clear")
        self.clear_btn.grid(row=3, column=3, padx=(20, 20), pady=(20, 20), sticky="nsew")

        #//////////// OUTPUT BOX ////////////
        self.output_box = ctk.CTkTextbox(self, width=250, font=ctk.CTkFont(size=14, weight='bold'))
        self.output_box.grid(row=0, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")

        #//////////// GTP OPTIONS TAB VIEW ////////////
        ###############################################################################
        self.gtp_options_tabview = ctk.CTkTabview(self, width=250)
        self.gtp_options_tabview.grid(row=0, column=2, padx=(20, 0), pady=(0, 0), sticky="nsew")
        self.gtp_options_tabview.add("Response")
        self.gtp_options_tabview.add("API")
        self.gtp_options_tabview.add("Data")
        self.gtp_options_tabview.tab("Response").grid_columnconfigure(0, weight=1)
        self.gtp_options_tabview.tab("API").grid_columnconfigure(0, weight=1)
        self.gtp_options_tabview.tab("Data").grid_columnconfigure(0, weight=1)
        
        self.response_tab_slider_frame = ctk.CTkScrollableFrame(self.gtp_options_tabview.tab("Response"))
        self.response_tab_slider_frame.grid(row=0, column=0, padx=(0, 0), pady=(0, 0), sticky="nsew")
        self.response_tab_slider_frame.grid_columnconfigure(0, weight=1)
        
        self.api_tab_slider_frame = ctk.CTkScrollableFrame(self.gtp_options_tabview.tab("API"))
        self.api_tab_slider_frame.grid(row=0, column=0, padx=(0, 0), pady=(0, 0), sticky="nsew")
        self.api_tab_slider_frame.grid_columnconfigure(0, weight=1)
        
        self.data_tab_slider_frame = ctk.CTkScrollableFrame(self.gtp_options_tabview.tab("Data"))
        self.data_tab_slider_frame.grid(row=0, column=0, padx=(0, 0), pady=(0, 0), sticky="nsew")
        self.data_tab_slider_frame.grid_columnconfigure(0, weight=1)
        
        #//////////// ENGINE OPTIONS ///////////
        self.engine_option_menu = ctk.CTkOptionMenu(
            self.response_tab_slider_frame, 
            dynamic_resizing=False, 
            values=self.cli.getEngines(),
            command=self.on_engine_option_chosen_event
        )
        self.engine_option_menu.grid(row=0, column=0, columnspan=2, sticky="ew", padx=(20, 10), pady=(10, 0))
        
        #//////////// RESPONSE TOKEN LIMIT INPUT ////////////
        self.response_token_limit_input = ctk.CTkButton(self.response_tab_slider_frame, text=f"Token Limit", command=self.open_response_token_limit_input_dialog_event)
        self.response_token_limit_input.grid(row=1, column=0, sticky="ew", padx=(20, 10), pady=(10, 0))
        
        self.response_token_limit_output = ctk.CTkLabel(self.response_tab_slider_frame, fg_color="#2B2B2B", corner_radius=6)
        self.response_token_limit_output.grid(row=1, column=1, sticky="ew", padx=(0, 10), pady=(10, 0))
        
        #//////////// RESPONSE COUNT INPUT ////////////
        self.response_count_input = ctk.CTkButton(self.response_tab_slider_frame, text=f"Response Count", command=self.open_response_count_input_dialog_event)
        self.response_count_input.grid(row=2, column=0, sticky="ew", padx=(20, 10), pady=(10, 0))
        
        self.response_count_output = ctk.CTkLabel(self.response_tab_slider_frame, fg_color="#2B2B2B", corner_radius=6)
        self.response_count_output.grid(row=2, column=1, sticky="ew", padx=(0, 10), pady=(10, 0))
        
        #//////////// BEST OF ////////////
        self.best_of_input = ctk.CTkButton(self.response_tab_slider_frame, text=f"Best of", command=self.open_best_of_input_dialog_event)
        self.best_of_input.grid(row=3, column=0, sticky="ew", padx=(20, 10), pady=(10, 0))
        
        self.best_of_output = ctk.CTkLabel(self.response_tab_slider_frame, fg_color="#2B2B2B", corner_radius=6)
        self.best_of_output.grid(row=3, column=1, sticky="ew", padx=(0, 10), pady=(10, 0))

        #//////////// FREQUENCY PENALTY ////////////
        self.frequency_penalty_input = ctk.CTkButton(self.response_tab_slider_frame, text=f"Frequency Penalty", command=self.open_frequency_penalty_input_dialog_event)
        self.frequency_penalty_input.grid(row=4, column=0, sticky="ew", padx=(20, 10), pady=(10, 0))
        
        self.frequency_penalty_output = ctk.CTkLabel(self.response_tab_slider_frame, fg_color="#2B2B2B", corner_radius=6)
        self.frequency_penalty_output.grid(row=4, column=1, sticky="ew", padx=(0, 10), pady=(10, 0))
        
        #//////////// PRESENCE PENALTY ////////////
        self.presence_penalty_input = ctk.CTkButton(self.response_tab_slider_frame, text=f"Presence Penalty", command=self.open_presence_penalty_input_dialog_event)
        self.presence_penalty_input.grid(row=5, column=0, sticky="ew", padx=(20, 10), pady=(10, 0))
        
        self.presence_penalty_output = ctk.CTkLabel(self.response_tab_slider_frame, fg_color="#2B2B2B", corner_radius=6)
        self.presence_penalty_output.grid(row=5, column=1, sticky="ew", padx=(0, 10), pady=(10, 0))
        
        #//////////// TIMEOUT ////////////
        self.timeout_input = ctk.CTkButton(self.response_tab_slider_frame, text=f"Time out", command=self.open_timeout_input_dialog_event)
        self.timeout_input.grid(row=6, column=0, sticky="ew", padx=(20, 10), pady=(10, 30))
        
        self.timeout_output = ctk.CTkLabel(self.response_tab_slider_frame, fg_color="#2B2B2B", corner_radius=6)
        self.timeout_output.grid(row=6, column=1, sticky="ew", padx=(0, 10), pady=(10, 30))

        #//////////// API BASE INPUT ////////////
        self.api_base_input = ctk.CTkButton(self.api_tab_slider_frame, text="API Base", command=self.open_api_base_input_dialog_event)
        self.api_base_input.grid(row=0, column=0, padx=20, pady=(10, 10), sticky="ew")
        
        self.api_base_output = ctk.CTkLabel(self.api_tab_slider_frame, fg_color="#2B2B2B", corner_radius=6)
        self.api_base_output.grid(row=1, column=0, sticky="ew", padx=(20, 20), pady=(0, 10))
        
        #//////////// API TYPE INPUT ////////////
        self.api_type_input = ctk.CTkButton(self.api_tab_slider_frame, text="API Type", command=self.open_api_type_input_dialog_event)
        self.api_type_input.grid(row=2, column=0, padx=20, pady=(10, 10), sticky="ew")
        
        self.api_type_output = ctk.CTkLabel(self.api_tab_slider_frame, fg_color="#2B2B2B", corner_radius=6)
        self.api_type_output.grid(row=3, column=0, sticky="ew", padx=(20, 20), pady=(0, 10))
        
        #//////////// API VERSION INPUT ////////////
        self.api_version_input = ctk.CTkButton(self.api_tab_slider_frame, text="API Version", command=self.open_api_version_input_dialog_event)
        self.api_version_input.grid(row=4, column=0, padx=20, pady=(10, 10), sticky="ew")
        
        self.api_version_output = ctk.CTkLabel(self.api_tab_slider_frame, fg_color="#2B2B2B", corner_radius=6)
        self.api_version_output.grid(row=5, column=0, sticky="ew", padx=(20, 20), pady=(0, 20))
        
        #//////////// ORGANIZATION INPUT ////////////
        self.organization_input = ctk.CTkButton(self.data_tab_slider_frame, text="Organization", command=self.open_organization_input_dialog_event)
        self.organization_input.grid(row=0, column=0, padx=20, pady=(10, 10))
        
        self.organization_output = ctk.CTkLabel(self.data_tab_slider_frame, fg_color="#2B2B2B", corner_radius=6)
        self.organization_output.grid(row=1, column=0, sticky="ew", padx=(0, 0), pady=(0, 10))
        
        #//////////// USER DEFINED DATA FILE INPUT ////////////
        self.user_defined_datafile_input = ctk.CTkButton(self.data_tab_slider_frame, text="User Defined Data-File", command=self.open_user_defined_datafile_input_dialog_event)
        self.user_defined_datafile_input.grid(row=2, column=0, padx=20, pady=(10, 10))
        
        self.user_defined_datafile_output = ctk.CTkLabel(self.data_tab_slider_frame, fg_color="#2B2B2B", corner_radius=6)
        self.user_defined_datafile_output.grid(row=3, column=0, sticky="ew", padx=(0, 0), pady=(0, 10))
        
        #//////////// JSONL DATA FILE INPUT ////////////
        self.jsonl_data_file_input = ctk.CTkButton(self.data_tab_slider_frame, text="JSONL Data File", command=self.open_jsonl_datafile_input_dialog_event)
        self.jsonl_data_file_input.grid(row=4, column=0, padx=20, pady=(10, 10))
    
        self.jsonl_datafile_output = ctk.CTkLabel(self.data_tab_slider_frame, fg_color="#2B2B2B", corner_radius=6)
        self.jsonl_datafile_output.grid(row=5, column=0, sticky="ew", padx=(0, 0), pady=(0, 20))
        
        #//////////// TRANSLATIONS FORM ////////////
        self.translation_form = tf.TranslationsForm(self)
        
        #//////////// Q&A FORM ////////////
        self.qa_form = qaf.QAForm(self)
        
        #//////////// IMAGE FORM ////////////
        self.img_form = imgf.ImageForm(self)
        
        #//////////// EMBEDDINGS VIEW ////////////
        self.embeddings_view = ev.EmbeddingsView(self)
        
        #//////////// AUDIO TRANSCRIPTION VIEW ////////////
        self.transcriptions_view = atv.AudioTranscriptionView(self)
        
        #//////////// EDIT VIEW ////////////
        self.edit_view = edv.EditView(self)
        
        #//////////// OUTPUT TEMPERATURE RADIO GROUP ////////////            
        self.output_temp_radiobutton_frame = ctk.CTkScrollableFrame(self)
        self.output_temp_radiobutton_frame.grid(row=0, column=3, padx=(20, 20), pady=(20, 0), sticky="nsew")
        self.output_temp_radio_var = tkinter.IntVar(value=0)
    
        self.output_temp_label_radio_group = ctk.CTkLabel(
            master=self.output_temp_radiobutton_frame, 
            text="Chat Output Temperature"
        )
        self.output_temp_label_radio_group.grid(row=0, column=2, columnspan=1, padx=10, pady=10, sticky="")
        
        self.temp_high_radio_button = ctk.CTkRadioButton(
            master=self.output_temp_radiobutton_frame, 
            text="high", 
            variable=self.output_temp_radio_var, 
            value=self.cli.temps["high"], 
            command=self.output_temp_radio_btn_selected
        )
        self.temp_high_radio_button.grid(row=1, column=2, pady=10, padx=20, sticky="nw")
        
        self.temp_high_color_box = ctk.CTkLabel(master=self.output_temp_radiobutton_frame, width=36, corner_radius=6, text="")
        self.temp_high_color_box.grid(row=1, column=3, sticky="w", padx=(0, 10), pady=(0, 10))
        
        self.temp_med_high_radio_button = ctk.CTkRadioButton(
            master=self.output_temp_radiobutton_frame, 
            text="mid-high", 
            variable=self.output_temp_radio_var, 
            value=self.cli.temps["med_high"], 
            command=self.output_temp_radio_btn_selected
        )
        self.temp_med_high_radio_button.grid(row=2, column=2, pady=10, padx=20, sticky="nw")
        
        self.temp_med_high_color_box = ctk.CTkLabel(master=self.output_temp_radiobutton_frame, width=36, corner_radius=6, text="")
        self.temp_med_high_color_box.grid(row=2, column=3, sticky="w", padx=(0, 10))
        
        self.temp_medium_radio_button = ctk.CTkRadioButton(
            master=self.output_temp_radiobutton_frame, 
            text="medium", 
            variable=self.output_temp_radio_var, 
            value=self.cli.temps["medium"], 
            command=self.output_temp_radio_btn_selected
        )
        self.temp_medium_radio_button.grid(row=3, column=2, pady=10, padx=20, sticky="nw")
        
        self.temp_medium_color_box = ctk.CTkLabel(master=self.output_temp_radiobutton_frame, width=36, corner_radius=6, text="")
        self.temp_medium_color_box.grid(row=3, column=3, sticky="w", padx=(0, 10))
        
        self.temp_med_low_radio_button = ctk.CTkRadioButton(
            master=self.output_temp_radiobutton_frame, 
            text="mid-low", 
            variable=self.output_temp_radio_var, 
            value=self.cli.temps["med_low"], 
            command=self.output_temp_radio_btn_selected
        )
        self.temp_med_low_radio_button.grid(row=4, column=2, pady=10, padx=20, sticky="nw")
        
        self.temp_med_low_color_box = ctk.CTkLabel(master=self.output_temp_radiobutton_frame, width=36, corner_radius=6, text="")
        self.temp_med_low_color_box.grid(row=4, column=3, sticky="w", padx=(0, 10))
        
        self.temp_low_radio_button = ctk.CTkRadioButton(
            master=self.output_temp_radiobutton_frame, 
            variable=self.output_temp_radio_var, 
            value=self.cli.temps["low"], 
            command=self.output_temp_radio_btn_selected
        )
        self.temp_low_radio_button.grid(row=5, column=2, pady=10, padx=20, sticky="nw")
        
        self.temp_low_color_box = ctk.CTkLabel(master=self.output_temp_radiobutton_frame, width=36, corner_radius=6, text="")
        self.temp_low_color_box.grid(row=5, column=3, sticky="w", padx=(0, 10))
        
        #//////////// INPUT BOX FRAME ////////////
        self.input_box_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.input_box_frame.grid(row=1, column=1, padx=(20, 0), pady=(0, 0), sticky="nsew")
        self.input_box_frame.grid_columnconfigure(0, weight=1)
        self.input_box_frame.grid_rowconfigure(0, weight=1)
        
        #//////////// INPUT BOX ////////////
        self.input_box = ctktextbox.CustomTkTextbox(
            self.input_box_frame, 
            font=ctk.CTkFont('Segoi UI', size=16)
        )
        self.input_box.grid(row=0, column=0, sticky='nsew')

        #//////////// SETTINGS SWITCHES ////////////
        self.settings_switches_frame = ctk.CTkScrollableFrame(self, label_text="Turn Settings on/off")
        self.settings_switches_frame.grid(row=1, column=2, padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.settings_switches_frame.grid_columnconfigure(0, weight=1)
        
        #//////////// CHAT ECHO ////////////    
        self.chat_echo_switch = ctk.CTkSwitch(master=self.settings_switches_frame, text=f"Echo", command=self.on_chat_echo_switch_changed_event)
        self.chat_echo_switch.grid(row=0, column=0, padx=10, pady=(0, 20))
        
        #//////////// CHAT STREAM ////////////
        self.chat_stream_switch = ctk.CTkSwitch(master=self.settings_switches_frame, text=f"Stream", command=self.on_chat_stream_switch_changed_event)
        self.chat_stream_switch.grid(row=1, column=0, padx=10, pady=(0, 20))
        
        #//////////// STOP LIST ////////////
        self.chat_stop_list_switch = ctk.CTkSwitch(master=self.settings_switches_frame, text=f"Stop List", command=self.on_chat_stop_list_switch_changed_event)
        self.chat_stop_list_switch.grid(row=2, column=0, padx=10, pady=(0, 20))
        
        #//////////// WRITE CHAT ////////////
        self.save_chat_switch = ctk.CTkSwitch(master=self.settings_switches_frame, text=f"Save Chat", command=self.on_save_chat_switch_changed_event)
        self.save_chat_switch.grid(row=3, column=0, padx=10, pady=(0, 20))

        #//////////// REQUEST TYPE RADIO GROUP ////////////
        self.request_type_slider_frame = ctk.CTkScrollableFrame(self)
        self.request_type_slider_frame.grid(row=1, column=3, padx=(20, 20), pady=(20, 0), sticky="nsew")
        self.request_type_slider_frame.grid_columnconfigure(0, weight=1)
        self.request_type_radio_var = tkinter.IntVar(value=0)
        
        self.output_request_type_radio_group = ctk.CTkLabel(
            master=self.request_type_slider_frame, 
            text="Request Types"
        )
        self.output_request_type_radio_group.grid(row=0, column=0, columnspan=1, sticky="nsew")

        self.chat_radio_btn = ctk.CTkRadioButton(
            master=self.output_request_type_radio_group,
            text="Chat",
            variable=self.request_type_radio_var,
            value=self.cli.request_types["chat"],
            command=self.request_type_radio_btn_selected
        )
        self.chat_radio_btn.grid(row=1, column=0, pady=(20, 0), padx=20, sticky="nw")
        
        self.images_radio_btn = ctk.CTkRadioButton(
            master=self.output_request_type_radio_group,
            text="Images",
            variable=self.request_type_radio_var,
            value=self.cli.request_types["images"],
            command=self.request_type_radio_btn_selected
            
        )
        self.images_radio_btn.grid(row=2, column=0, pady=(20, 0), padx=20, sticky="nw")
        
        self.audio_radio_btn = ctk.CTkRadioButton(
            master=self.output_request_type_radio_group,
            text="Audio",
            variable=self.request_type_radio_var,
            value=self.cli.request_types["audio"],
            command=self.request_type_radio_btn_selected
            
        )
        self.audio_radio_btn.grid(row=3, column=0, pady=(20, 0), padx=20, sticky="nw")
        
        self.translation_radio_btn = ctk.CTkRadioButton(
            master=self.output_request_type_radio_group,
            text="Translations",
            variable=self.request_type_radio_var,
            value=self.cli.request_types["translation"],
            command=self.request_type_radio_btn_selected
            
        )
        self.translation_radio_btn.grid(row=4, column=0, pady=(20, 0), padx=20, sticky="nw")
        
        self.embeddings_radio_btn = ctk.CTkRadioButton(
            master=self.output_request_type_radio_group,
            text="Embeddings",
            variable=self.request_type_radio_var,
            value=self.cli.request_types["embeddings"],
            command=self.request_type_radio_btn_selected
        )
        self.embeddings_radio_btn.grid(row=5, column=0, pady=(20, 0), padx=20, sticky="nw")
        
        self.files_radio_btn = ctk.CTkRadioButton(
            master=self.output_request_type_radio_group,
            text="Files",
            variable=self.request_type_radio_var,
            value=self.cli.request_types["files"],
            command=self.request_type_radio_btn_selected
        )
        self.files_radio_btn.grid(row=6, column=0, pady=(20, 0), padx=20, sticky="nw")
        
        self.fine_tuning_radio_btn = ctk.CTkRadioButton(
            master=self.output_request_type_radio_group,
            text="Fine Tuning",
            variable=self.request_type_radio_var,   
            value=self.cli.request_types["fine_tuning"], 
            command=self.request_type_radio_btn_selected
        )
        self.fine_tuning_radio_btn.grid(row=7, column=0, pady=(20, 0), padx=20, sticky="nw")
        
        self.moderations_radio_btn = ctk.CTkRadioButton(
            master=self.output_request_type_radio_group,
            text="Moderations",
            variable=self.request_type_radio_var,
            value=self.cli.request_types["moderations"],
            command=self.request_type_radio_btn_selected
        )
        self.moderations_radio_btn.grid(row=8, column=0, pady=(20, 0), padx=20, sticky="nw")
        
        self.build_request_radio_btn = ctk.CTkRadioButton(
            master=self.output_request_type_radio_group,
            text="Build Requests",
            variable=self.request_type_radio_var,
            value=self.cli.request_types["build_requests"],
            command=self.request_type_radio_btn_selected    
        )
        self.build_request_radio_btn.grid(row=9, column=0, pady=(20, 0), padx=20, sticky="nw")
        
        self.sentement_radio_btn = ctk.CTkRadioButton(
            master=self.output_request_type_radio_group,
            text="Sentement Analysis",
            variable=self.request_type_radio_var,
            value=self.cli.request_types["sentement"],
            command=self.request_type_radio_btn_selected    
        )
        self.sentement_radio_btn.grid(row=10, column=0, pady=(20, 0), padx=20, sticky="nw")

        self.qa_radio_btn = ctk.CTkRadioButton(
            master=self.output_request_type_radio_group,
            text="Q & A",
            variable=self.request_type_radio_var,
            value=self.cli.request_types["qa"],
            command=self.request_type_radio_btn_selected    
        )
        self.qa_radio_btn.grid(row=11, column=0, pady=(20, 0), padx=20, sticky="nw")

        self.summarization_radio_btn = ctk.CTkRadioButton(
            master=self.output_request_type_radio_group,
            text="Summarization",
            variable=self.request_type_radio_var,
            value=self.cli.request_types["summarization"],
            command=self.request_type_radio_btn_selected    
        )
        self.summarization_radio_btn.grid(row=12, column=0, pady=(20, 0), padx=20, sticky="nw")
        
        self.code_gen_radio_btn = ctk.CTkRadioButton(
            master=self.output_request_type_radio_group,
            text="Code Generation",
            variable=self.request_type_radio_var,
            value=self.cli.request_types["code_gen"],
            command=self.request_type_radio_btn_selected    
        )
        self.code_gen_radio_btn.grid(row=13, column=0, pady=(20, 0), padx=20, sticky="nw")
        
        self.edit_radio_btn = ctk.CTkRadioButton(
            master=self.output_request_type_radio_group,
            text="Edits",
            variable=self.request_type_radio_var,
            value=self.cli.request_types["edits"],
            command=self.request_type_radio_btn_selected    
        )
        self.edit_radio_btn.grid(row=14, column=0, pady=(20, 0), padx=20, sticky="nw")
        
        
        #//////////// DEFAULT VALUES ////////////
        self.sidebar_username_btn.configure(state="normal", text=" Username")
        self.sidebar_exit_btn.configure(state="normal", text="Exit")
        self.sidebar_set_key_btn.configure(state="normal", text="API Key")
        self.sidebar_change_color_btn.configure(state="normal", text="Color")

        self.temp_high_radio_button.configure(text="High")
        self.temp_med_high_radio_button.configure(text="Med-High")
        self.temp_medium_radio_button.configure(text="Medium")
        self.temp_med_low_radio_button.configure(text="Med-Low")
        self.temp_low_radio_button.configure(text="Low")
        
        #////// GUI LOADED //////
        self.loadOptions()
        self._config.saveConfig()
        self.setGUIShownFlag()
        if self.REQUEST_TYPE == 0:
            self.setOutput(self.cli.greetUser(self.USER, self.KEY_CONFIG_FILE), "chat")
        elif self.REQUEST_TYPE == 1:
            self.setOutput("Commencing image requests ...")
        elif self.REQUEST_TYPE == 2:
            self.setOutput("Commencing audio requests ...")
        elif self.REQUEST_TYPE == 3:
            self.setOutput("Commencing embeddings ...")
        elif self.REQUEST_TYPE == 4:
            self.setOutput("Commencing file requests ...")
        elif self.REQUEST_TYPE == 5:
            self.setOutput("Commencing fine-tuning ...")
        elif self.REQUEST_TYPE == 6:
            self.setOutput("Commencing moderations ...")
        elif self.REQUEST_TYPE == 7:
            self.setOutput("Commencing build requests ...")
        elif self.REQUEST_TYPE == 8:
            self.setOutput("Commencing translations ...")
        elif self.REQUEST_TYPE == 9:
            self.setOutput("Commencing sentement analysis ...")
        elif self.REQUEST_TYPE == 10:
            self.setOutput("Commencing q & a ...")
        elif self.REQUEST_TYPE == 11:
            self.setOutput("Commencing summarization ...")
        elif self.REQUEST_TYPE == 12:
            self.setOutput("Commencing code generation ...")
        elif self.REQUEST_TYPE == 13:
            self.setOutput("Commencing edits ...")
        
        #//////////// GUI METHODS ////////////

    def setGUIShownFlag(self):
        self.stdops.createFile(self.GUI_SHOWN_FLAG_FILE)
    
    def loadOptions(self):
        self.output_box.configure(text_color=self._OUTPUT_COLOR)
        self.send_btn.configure(border_color=self._OUTPUT_COLOR)
        self.clear_btn.configure(border_color=self._OUTPUT_COLOR)
        self.command_entry.configure(border_color=self._OUTPUT_COLOR)
        self.sidebar_logo.configure(text_color=self._OUTPUT_COLOR)
        self.appearance_mode_label.configure(text_color=self._OUTPUT_COLOR)
        self.scaling_label.configure(text_color=self._OUTPUT_COLOR)
        self.output_temp_label_radio_group.configure(text_color=self._OUTPUT_COLOR)
        self.change_color_btn_label.configure(text_color=self._OUTPUT_COLOR)
        self.settings_switches_frame.configure(label_text_color=self._OUTPUT_COLOR)
        self.output_request_type_radio_group.configure(text_color=self._OUTPUT_COLOR)
        
        self.response_token_limit_output.configure(text=f"{self._config.getOption('chat', 'response_token_limit')}")
        self.response_count_output.configure(text=f"{self._config.getOption('chat', 'response_count')}")
        self.best_of_output.configure(text=f"{self._config.getOption('chat', 'best_of')}")
        self.frequency_penalty_output.configure(text=f"{self._config.getOption('chat', 'frequency_penalty')}")
        self.presence_penalty_output.configure(text=f"{self._config.getOption('chat', 'presence_penalty')}")
        self.timeout_output.configure(text=f"{self._config.getOption('chat', 'timeout')}")
        self.api_base_output.configure(text=f"{self._config.getOption('chat', 'api_base')}")
        self.api_type_output.configure(text=f"{self._config.getOption('chat', 'api_type')}")
        self.api_version_output.configure(text=f"{self._config.getOption('chat', 'api_version')}")
        self.organization_output.configure(text=f"{self._config.getOption('user', 'organization')}")
        self.user_defined_datafile_output.configure(text=f"{self._config.getOption('chat', 'user_defined_data_file')}")
        self.jsonl_datafile_output.configure(text=f"{self._config.getOption('chat', 'jsonl_data_file')}")
        
        if self._config.getOption("chat", "echo_chat") == "True":
            self.chat_echo_switch.select()
            self.ECHO_CHAT = True
            
        if self._config.getOption("chat", "stream_chat") == "True":
            self.chat_stream_switch.select()
            self.CHAT_LOG_PATH = self._config.getOption("chat", "chat_log_path")

        if self._config.getOption("chat", "use_stop_list") == "True":
            self.chat_stop_list_switch.select()
            self.USE_STOP_LIST = True
            
        if self._config.getOption("chat", "chat_to_file") == "True":
            self.save_chat_switch.select()
            self.SAVE_CHAT = True         
            
        if self._config.getOption("chat", "stream_chat") == "True":
            self.chat_stream_switch.select()
            self.STREAM_CHAT = True
            
        #////// OUTPUT TEMP RADIO GROUP
        output_temp = float(self._config.getOption("chat", "chat_temperature"))
        if output_temp == 0:
            self.temp_low_radio_button.select()
            self.output_temp_radio_btn_selected(True)
        elif output_temp == 0.5:
            self.temp_med_low_radio_button.select()
            self.output_temp_radio_btn_selected(True)
        elif output_temp == 1:
            self.temp_medium_radio_button.select()
            self.output_temp_radio_btn_selected(True)
        elif output_temp == 1.5:
            self.temp_med_high_radio_button.select()
            self.output_temp_radio_btn_selected(True)
        else:
            self.temp_high_radio_button.select()
            self.output_temp_radio_btn_selected(True)
        #     self.request_types = {
        #     "chat": 0, 
        #     "images": 1, 
        #     "audio" : 2, 
        #     "embeddings": 3, 
        #     "files": 4, 
        #     "fine_tuning": 5, 
        #     "moderations": 6, 
        #     "build_requests": 7,
        #     "translation": 8,
        #     "sentement": 9
        # }
        _request_type = int(self._config.getOption("chat", "request_type"))
        if _request_type == 0:
            self.chat_radio_btn.select()
            self.request_type_radio_btn_selected()
        elif _request_type == 1:
            self.images_radio_btn.select()
            self.request_type_radio_btn_selected()
        elif _request_type == 2:
            self.audio_radio_btn.select()
            self.request_type_radio_btn_selected()
        elif _request_type == 3:
            self.embeddings_radio_btn.select()
            self.request_type_radio_btn_selected()
        elif _request_type == 4:
            self.files_radio_btn.select()
            self.request_type_radio_btn_selected()
        elif _request_type == 5:
            self.fine_tuning_radio_btn.select()
            self.request_type_radio_btn_selected()
        elif _request_type == 6:
            self.moderations_radio_btn.select()
            self.request_type_radio_btn_selected()
        elif _request_type == 7:
            self.build_request_radio_btn.select()
            self.request_type_radio_btn_selected()
        elif _request_type == 8:
            self.translation_radio_btn.select()
            self.request_type_radio_btn_selected()
        elif _request_type == 9:
            self.sentement_radio_btn.select()
            self.request_type_radio_btn_selected()
        elif _request_type == 10:
            self.qa_radio_btn.select()
            self.request_type_radio_btn_selected()
        elif _request_type == 11:
            self.summarization_radio_btn.select()
            self.request_type_radio_btn_selected()
        elif _request_type == 12:
            self.code_gen_radio_btn.select()
            self.request_type_radio_btn_selected()
        elif _request_type == 13:
            self.edit_radio_btn.select()
            self.request_type_radio_btn_selected()
            
        self.engine_option_menu.set(self._config.getOption("chat", "chat_engine"))
        self.appearance_mode_option_menu.set(self._config.getOption("ui", "theme"))
        self.scaling_option_menu.set(self._config.getOption("ui", "ui_scaling"))
        
    def clearAll(self):
        self.clearInput()
        self.clearOutput()
        self.command_entry.delete(0, tk.END)     
    
    def on_save_chat_switch_changed_event(self) -> None:
        state = self.save_chat_switch.get()
        if state == 0:
            self.SAVE_CHAT = False
            self.setOutput("[Save chat]: Off", "cli")
            self._config.openConfig()
            self._config.setOption("chat", "chat_to_file", False)
            self._config.saveConfig()
        else:
            if self.CHAT_LOG_PATH == None:
                self.CHAT_LOG_PATH = self.openFileDialog()
                self.SAVE_CHAT = True
                self.setOutput("[Save chat]: On", "cli")
            else:
                self.SAVE_CHAT = True
                self.setOutput("[Save chat]: On", "cli")
            self._config.openConfig()
            self._config.setOption("chat", "chat_to_file", True)
            self._config.setOption("chat", "chat_log_path", self.CHAT_LOG_PATH)
            self._config.saveConfig()
            
    def openFileDialog(self) -> (str | bool):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.setOutput(f"Selected file: {file_path}", "cli")
            return file_path
        else:
            self.setOutput(f"No file selected", "cli")
            return False
            
    def on_chat_echo_switch_changed_event(self) -> None:
        state = self.chat_echo_switch.get()
        if state == 0:
            self.cli.setChatEcho(False)
            self.setOutput("[Chat echo]: Off", "cli")
            self._config.openConfig()
            self._config.setOption("chat", "echo_chat", False)
            self._config.saveConfig()
        else:
            self.cli.setChatEcho(True)
            self.setOutput("[Chat echo]: On", "cli")
            self._config.openConfig(    )
            self._config.setOption("chat", "echo_chat", True)
            self._config.saveConfig()

    def on_chat_stream_switch_changed_event(self) -> None:
        state = self.chat_stream_switch.get()
        if state == 0:
            self.cli.setChatStream(False)
            self.setOutput("[Chat stream]: Off", "cli")
            self._config.openConfig()
            self._config.setOption("chat", "stream_chat", False)
            self._config.saveConfig()
        else:
            self.cli.setChatStream(True)
            self.setOutput("[Chat stream]: On", "cli")
            self._config.openConfig()
            self._config.setOption("chat", "stream_chat", True)
            self._config.saveConfig()
            
    def on_chat_stop_list_switch_changed_event(self) -> None:
        state = self.chat_stop_list_switch.get()
        if state == 0:
            self.setOutput("[Chat stop list]: Off", "cli")
            self._config.openConfig()
            self._config.setOption("chat", "use_stop_list", False)
            self._config.saveConfig()
        else:
            dialog = ctk.CTkInputDialog(text="Enter a list of words you want the output to stop at if encountered\nWords should be quoted!: ", title="Chat Stop List")
            _stoplist = str(dialog.get_input())
            if len(_stoplist) != 0 and _stoplist != "None":
                self.cli.setStopList(_stoplist)
                self.setOutput("[Chat stop list]: On", "cli")
                self._config.openConfig()
                self._config.setOption("chat", "use_stop_list", True)
                self._config.saveConfig()
                return True
            else:
                self._config.openConfig()
                self._config.setOption("chat", "use_stop_list", False)
                self._config.saveConfig()
                return False
            
    def on_engine_option_chosen_event(self, engine) -> None:
        self.cli.setEngine(f"{engine}")
        self.CHAT_ENGINE = engine
        self.setOutput(f"Engine changed to: {engine}", "cli")
            
    def open_jsonl_datafile_input_dialog_event(self) -> bool:
        _path = self.openFileDialog()
        if _path == "None":
            return False
        if _path == False:
            return False
        self.cli.setJSONLDataFile(_path)
        self.setOutput(f"File set: [{self.cli.getJSONLDataFile()}]", "cli")
        self.jsonl_datafile_output.configure(text=f"{_path}")
        return True
 
    def open_user_defined_datafile_input_dialog_event(self) -> bool:
        _path = self.openFileDialog()
        if _path == "None":
            return False
        if _path == False:
            return False
        self.cli.setUserDefinedFileName(_path)
        self.setOutput(f"File set: [{self.cli.getUserDefinedFileName()}]", "cli")
        self.user_defined_datafile_output.configure(text=f"{_path}")
        return True
  
    def open_organization_input_dialog_event(self) -> bool:
        dialog = ctk.CTkInputDialog(text="Enter organization: ", title="Organization Input")
        organization = str(dialog.get_input())
        if len(organization) != 0 and organization != "None":
            self.cli.setOrganization(organization)
            self.ORGANIZATION = organization
            self.setOutput(f"Organization changed to: [{self.cli.getOrganization()}]", "cli")
            self.organization_output.configure(text=f"{organization}")
            return True
        else:
            return False
        
    def open_api_version_input_dialog_event(self) -> bool:
        dialog = ctk.CTkInputDialog(text="Enter the API version: ", title="API Version Input")
        api_version = str(dialog.get_input())
        if len(api_version) != 0 and api_version != "None":
            self.cli.setAPIVersion(api_version)
            self.API_VERSION = api_version
            self.setOutput(f"API version changed to: [{self.cli.getAPIVersion()}]", "cli")
            self.api_version_output.configure(text=f"{api_version}")
            return True
        else:
            return False
            
    def open_api_type_input_dialog_event(self) -> bool:
        dialog = ctk.CTkInputDialog(text="Enter the API type: ", title="API Type Input")
        api_type = str(dialog.get_input())
        if len(api_type) != 0 and api_type != "None":
            self.cli.setAPIType(api_type)
            self.API_TYPE = api_type
            self.setOutput(f"API type changed to: [{self.cli.getAPIType()}]", "cli")
            self.api_type_output.configure(text=f"{api_type}")
            return True
        else:
            return False

    def open_api_base_input_dialog_event(self) -> bool:
        dialog = ctk.CTkInputDialog(text="Enter the API base: ", title="API Base Input")
        api_base = str(dialog.get_input())
        if len(api_base) != 0 and api_base != "None":
            self.cli.setAPIBase(api_base)
            self.API_BASE = api_base
            self.setOutput(f"API base changed to: [{self.cli.getAPIBase()}]", "cli")
            self.api_base_output.configure(text=f"{api_base}")
            return True
        else:
            return False

    def open_response_token_limit_input_dialog_event(self) -> bool: 
        dialog = ctk.CTkInputDialog(text="Enter the response token limit: ", title="Response Token Limit Input")
        _token_limit = str(dialog.get_input())
        if _token_limit.isdigit() and _token_limit != "None":
            self.cli.setResponseTokenLimit(int(_token_limit))
            self.RESPONSE_TOKEN_LIMIT = int(_token_limit)
            self.setOutput(f"Response token limit changed to: [{self.cli.getResponseTokenLimit()}]", "cli")
            self.response_token_limit_output.configure(text=f"{self.RESPONSE_TOKEN_LIMIT}")
            return True
        else:
            return False
        
    def open_response_count_input_dialog_event(self) -> bool:
        dialog = ctk.CTkInputDialog(text="Enter the response count: ", title="Response Count Input")
        _response_count = str(dialog.get_input())
        if _response_count.isdigit() and _response_count != "None":
            self.cli.setResponseCount(int(_response_count))
            self.RESPONSE_COUNT = int(_response_count)
            self.setOutput(f"Response count changed to: [{self.cli.getResponseCount()}]", "cli")
            self.response_count_output.configure(text=f"{self.RESPONSE_COUNT}")
            return True
        else:
            return False
    
    def open_best_of_input_dialog_event(self) -> bool:
        dialog = ctk.CTkInputDialog(text="Generates best_of completions: ", title="Best of Input")
        _best_of = str(dialog.get_input())
        if _best_of.isdigit() and _best_of != "None":
            self.cli.setBestOf(int(_best_of))
            self.BEST_OF = int(_best_of)
            self.setOutput(f"Best of changed to best of: [{self.cli.getBestOf()}]", "cli")
            self.best_of_output.configure(text=f"{self.BEST_OF}")
            return True
        else:
            return False
        
    def open_frequency_penalty_input_dialog_event(self) -> bool:
        dialog = ctk.CTkInputDialog(text="Change the response token frequency penalty: ", title="Frequency Penalty Input")
        _freq_penalty = str(dialog.get_input())
        if _freq_penalty.isdigit() and _freq_penalty != "None":
            self.cli.setFrequencyPenalty(int(_freq_penalty))
            self.FREQUENCY_PENALTY = int(_freq_penalty)
            self.setOutput(f"Frequency penalty changed to: [{self.cli.getFrequencyPenalty()}]", "cli")
            self.frequency_penalty_output.configure(text=f"{self.FREQUENCY_PENALTY}")
            return True
        else:
            return False
        
    def open_presence_penalty_input_dialog_event(self) -> bool:
        dialog = ctk.CTkInputDialog(text="Change the response token presence penalty: ", title="Presence Penalty Input")
        _presence_penalty = str(dialog.get_input())
        if _presence_penalty.isdigit() and _presence_penalty != "None":
            self.cli.setPresencePenalty(int(_presence_penalty))
            self.PRESENCE_PENALTY = int(_presence_penalty)
            self.setOutput(f"Presence penalty changed to: [{self.cli.getPresencePenalty()}]", "cli")
            self.presence_penalty_output.configure(text=f"{self.PRESENCE_PENALTY}")
            return True
        else:
            return False
    
    def open_timeout_input_dialog_event(self) -> bool:
        dialog = ctk.CTkInputDialog(text="Timeout length: ", title="Timeout Length Input")
        _timeout = str(dialog.get_input())
        if _timeout.isdigit() and _timeout != "None":
            self.cli.setTimeout(int(_timeout))
            self.TIMEOUT = int(_timeout)
            self.setOutput(f"Time out changed to: [{self.cli.getTimeout()}]", "cli")
            self.timeout_output.configure(text=f"{self.TIMEOUT}")
            return True
        else:
            return False
        
    def openChatOutputTempForm(self):
        self.translation_form.grid_forget()
        self.qa_form.grid_forget()
        self.img_form.grid_forget()
        self.embeddings_view.grid_forget()
        self.transcriptions_view.grid_forget()
        self.edit_view.grid_forget()
        self.output_temp_radiobutton_frame.grid(row=0, column=3, padx=(20, 20), pady=(20, 0), sticky="nsew")
        
    def openLanguageTranslationForm(self):
        self.output_temp_radiobutton_frame.grid_forget()
        self.qa_form.grid_forget()
        self.img_form.grid_forget()
        self.embeddings_view.grid_forget()
        self.transcriptions_view.grid_forget()
        self.edit_view.grid_forget()
        self.translation_form.grid(row=0, column=3, padx=(20, 20), pady=(20, 0), sticky="nsew")
        
    def openQAForm(self):
        self.output_temp_radiobutton_frame.grid_forget()
        self.translation_form.grid_forget()
        self.img_form.grid_forget()
        self.embeddings_view.grid_forget()
        self.transcriptions_view.grid_forget()
        self.edit_view.grid_forget()
        self.qa_form.grid(row=0, column=3, padx=(20, 20), pady=(20, 0), sticky="nsew")
        
    def openImageForm(self):
        self.output_temp_radiobutton_frame.grid_forget()
        self.translation_form.grid_forget()
        self.qa_form.grid_forget()
        self.embeddings_view.grid_forget()
        self.transcriptions_view.grid_forget()
        self.edit_view.grid_forget()
        self.img_form.grid(row=0, column=3, padx=(20, 20), pady=(20, 0), sticky="nsew")
       
    def openEmbeddingsView(self):
        self.output_temp_radiobutton_frame.grid_forget()
        self.translation_form.grid_forget()
        self.qa_form.grid_forget()
        self.img_form.grid_forget()
        self.transcriptions_view.grid_forget()
        self.edit_view.grid_forget()
        self.embeddings_view.grid(row=0, column=3, padx=(20, 20), pady=(20, 0), sticky="nsew")
         
    def openTranscriptionsView(self):
        self.output_temp_radiobutton_frame.grid_forget()
        self.translation_form.grid_forget()
        self.qa_form.grid_forget()
        self.img_form.grid_forget()
        self.embeddings_view.grid_forget()
        self.edit_view.grid_forget()
        self.transcriptions_view.grid(row=0, column=3, padx=(20, 20), pady=(20, 0), sticky="nsew")
     
    def openEditView(self):
        self.output_temp_radiobutton_frame.grid_forget()
        self.translation_form.grid_forget()
        self.qa_form.grid_forget()
        self.img_form.grid_forget()
        self.embeddings_view.grid_forget()
        self.transcriptions_view.grid_forget()
        self.edit_view.grid(row=0, column=3, padx=(20, 20), pady=(20, 0), sticky="nsew")
     
    def request_type_radio_btn_selected(self):
        selected_value = self.request_type_radio_var.get()
        if selected_value == self.cli.request_types["chat"]:
            self.cli.setRequestType(selected_value)
            self.REQUEST_TYPE = selected_value
            self.openChatOutputTempForm()
            self.setOutput(f"Request type set  to: ({selected_value}) Chat", "cli")
            
        elif selected_value == self.cli.request_types["images"]:
            self.cli.setRequestType(selected_value)
            self.REQUEST_TYPE = selected_value
            self.openImageForm()
            self.setOutput(f"Request type set to: ({selected_value}) Images", "cli")
            
        elif selected_value == self.cli.request_types["audio"]:
            self.cli.setRequestType(selected_value)
            self.REQUEST_TYPE = selected_value
            self.openTranscriptionsView()
            self.setOutput(f"Request type set to: ({selected_value}) Audio", "cli")
            
        elif selected_value == self.cli.request_types["embeddings"]:
            self.cli.setRequestType(selected_value)
            self.REQUEST_TYPE = selected_value
            self.openEmbeddingsView()
            self.setOutput(f"Request type set to: ({selected_value}) Embeddings", "cli")
            
        elif selected_value == self.cli.request_types["files"]:
            self.cli.setRequestType(selected_value)
            self.REQUEST_TYPE = selected_value
            self.setOutput(f"Request type set to: ({selected_value}) Files", "cli")
            
        elif selected_value == self.cli.request_types["fine_tuning"]:
            self.cli.setRequestType(selected_value)
            self.REQUEST_TYPE = selected_value
            self.setOutput(f"Request type set to: ({selected_value}) Fine-Tuning", "cli")
        
        elif selected_value == self.cli.request_types["moderations"]:
            self.cli.setRequestType(selected_value)
            self.REQUEST_TYPE = selected_value
            self.setOutput(f"Request type set to: ({selected_value}) Moderations", "cli")
        
        elif selected_value == self.cli.request_types["build_requests"]:
            self.cli.setRequestType(selected_value)
            self.REQUEST_TYPE = selected_value
            self.setOutput(f"Request type set to: ({selected_value}) Build Requests", "cli")
        
        elif selected_value == self.cli.request_types["translation"]:
            self.cli.setRequestType(selected_value)
            self.REQUEST_TYPE = selected_value
            self.openLanguageTranslationForm()
            self.setOutput(f"Request type set to: ({selected_value}) Translation", "cli")
            
        elif selected_value == self.cli.request_types["sentement"]:
            self.cli.setRequestType(selected_value)
            self.REQUEST_TYPE = selected_value
            self.setOutput(f"Request type set to: ({selected_value}) Senetement Analysis", "cli")
            
        elif selected_value == self.cli.request_types["qa"]:
            self.cli.setRequestType(selected_value)
            self.REQUEST_TYPE = selected_value
            self.openQAForm()
            self.setOutput(f"Request type set to: ({selected_value}) Q & A", "cli")
            
        elif selected_value == self.cli.request_types["summarization"]:
            self.cli.setRequestType(selected_value)
            self.REQUEST_TYPE = selected_value
            self.setOutput(f"Request type set to: ({selected_value}) Summarization", "cli")
            
        elif selected_value == self.cli.request_types["code_gen"]:
            self.cli.setRequestType(selected_value)
            self.REQUEST_TYPE = selected_value
            self.setOutput(f"Request type set to: ({selected_value}) Code Generation", "cli")
        
        elif selected_value == self.cli.request_types["edits"]:
            self.cli.setRequestType(selected_value)
            self.REQUEST_TYPE = selected_value
            self.openEditView()
            self.setOutput(f"Request type set to: ({selected_value}) Edits", "cli")
    
    def output_temp_radio_btn_selected(self, _initial: bool = False) -> bool:
        selected_value = self.output_temp_radio_var.get()
        if selected_value == self.cli.temps["high"]:
            self.cli.setTemperature(2)
            self.CHAT_TEMP = selected_value
            self.temp_high_color_box.configure(fg_color="#ff595e", text_color="#000000", text="2.0")
            self.temp_med_high_color_box.configure(fg_color="#343638", text="")
            self.temp_medium_color_box.configure(fg_color="#343638", text="")
            self.temp_med_low_color_box.configure(fg_color="#343638", text="")
            self.temp_low_color_box.configure(fg_color="#343638", text="")
            if not _initial:
                self.setOutput(f"Temperature changed to: High (2.0)", "cli")
        if selected_value == self.cli.temps["med_high"]:
            self.cli.setTemperature(1.5)
            self.CHAT_TEMP = selected_value
            self.temp_high_color_box.configure(fg_color="#343638", text="")
            self.temp_med_high_color_box.configure(fg_color="#f4a261", text_color="#000000", text="1.5")
            self.temp_medium_color_box.configure(fg_color="#343638", text="")
            self.temp_med_low_color_box.configure(fg_color="#343638", text="")
            self.temp_low_color_box.configure(fg_color="#343638", text="")
            if not _initial:
                self.setOutput(f"Temperature changed to: Med-High (1.5)", "cli")
        elif selected_value == self.cli.temps["medium"]:
            self.cli.setTemperature(1)
            self.CHAT_TEMP = selected_value
            self.temp_high_color_box.configure(fg_color="#343638", text="")
            self.temp_med_high_color_box.configure(fg_color="#343638", text="")
            self.temp_medium_color_box.configure(fg_color="#ffca3a", text_color="#000000", text="1.0")
            self.temp_med_low_color_box.configure(fg_color="#343638", text="")
            self.temp_low_color_box.configure(fg_color="#343638", text="")
            if not _initial:
                self.setOutput(f"Temperature changed to: Medium (1.0)", "cli")
        if selected_value == self.cli.temps["med_low"]:
            self.cli.setTemperature(0.5)
            self.CHAT_TEMP = selected_value
            self.temp_high_color_box.configure(fg_color="#343638", text="")
            self.temp_med_high_color_box.configure(fg_color="#343638", text="")
            self.temp_medium_color_box.configure(fg_color="#343638", text="")
            self.temp_med_low_color_box.configure(fg_color="#2a9d8f", text_color="#000000", text="0.5")
            self.temp_low_color_box.configure(fg_color="#343638", text="")
            if not _initial:
                self.setOutput(f"Temperature changed to: Med-Low (0.5)", "cli")
        elif selected_value == self.cli.temps["low"]:
            self.cli.setTemperature(0)
            self.CHAT_TEMP = selected_value
            self.temp_high_color_box.configure(fg_color="#343638", text="")
            self.temp_med_high_color_box.configure(fg_color="#343638", text="")
            self.temp_medium_color_box.configure(fg_color="#343638", text="")
            self.temp_med_low_color_box.configure(fg_color="#343638", text="")
            self.temp_low_color_box.configure(fg_color="#1982c4", text_color="#000000", text="0.0")
            if not _initial:
                self.setOutput(f"Temperature changed to: Low (0.0))", "cli")
        
    def change_appearance_mode_event(self, _new_appearance_mode: str) -> bool:
        _theme = _new_appearance_mode
        if len(_theme) != 0 and _theme != "None":
            ctk.set_appearance_mode(_theme)
            self.THEME = _theme
            self.setOutput(f"Appearance mode changed to: {self.THEME}", "cli")
            self._config.openConfig()
            self._config.setOption("ui", "theme", self.THEME)
            self._config.saveConfig()
            return True
        else:
            self.setOutput(f"Appearance mode: {_theme} doesn\'t exist!\nOptions are [Light|Dark|System]", "cli")
            return False
            
    def change_output_color_event(self) -> None:
        color = colorchooser.askcolor(title="Select Color")
        self._OUTPUT_COLOR = f"{color[1]}"
        self.output_box.configure(text_color=self._OUTPUT_COLOR)
        self.send_btn.configure(border_color=self._OUTPUT_COLOR)
        self.clear_btn.configure(border_color=self._OUTPUT_COLOR)
        self.command_entry.configure(border_color=self._OUTPUT_COLOR)
        self.sidebar_logo.configure(text_color=self._OUTPUT_COLOR)
        self.appearance_mode_label.configure(text_color=self._OUTPUT_COLOR)
        self.scaling_label.configure(text_color=self._OUTPUT_COLOR)
        self.output_temp_label_radio_group.configure(text_color=self._OUTPUT_COLOR)
        self.change_color_btn_label.configure(text_color=self._OUTPUT_COLOR)
        self.settings_switches_frame.configure(label_text_color=self._OUTPUT_COLOR)
        self.output_request_type_radio_group.configure(text_color=self._OUTPUT_COLOR)
     
        self._config.openConfig()
        self._config.setOption("ui", "color", f"{self._OUTPUT_COLOR}")
        self._config.saveConfig()
               
    def change_scaling_event(self, new_scaling: str) -> None:
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        self.UI_SCALING = new_scaling_float
        ctk.set_widget_scaling(new_scaling_float)
        self._config.openConfig()
        self._config.setOption("ui", "chat_temperature", f"{self.UI_SCALING}%%")
        self._config.saveConfig()

    def sidebar_username_btn_event(self) -> None:
        self.setUsername()
    
    def sidebar_exit_btn_event(self) -> None:
        sys.exit(0)
 
    def disableFunctionality(self) -> None:
        self.command_entry.configure(state="disabled", placeholder_text="Disabled! Enter an api key to re-enable functionality. https://openai.com/")
        self.input_box.configure(state="disabled")
        self.chat_echo_switch.configure(state="disabled")
        self.chat_stream_switch.configure(state="disabled")
        self.chat_stop_list_switch.configure(state="disabled")
        self.save_chat_switch.configure(state="disabled")
        self.chat_radio_btn.configure(state="disabled")
        self.images_radio_btn.configure(state="disabled")
        self.audio_radio_btn.configure(state="disabled")
        self.embeddings_radio_btn.configure(state="disabled")
        self.files_radio_btn.configure(state="disabled")
        self.fine_tuning_radio_btn.configure(state="disabled")
        self.moderations_radio_btn.configure(state="disabled")
        self.build_request_radio_btn.configure(state="disabled")
        self.temp_high_radio_button.configure(state="disabled")
        self.temp_medium_radio_button.configure(state="disabled")
        self.temp_low_radio_button.configure(state="disabled")
        
    def enableFunctionality(self) -> None:
        self.command_entry.configure(state="normal", placeholder_text="Enter a command. Try 'help' for a list of commands.")
        self.input_box.configure(state="normal")
        self.chat_echo_switch.configure(state="normal")
        self.chat_stream_switch.configure(state="normal")
        self.chat_stop_list_switch.configure(state="normal")
        self.save_chat_switch.configure(state="normal")
        self.chat_radio_btn.configure(state="normal")
        self.images_radio_btn.configure(state="normal")
        self.audio_radio_btn.configure(state="normal")
        self.embeddings_radio_btn.configure(state="normal")
        self.files_radio_btn.configure(state="normal")
        self.fine_tuning_radio_btn.configure(state="normal")
        self.moderations_radio_btn.configure(state="normal")
        self.build_request_radio_btn.configure(state="normal")
        self.temp_high_radio_button.configure(state="normal")
        self.temp_medium_radio_button.configure(state="normal")
        self.temp_low_radio_button.configure(state="normal")
        
    def sidebar_set_key_btn_event(self) -> None:
        dialog = ctk.CTkInputDialog(text=f"{self.USER} enter your OpenAI API key: ", title="API Key")
        api_key = str(dialog.get_input())
        if self.cli.validateAPIKey(api_key):
            if self.cli.setAPIKey(api_key):
                self.API_KEY = api_key
                self.setOutput("Api key has been set successfully!", "cli")
                self.enableFunctionality()
            else:
                self.disableFunctionality()
                self.setOutput("API key is not valid!\nPlease enter a valid api key otherwise you will lose all\nfunctionality and probably experience errors!", "cli") 
        else:
            self.disableFunctionality()
            self.setOutput("API key is not valid!\nPlease enter a valid api key otherwise you will lose all\nfunctionality and probably experience errors!", "cli") 
        
    def clearInput(self) -> None:
        self.input_box.delete("1.0", tk.END)
    
    def clearOutput(self) -> None:
        self.output_box.delete("1.0", tk.END)
        self.clearInput()
    
    def getUserInput(self) -> str:
        user_query_input = self.input_box.get("1.0", tk.END).strip()
        user_cmd_input = self.command_entry.get()
        inputs = {
            "query": user_query_input, 
            "command": user_cmd_input      
        }
        return inputs

    def setOutput(self, output: str, type: str = "cli") -> None:
        if type == "chat":
            self.output_box.insert(tk.END, f"\n{self.nt.time(False)} [{self.cli.engine}]: {output}\n")
        elif type == "cli":
            self.output_box.insert(tk.END, f"\n{self.nt.time(False)} [wingtp]: {output}\n")
        elif type == "user":
            self.output_box.insert(tk.END, f"\n{self.nt.time(False)} [{self.USER}]: {output}\n")

    def getUsername(self) -> str:
        self._config.openConfig()
        self.USER = self._config.getOption("user", "username")
        self._config.saveConfig()

    def setUsername(self) -> None:
        dialog = ctk.CTkInputDialog(text="Enter a new username: ", title="Username Input")
        _username = str(dialog.get_input())
        self._config.openConfig()
        self._config.setOption("user", "username", f"{_username}")
        self._config.saveConfig()
        self.USER = _username
        self.sidebar_username_btn.configure(text=f"{self.USER}")

    def processQueryRequest(self, request: str) -> bool:
        self.cli.setRequest(request)
        self.cli.requestData()
        response = None
        if self.REQUEST_TYPE == self.cli.request_types["chat"]:
            response = self.cli.getResponse()
        elif self.REQUEST_TYPE == self.cli.request_types["images"]:
            response = self.cli.getImageURLResponse()
        elif self.REQUEST_TYPE == self.cli.request_types["audio"]:
            response = self.cli.getTranscriptResponse()
        elif self.REQUEST_TYPE == self.cli.request_types["embeddings"]:
            response = self.cli.getEmbeddingsResponse()
        elif self.REQUEST_TYPE == self.cli.request_types["files"]:
            response = self.cli.getFilesResponse()
        elif self.REQUEST_TYPE == self.cli.request_types["fine_tuning"]:
            print("Response type not implemented in gui.py/processQueryRequest()")
        elif self.REQUEST_TYPE == self.cli.request_types["moderations"]:
            response = self.cli.getModerationResponse()
        elif self.REQUEST_TYPE == self.cli.request_types["build_requests"]:
            print("Response type not implemented in gui.py/processQueryRequest()")
        elif self.REQUEST_TYPE == self.cli.request_types["translation"]:
            response = self.cli.getResponse()
        elif self.REQUEST_TYPE == self.cli.request_types["sentement"]:
            response = self.cli.getResponse()
        elif self.REQUEST_TYPE == self.cli.request_types["qa"]:
            response = self.cli.getResponse()
        elif self.REQUEST_TYPE == self.cli.request_types["summarization"]:
            response = self.cli.getResponse()
        elif self.REQUEST_TYPE == self.cli.request_types["code_gen"]:
            response = self.cli.getResponse()
        elif self.REQUEST_TYPE == self.cli.request_types["edits"]:
            response = self.cli.getEditResponse()
            
        self.clearInput()
        self.setOutput(request, "user")
        self.setOutput(response, "chat")
        if self.SAVE_CHAT:
            if self.cli.saveChat(self.CHAT_LOG_PATH, f"{request}\n{response}\n") != False:
                return True
            else:
                return False
    
    def processCommandRequest(self, request: str) -> None:
        if self.SAVE_CHAT:
            self.cli.saveChat(self.CHAT_LOG_PATH, f"{request}")
        if request == self.commands[0]:
            self.setOutput("Goodbye! ...", "cli")
            sys.exit()
        elif request == self.commands[1]:
            self.open_response_token_limit_input_dialog_event()
            self.clearInput()
        elif request == self.commands[2]:
            engine = simpledialog.askstring("Input", "Set the engine: ")
            self.cli.setEngine(engine)
            self.clearInput()
            self.setOutput(f"Engine set to {self.cli.getEngine()}", "cli")
        elif request == self.commands[3]:
            self.open_response_count_input_dialog_event()
            self.clearInput()
        elif request == self.commands[4]:
            self.open_api_base_input_dialog_event()
            self.clearInput()
        elif request == self.commands[5]:
            self.open_api_type_input_dialog_event()
            self.clearInput()
        elif request == self.commands[6]:
            self.open_api_version_input_dialog_event()
            self.clearInput()
        elif request == self.commands[7]:
            self.open_organization_input_dialog_event()
            self.clearInput()
        elif request == self.commands[8]:
            self.open_user_defined_datafile_input_dialog_event()
            self.clearInput()
        elif request == self.commands[9]:
            self.open_jsonl_datafile_input_dialog_event()
            self.clearInput()
        elif request == self.commands[10]:
            self.clearInput()
            self.setOutput(self.cli._help.__doc__, "cli")
        elif request == self.commands[11]:
            self.clearOutput()
        elif request.split(' ')[0] == self.commands[12]:
            self.change_appearance_mode_event(request.split(' ')[1])
        elif request == self.commands[13]:
            self.change_output_color_event()
        else:
            self.setOutput(self.cli._help.__doc__, "cli")
    
    def process_input(self) -> None:
        request_type = self.cli.request_type
        request = self.getUserInput()
        query_request = request["query"]
        command_request = request["command"]
        if len(query_request) == 0 and request_type == 2:
            self.processQueryRequest(" ")
            return True
        if request_type == 4:
            self.processQueryRequest(" ")
            return True
        if len(query_request) != 0:
            self.processQueryRequest(query_request)
            return True
        if len(command_request) != 0:
            self.processCommandRequest(command_request)
            return True
        if len(command_request) != 0 and len(query_request) != 0:
            self.processCommandRequest(command_request)
            self.processQueryRequest(query_request)
            return True
        if len(query_request) == 0 and len(command_request) == 0:
            self.setOutput(f" \
Try entering text into the chat window to receive a reponse.\n \
Or you can use one of the following commands by entering one\n \
into the command input under the chat window.\n \
{self.cli._help.__doc__}", "cli")
            return False    
