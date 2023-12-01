import sys
from gui import gui_utils, omniverse_main
import keyring

from src.data.session_manager import SessionManager
from src.audio.audio_manager import AudioManager
from src.art.art_manager import ArtManager
from src.llms.llm_manager import LLMManager
from src.personas.persona_manager import PersonaManager
from src.worlds.world_manager import WorldManager
from langchain.callbacks.base import BaseCallbackManager
from gui.callbacks.streaming_browser_out import StreamingBrowserCallbackHandler

from gui.modes.world_mode.world_mode import WorldMode
from gui.modes.canvas_mode.canvas_mode import CanvasMode, CanvasView
from gui.modes.blueprint_mode.blueprint_mode import BlueprintMode

from gui.components.chat_interface import ChatInterface

from gui.components.login_window import LoginWindow
from gui.components.user_creation_window import UserCreationWindow
from gui.components.workspace_creation_window import WorkspaceCreationWindow
from gui.components.developer_window import DeveloperWindow

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QToolBar, QHBoxLayout, QButtonGroup, QMenu, QMenuBar
from PyQt6.QtGui import QPixmap, QIcon, QAction
from PyQt6.QtCore import QTimer, Qt


from src import constants
from src.data import data_utils
from src.logger_utils import create_logger

class Omniverse(QMainWindow, omniverse_main.Ui_MainWindow):
    """The Omniverse class that handles user interaction in the main window."""
    def __init__(self, parent=None):
        super(Omniverse, self).__init__(parent)
        data_utils.ensure_dir_exists('local')
        data_utils.ensure_dir_exists('logs')
        self.logger = create_logger(__name__, constants.SYSTEM_LOG_FILE)
        self.logger.info("Starting Omniverse...")
        self.setupUi(self)

        self.menu_bar = None
        self.chat_interface = None # Used across all modes in the right splitter panel
        self.login_window = None # For all logins
        self.user_creation_window = None # For individual user creation
        self.workspace_creation_window = None # For initial workspace creation
        self.workspace_settings_window = None # For adjusting workspace settings
        self.developer_window = None # Just widgets for testing
        self.user_profile_window = None # For adjusting the public profile for the current user or user as selected by admin
        self.user_settings_window = None # For adjusting the settings for the current user or user as selected by admin
        self.key_vault_window = None # For adjusting the key vault for the current user or user as selected by admin
        self.users_management_window = None # Used by admin to manage users

        self.session = SessionManager()
        self.llm_manager = None
        
        # Ensure that the workspace is setup
        if keyring.get_password(constants.TITLE, constants.WORKSPACE) is None:
            self.status_bar.showMessage("Omniverse workspace not in keyring. New workspace creation required.")
            self.logger.info("Omniverse workspace not in keyring. New workspace creation required.")
            self.workspace_creation_window = WorkspaceCreationWindow()
            self.workspace_creation_window.workspace_created_signal.connect(self.workspace_created)
            self.workspace_creation_window.closed_signal.connect(self.workspace_setup_window_closed)
            self.workspace_creation_window.show()

        # Let's implement this in a workspace settings window later        
        # keyring.delete_password(constants.TITLE, constants.WORKSPACE) # Uncomment to reset workspace encryption key
        
        # Establish the modes available to the user
        self.mode_index = 0 
        self.modes = [WorldMode(parent=self), CanvasMode(parent=self), BlueprintMode(parent=self)]
        
        # Setup windows, modes, and callbacks
        self.initialize_gui()

        # Login if there are users in the workspace and the workspace is in the keyring, otherwise create a new user
        if self.session.get_user_count() > 0 and keyring.get_password(constants.TITLE, constants.WORKSPACE) is not None:
            self.login_start()
        elif keyring.get_password(constants.TITLE, constants.WORKSPACE) is not None:
            self.create_new_user()

        # Initialize a QTimer for continuous constant advancement
        self.frame_timer = QTimer(self)
        self.frame_timer.timeout.connect(self.advance)
        self.frame_timer.start(int(1000/constants.FRAMES_PER_SECOND))

        # Still in design phase
        self.persona_manager = PersonaManager(session=self.session)
        self.world_manager = WorldManager(session=self.session, personas=self.persona_manager.personas)

        self.chat_interface.populate_personas(self.persona_manager.get_persona_names())
        self.chat_interface.populate_models(["gpt-3.5-turbo", "text-davinci-003"]) # Just until LLM Manager is replaced
        self.chat_interface.populate_protocols(["Assistant", "Storyteller", "Tutor"]) # Just until LLM Manager is replaced

        # Setup the AI art manager
        self.art_manager = ArtManager(session=self.session)
        self.current_generated_image = None

        # Setup the audio manager
        self.audio = AudioManager()
        # self.audio.play_sound_effect("click") # For testing

        # Setup the language model manager here for now
        # TODO: Dynamically generate this from json data for each persona: currently at design phase in the new personas package
        # We'll just stream tokens to the browsers in the developer window until we have display containers setup in Blueprint Mode
        self.browser_callbacks = {
            "response": BaseCallbackManager([StreamingBrowserCallbackHandler(self.chat_interface.chat_output_browser)]),
            "sentiment": BaseCallbackManager([StreamingBrowserCallbackHandler(self.developer_window.sentiment_browser)]),
            "entity": BaseCallbackManager([StreamingBrowserCallbackHandler(self.developer_window.entity_browser)]),
            "knowledge": BaseCallbackManager([StreamingBrowserCallbackHandler(self.developer_window.knowledge_browser)]),
            "summary": BaseCallbackManager([StreamingBrowserCallbackHandler(self.developer_window.summary_browser)])
        }

        self.llm_manager = LLMManager(session=self.session, 
                                      browser_callbacks=self.browser_callbacks, 
                                      assistant_id=self.persona_manager.current_responder.name)
        self.current_user_prompt = ""
        self.current_generated_response = ""        

    def advance(self):
        """ Advance the application by one frame."""
        delta_time = self.frame_timer.interval() / 1000.0
        for mode in self.modes:
            mode.advance(delta_time)

    def generate_response(self):
        """ Generate a response from the current user prompt."""
        self.status_bar.showMessage("Grounding Response...")
        self.logger.info("Grounding Response...")
        self.current_user_prompt = self.chat_interface.get_user_input()
        self.chat_interface.send_user_input_to_chat(self.session.current_user.name, self.persona_manager.current_responder.name)
        self.developer_window.clear()
        QApplication.processEvents()
        self.llm_manager.preprocessing(self.current_user_prompt)
        self.status_bar.showMessage("Generating Response...")
        self.logger.info("Generating Response...")
        self.current_generated_response = self.llm_manager.generate_response()
        self.llm_manager.report_tokens()
        self.status_bar.showMessage("Processing response...")
        self.logger.info("Processing response...")
        if self.audio.text_to_speech:
            self.audio.play_text_to_speech(self.current_generated_response)
        QApplication.processEvents()
        self.llm_manager.postprocessing(self.current_user_prompt, self.current_generated_response)
        self.status_bar.showMessage("Response Complete")
        self.logger.info("Response Complete")
  
    def generate_image(self):
        """ Generate an image from the current user prompt."""
        self.status_bar.showMessage("Generating Image...")
        self.logger.info("Generating Image...")
        prompt = self.chat_interface.get_user_input()
        QApplication.processEvents()
        pixmap = QPixmap()
        pixmap.loadFromData(self.art_manager.generate_image(prompt))
        if not pixmap.isNull():
            self.current_generated_image = pixmap
            for mode in self.modes:
                if isinstance(mode.display, CanvasView):
                    mode.display.subject_pixmap = self.current_generated_image
            self.status_bar.showMessage("Image Generation Complete")
            self.logger.info("Image Generation Complete")
        else:
            self.status_bar.showMessage("Image Generation Failed")
            self.logger.info("Image Generation Failed")

    def toggle_text_to_speech(self):
        """ Toggle text to speech on and off."""
        self.status_bar.showMessage("Text to speech is now " + ("on." if not self.audio.text_to_speech else "off."))
        self.audio.text_to_speech = not self.audio.text_to_speech
        if self.audio.text_to_speech:
            self.chat_interface.tts_button.setIcon(QIcon(self.chat_interface.tts_on_button_icon_pixmap))
        else:
            self.chat_interface.tts_button.setIcon(QIcon(self.chat_interface.tts_off_button_icon_pixmap))

    def toggle_speech_to_text(self):
        """ Toggle speech to text on and off."""
        self.status_bar.showMessage("Speech to text is now " + ("on." if not self.audio.speech_to_text else "off."))
        self.audio.speech_to_text = not self.audio.speech_to_text
        if self.audio.speech_to_text:
            self.chat_interface.stt_button.setIcon(QIcon(self.chat_interface.stt_on_button_icon_pixmap))
        else:
            self.chat_interface.stt_button.setIcon(QIcon(self.chat_interface.stt_off_button_icon_pixmap))
    
    def set_mode(self):
        """ Set the current mode."""        
        self.mode_index = self.mode_selector_button_group.checkedId()
        self.displays_stacked_widget.setCurrentIndex(self.mode_index)
        self.controls_stacked_widget.setCurrentIndex(self.mode_index)
        for mode_index, mode in enumerate(self.modes):
            for tool_bar in mode.get_tool_bars():
                tool_bar.setVisible(mode_index == self.mode_index)
        self.status_bar.showMessage("Now in " + self.modes[self.mode_index].name)

    def setup_window(self):
        """ Setup the main window."""
        self.setWindowTitle(f"{constants.TITLE} {constants.VERSION}")
        self.setWindowIcon(QIcon(gui_utils.load_icon("application-icon.ico")))
        self.main_splitter.setSizes([int(self.width() * 0.20), int(self.width() * 0.50), int(self.width() * 0.30)])
        self.status_bar.showMessage("Welcome to the Omniverse!")

    def create_mode_selector_tool_bar(self):
        """ Create the mode selector tool bar."""
        self.mode_selector_tool_bar = QToolBar(self)
        self.mode_selector_tool_bar.setMovable(False)
        self.mode_selector_tool_bar.setFloatable(False)
        self.mode_selector_tool_bar.setOrientation(Qt.Orientation.Horizontal)
        self.mode_selector_tool_bar.setHidden(False)
        self.mode_selector_tool_bar.setContentsMargins(2,2,2,2)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.mode_selector_tool_bar)
        self.mode_selector_tool_bar.setFixedHeight(40)

    def add_modes(self):
        """ Add the modes to the application."""
        self.mode_selection_widget = QWidget()
        self.mode_selection_layout = QHBoxLayout()
        self.mode_selection_widget.setLayout(self.mode_selection_layout)
        self.mode_selection_layout.setContentsMargins(2,2,2,2)
        self.mode_selector_button_group = QButtonGroup()
        self.mode_selector_button_group.setExclusive(True)
        self.mode_selector_button_group.buttonClicked.connect(self.set_mode) 
        self.mode_selector_tool_bar.addWidget(self.mode_selection_widget)

        for mode_index, mode in enumerate(self.modes):
            for tool_bar in mode.get_tool_bars():
                self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tool_bar)

            mode_button = mode.button
            mode_button.setCheckable(True)
            mode_button.setAutoExclusive(True)
            self.mode_selector_button_group.addButton(mode_button, mode_index)
            self.mode_selection_layout.addWidget(mode_button)
            self.mode_selector_tool_bar.addWidget(mode_button)
            if mode_index == self.mode_index:
                mode_button.setChecked(True)  

            self.displays_stacked_widget.addWidget(mode.display)
            self.controls_stacked_widget.addWidget(mode.control_widget)

    def login_successful(self):
        """ Perform actions after a successful login."""
        self.status_bar.showMessage(f"Logged in as {self.session.current_user.name}")
        self.logger.info(f"Logged in as {self.session.current_user.name}.")
        self.session.setup()
        self.llm_manager.setup()
        self.persona_manager.setup()
        if not self.isActiveWindow():
            self.activateWindow()
        self.raise_()
    
    def login_start(self):
        """ Start the login process."""
        self.status_bar.showMessage("Login to start session")
        self.login_window = LoginWindow(session=self.session)
        self.login_window.closed_signal.connect(self.login_window_closed)
        self.login_window.login_succeeded_signal.connect(self.login_successful)
        self.login_window.create_new_user_clicked_signal.connect(self.create_new_user)
        self.login_window.show()

    def new_user_created(self):
        """ Perform actions after a new user is created."""
        self.status_bar.showMessage(f"New user created: {self.session.get_most_recent_user().name}")
        self.logger.info(f"New user created: {self.session.get_most_recent_user().name}")
        self.user_creation_window.hide()
        self.login_start()
        if not self.isActiveWindow():
            self.activateWindow()
        self.raise_()
        
    def create_new_user(self):
        self.status_bar.showMessage("Creating a new user")
        """ Create a new user."""
        if self.login_window is not None: self.login_window.hide()
        # User Creation Window
        self.user_creation_window = UserCreationWindow(session=self.session)
        self.user_creation_window.user_created_signal.connect(self.new_user_created)
        self.user_creation_window.window_closed_signal.connect(self.user_creation_window_closed)
        self.user_creation_window.show()

    def on_exit(self):
        """ Close the application. """
        self.status_bar.showMessage("Exiting the Omniverse...")
        self.logger.info("Exiting the Omniverse...")
        self.session.close()
        self.logger.info("Goodbye!")
        self.close()
    
    def workspace_setup_window_closed(self):
        """ Perform actions after the workspace setup is closed."""
        if keyring.get_password(constants.TITLE, constants.WORKSPACE) is None:
            self.close()
    
    def login_window_closed(self):
        """ Perform actions after the login is closed."""
        if self.session.current_user is None:
            self.close()

    def user_creation_window_closed(self):
        """ Perform actions after the user creation is closed."""
        if self.session.get_user_count() == 0:
            self.close()
        else:
            self.login_start()

    def workspace_created(self):
        """ Perform actions after a new workspace is created."""
        self.status_bar.showMessage("New workspace created")
        key = self.workspace_creation_window.get_key()
        self.login_window = LoginWindow(session=self.session) # Reset the login window
        self.workspace_creation_window.close()
        if key is not None and key != "":
            keyring.set_password(constants.TITLE, constants.WORKSPACE, key)
            if self.session.get_user_count() == 0:
                self.create_new_user()
            else:
                self.login_start()

    def toggle_developer_window(self):
        """ Toggle the developer window."""
        if self.developer_window.isVisible():
            self.developer_window.hide()
        else:
            self.developer_window.show()

    def persona_changed(self, persona):
        self.logger.info(f"Persona changed to {persona}")
        self.persona_manager.set_current_responder(persona)
        if self.llm_manager is not None:
            self.llm_manager.assistant_id = persona

    def model_changed(self, model):
        print("Model changed to", model)
        if self.llm_manager is not None:
            self.llm_manager.set_model(model)

    def protocol_changed(self, protocol):
        self.logger.info(f"Protocol changed to {protocol}")
        if self.llm_manager is not None:
            self.llm_manager.set_protocol(protocol)
    
    def temperature_changed(self, temperature):
        self.logger.info(f"Temperature changed to {temperature/10.0}")
        if self.llm_manager is not None:
            self.llm_manager.set_temperature(temperature/10.0)

    def setup_menu_bar(self):
        """ Setup the menu bar."""
        self.menu_bar = QMenuBar()
        self.setMenuBar(self.menu_bar)

        # File Menu
        self.file_menu = self.menu_bar.addMenu("File")
        self.new_file_action = QAction("New", self)
        self.new_file_action.setIcon(QIcon(gui_utils.load_icon("button-new.ico")))
        self.file_menu.addAction(self.new_file_action)
        self.open_action = QAction("Open", self)
        self.open_action.setIcon(QIcon(gui_utils.load_icon("button-open.ico")))
        self.file_menu.addAction(self.open_action)
        self.save_action = QAction("Save", self)
        self.save_action.setIcon(QIcon(gui_utils.load_icon("button-save.ico")))
        self.file_menu.addAction(self.save_action)
        self.save_as_action = QAction("Save As...", self)
        self.file_menu.addAction(self.save_as_action)
        self.file_menu.addSeparator()
        self.import_action = QAction("Import", self)
        self.import_action.setIcon(QIcon(gui_utils.load_icon("file-import.ico")))
        self.file_menu.addAction(self.import_action)
        self.export_action = QAction("Export", self)
        self.export_action.setIcon(QIcon(gui_utils.load_icon("file-export.ico")))
        self.file_menu.addAction(self.export_action)
        self.file_menu.addSeparator()
        self.exit_action = QAction("Exit", self)
        self.exit_action.triggered.connect(self.on_exit)
        self.file_menu.addAction(self.exit_action)

        #Persona Menu
        self.persona_menu = self.menu_bar.addMenu("Persona")
        self.new_persona_action = QAction("New Persona", self)
        self.new_persona_action.setIcon(QIcon(gui_utils.load_icon("persona-icon.ico")))
        self.persona_menu.addAction(self.new_persona_action)

        # Let's combine User and Admin menus into one called Session
        # Session Menu
        self.session_menu = self.menu_bar.addMenu("Session")
        self.user_profile_action = QAction("User Profile", self)
        # self.user_profile_action.setIcon(QIcon(gui_utils.load_icon("button-user-profile.ico")))
        self.session_menu.addAction(self.user_profile_action)
        self.user_settings_action = QAction("User Settings", self)
        # self.user_settings_action.setIcon(QIcon(gui_utils.load_icon("button-user-settings.ico")))
        self.session_menu.addAction(self.user_settings_action)
        self.key_vault_action = QAction("Key Vault", self)
        #self.key_vault_action.setIcon(QIcon(gui_utils.load_icon("button-key-vault.ico")))
        self.session_menu.addAction(self.key_vault_action)

        self.session_menu.addSeparator()
        self.logout_action = QAction("Logout", self)
        # self.logout_action.setIcon(QIcon(gui_utils.load_icon("button-logout.ico")))
        self.session_menu.addAction(self.logout_action)
        self.session_menu.addSeparator()
        self.workspace_settings_action = QAction("Workspace", self)
        # self.workspace_settings_action.setIcon(QIcon(gui_utils.load_icon("button-settings.ico")))
        self.session_menu.addAction(self.workspace_settings_action)
        self.users_panel_action = QAction("Users", self)
        # self.users_panel_action.setIcon(QIcon(gui_utils.load_icon("button-users.ico")))
        self.session_menu.addAction(self.users_panel_action)
        self.session_menu.addSeparator()
        self.developer_window_action = QAction("Developer", self)
        # self.developer_window_action.setIcon(QIcon(gui_utils.load_icon("button-developer.ico")))
        self.developer_window_action.triggered.connect(self.toggle_developer_window)
        self.session_menu.addAction(self.developer_window_action)

        
        # Help Menu
        self.help_menu = self.menu_bar.addMenu("Help")
        self.about_action = QAction("About", self)
        self.help_menu.addAction(self.about_action)

    def setup_chat_interface(self):
        self.chat_interface_widget = QWidget()
        self.chat_interface = ChatInterface(self.chat_interface_widget)
        self.chat_stacked_widget.addWidget(self.chat_interface_widget)    
        self.chat_interface.tts_button.clicked.connect(self.toggle_text_to_speech)
        self.chat_interface.stt_button.clicked.connect(self.toggle_speech_to_text)
        self.chat_interface.generate_text_button.clicked.connect(self.generate_response)
        self.chat_interface.generate_image_button.clicked.connect(self.generate_image)
        self.chat_interface.persona_combo_box.currentTextChanged.connect(self.persona_changed)
        self.chat_interface.model_combo_box.currentTextChanged.connect(self.model_changed)
        self.chat_interface.protocol_combo_box.currentTextChanged.connect(self.protocol_changed)
        self.chat_interface.temperature_slider.valueChanged.connect(self.temperature_changed)


    def initialize_gui(self):
        """ Initialize the GUI. """
        self.setup_window()
        self.setup_menu_bar()
        self.create_mode_selector_tool_bar()
        self.add_modes()
        self.set_mode()
        self.setup_chat_interface()
        self.show()

        # Developer window
        self.developer_window = DeveloperWindow()
       

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Omniverse()
    app.aboutToQuit.connect(window.on_exit)
    sys.exit(app.exec())


    