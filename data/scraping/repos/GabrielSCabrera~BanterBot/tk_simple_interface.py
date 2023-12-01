import logging
import tkinter as tk
from tkinter import ttk
from typing import Optional, Union

from banterbot.data.enums import ToneMode
from banterbot.data.prompts import Greetings
from banterbot.extensions.interface import Interface
from banterbot.managers.azure_neural_voice_manager import AzureNeuralVoiceManager
from banterbot.managers.openai_model_manager import OpenAIModelManager
from banterbot.models.azure_neural_voice_profile import AzureNeuralVoiceProfile
from banterbot.models.openai_model import OpenAIModel


class TKSimpleInterface(tk.Tk, Interface):
    """
    TKSimpleInterface is a graphical user interface (GUI) for the BanterBot package that uses OpenAI models for
    generating responses, Azure Neural Voices for text-to-speech, and Azure speech-to-text recognition. The class
    inherits from both tkinter.Tk and Interface, providing a seamless integration of the chatbot functionality with a
    user-friendly interface.

    The GUI allows users to interact with the chatbot by entering their name and message, and it displays the
    conversation history in a scrollable text area. Users can send messages by pressing the "Send" button or the "Enter"
    key. The chatbot's responses are generated using the specified OpenAI model and can be played back using the
    specified Azure Neural Voice. Additionally, users can toggle speech-to-text input by pressing the "Listen" button.
    """

    def __init__(
        self,
        model: OpenAIModel = OpenAIModelManager.load("gpt-3.5-turbo"),
        voice: AzureNeuralVoiceProfile = AzureNeuralVoiceManager.load("Aria"),
        style: str = "chat",
        languages: Optional[Union[str, list[str]]] = None,
        system: Optional[str] = None,
        tone_mode: Optional[ToneMode] = None,
        tone_mode_model: OpenAIModel = None,
        phrase_list: Optional[list[str]] = None,
        assistant_name: Optional[str] = None,
    ) -> None:
        """
        Initialize the TKSimpleInterface class, which inherits from both tkinter.Tk and Interface.

        Args:
            model (OpenAIModel, optional): The OpenAI model to be used for generating responses.
            voice (AzureNeuralVoice, optional): The Azure Neural Voice to be used for text-to-speech.
            style (str, optional): The style of the conversation (e.g., "cheerful", "sad", "chat").
            languages (Optional[Union[str, list[str]]]): The languages supported by the speech-to-text recognizer.
            system (Optional[str]): An initialization prompt that can be used to set the scene.
            tone_mode (bool): Which tone evaluation mode to use.
            phrase_list(list[str], optional): Optionally provide the recognizer with context to improve recognition.
        """
        logging.debug(f"TKMultiplayerInterface initialized")

        tk.Tk.__init__(self)
        Interface.__init__(
            self,
            model=model,
            voice=voice,
            style=style,
            languages=languages,
            system=system,
            tone_mode=tone_mode,
            tone_mode_model=tone_mode_model,
            phrase_list=phrase_list,
            assistant_name=assistant_name,
        )

        # Bind the `_quit` method to program exit, in order to guarantee the stopping of all running threads.
        self.protocol("WM_DELETE_WINDOW", self._quit)

    def update_conversation_area(self, word: str) -> None:
        """
        Update the conversation area with the specified word.

        Args:
            word (str): The word to be added to the conversation area.
        """
        super().update_conversation_area(word)
        self.conversation_area["state"] = tk.NORMAL
        self.conversation_area.insert(tk.END, word)
        self.conversation_area.update_idletasks()
        self.conversation_area["state"] = tk.DISABLED
        self.conversation_area.see(tk.END)

    def select_all_on_focus(self, event) -> None:
        """
        Select all text in the widget when it receives focus.

        Args:
            event: The event object containing information about the focus event.
        """
        widget = event.widget
        if widget == self.name_entry:
            self._name_entry_focused = True
            widget.selection_range(0, tk.END)
            widget.icursor(tk.END)
        else:
            self._name_entry_focused = False

    def prompt(self) -> None:
        """
        Prompt the user for input and process the message.
        This method retrieves the user's name and message, and then calls the superclass's prompt method.
        """
        user_name = self.name_entry.get().split(" ")[0].strip()
        user_message = self.message_entry.get()
        if not user_message:
            return
        else:
            super().prompt(user_message, user_name)
            self.message_entry.delete(0, tk.END)

    def listener_toggle(self) -> None:
        """
        Toggle the speech-to-text functionality.
        """
        if self._listening_toggle:
            self.listen_btn["text"] = "Listen"
        else:
            self.listen_btn["text"] = "Stop"

        user_name = self.name_entry.get().split(" ")[0].strip()
        super().listener_toggle(user_name)

    def run(self, greet: bool = False) -> None:
        """
        Run the BanterBot application. This method starts the main event loop of the tkinter application.

        Args:
            greet (bool): If True, greets the user unprompted on initialization.
        """
        if greet:
            self.system_prompt(Greetings.UNPROMPTED_GREETING.value)
        self.mainloop()

    def _quit(self) -> None:
        """
        This method is called on exit, and interrupts any currently running activity.
        """
        self._openai_service.interrupt()
        self._speech_synthesis_service.interrupt()
        self._speech_recognition_service.interrupt()
        self.quit()
        self.destroy()

    def _init_gui(self) -> None:
        """
        Initialize the graphical user interface for the BanterBot application.
        This method sets up the main window, conversation area, input fields, and buttons.
        """
        self.title(f"BanterBot {self._model.model}")
        self.configure(bg="black")
        self.geometry("1024x768")
        self._font = ("Cascadia Code", 16)

        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure(".", font=self._font, bg="black", fg="white")
        self.style.configure("Vertical.TScrollbar", background="black", bordercolor="black", arrowcolor="black")

        self.history_frame = ttk.Frame(self)
        self.conversation_area = tk.Text(
            self.history_frame, wrap=tk.WORD, state=tk.DISABLED, bg="black", fg="white", font=self._font
        )
        self.conversation_area.grid(row=0, column=0, ipadx=5, padx=5, pady=5, sticky="nsew")
        self.history_frame.rowconfigure(0, weight=1)
        self.history_frame.columnconfigure(0, weight=1)

        self.scrollbar = ttk.Scrollbar(self.history_frame, command=self.conversation_area.yview)
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.conversation_area["yscrollcommand"] = self.scrollbar.set

        self.history_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.entry_frame = ttk.Frame(self)
        self.entry_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        self.name_entry = tk.Entry(
            self.entry_frame, bg="black", fg="white", insertbackground="white", font=self._font, width=12
        )
        self.name_entry.grid(row=0, column=0, padx=(5, 0), pady=5, sticky="nsew")
        self.name_entry.insert(0, "User")
        self.name_entry.bind("<FocusIn>", self.select_all_on_focus)
        self._name_entry_focused = False

        self.message_entry = tk.Entry(
            self.entry_frame, bg="black", fg="white", insertbackground="white", font=self._font
        )
        self.message_entry.grid(row=0, column=1, padx=(5, 5), pady=5, sticky="nsew")
        self.message_entry.focus_set()
        self.entry_frame.columnconfigure(1, weight=1)

        self.send_btn = ttk.Button(self.entry_frame, text="Send", command=self.prompt, width=7)
        self.send_btn.grid(row=0, column=2, padx=(0, 5), pady=5, sticky="nsew")

        self.listen_btn = ttk.Button(self.entry_frame, text="Listen", command=self.listener_toggle, width=7)
        self.listen_btn.grid(row=0, column=3, padx=(0, 5), pady=5, sticky="nsew")

        self.message_entry.bind("<Return>", lambda event: self.prompt())
