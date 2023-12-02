import logging
import threading
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


class TKMultiplayerInterface(tk.Tk, Interface):
    """
    A graphical user interface (GUI) class that enables interaction with the BanterBot chatbot in a multiplayer mode.
    It supports functionalities such as text input, text-to-speech and speech-to-text capabilities for up to 9 users
    simultaneously, based on OpenAI and Azure services.

    This class inherits from tkinter's Tk class and a custom Interface class, allowing it to be displayed as a
    standalone window and follow a specific chatbot interaction protocol respectively.
    """

    def __init__(
        self,
        model: OpenAIModel = OpenAIModelManager.load("gpt-3.5-turbo"),
        voice: AzureNeuralVoiceProfile = AzureNeuralVoiceManager.load("Aria"),
        style: str = "chat",
        languages: Optional[Union[str, list[str]]] = None,
        tone_mode: Optional[ToneMode] = None,
        tone_mode_model: OpenAIModel = None,
        system: Optional[str] = None,
        phrase_list: Optional[list[str]] = None,
        assistant_name: Optional[str] = None,
    ) -> None:
        """
        Initialize the TKMultiplayerInterface class, which inherits from both tkinter.Tk and Interface.

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

    def listener_activate(self, idx: int) -> None:
        user_name = self.name_entries[idx].get().split(" ")[0].strip()
        super().listener_activate(user_name)

    def request_response(self) -> None:
        if self._messages:
            # Interrupt any currently active ChatCompletion, text-to-speech, or speech-to-text streams
            self.interrupt()
            self._thread_queue.add_task(threading.Thread(target=self.respond, daemon=True))

    def run(self, greet: bool = False) -> None:
        """
        Run the BanterBot application. This method starts the main event loop of the tkinter application.

        Args:
            greet (bool): If True, greets the user unprompted on initialization.
        """
        if greet:
            self.system_prompt(Greetings.UNPROMPTED_GREETING.value)
        self.mainloop()

    def select_all_on_focus(self, event) -> None:
        widget = event.widget
        if widget == self.name_entry:
            self._name_entry_focused = True
            widget.selection_range(0, tk.END)
            widget.icursor(tk.END)
        else:
            self._name_entry_focused = False

    def update_conversation_area(self, word: str) -> None:
        super().update_conversation_area(word)
        self.conversation_area["state"] = tk.NORMAL
        self.conversation_area.insert(tk.END, word)
        self.conversation_area.update_idletasks()
        self.conversation_area["state"] = tk.DISABLED
        self.conversation_area.see(tk.END)

    def _quit(self) -> None:
        """
        This method is called on exit, and interrupts any currently running activity.
        """
        self.interrupt()
        self.quit()
        self.destroy()

    def _init_gui(self) -> None:
        self.title(f"BanterBot {self._model.model}")
        self.configure(bg="black")
        self.geometry("1024x565")
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

        self.panel_frame = ttk.Frame(self)
        self.panel_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.name_entries = []
        self.listen_buttons = []

        for i in range(9):
            name_entry = tk.Entry(
                self.panel_frame, bg="black", fg="white", insertbackground="white", font=self._font, width=12
            )
            name_entry.grid(row=i, column=0, padx=(5, 0), pady=5, sticky="nsew")
            name_entry.insert(0, f"User {i+1}")
            self.name_entries.append(name_entry)

            listen_button = ttk.Button(self.panel_frame, text="Listen", width=7)
            listen_button.grid(row=i, column=1, padx=(0, 5), pady=5, sticky="nsew")
            listen_button.bind(f"<ButtonPress-1>", lambda event, i=i: self.listener_activate(i))
            listen_button.bind(f"<ButtonRelease-1>", lambda event: self.listener_deactivate())
            self.listen_buttons.append(listen_button)

            self.bind(f"<KeyPress-{i+1}>", lambda event, i=i: self.listener_activate(i))
            self.bind(f"<KeyRelease-{i+1}>", lambda event: self.listener_deactivate())

        self.request_btn = ttk.Button(self.panel_frame, text="Respond", width=7)
        self.request_btn.grid(row=9, column=0, padx=(5, 0), pady=5, sticky="nsew")

        self.request_btn.bind(f"<ButtonRelease-1>", lambda event: self.request_response())
        self.bind("<Return>", lambda event: self.request_response())
