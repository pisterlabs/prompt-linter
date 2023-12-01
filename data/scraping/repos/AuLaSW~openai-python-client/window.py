"""
This module creates windows for the openai-client program.
"""
from __future__ import annotations
import tkinter as tk
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from openaiclient.view.frame.settings.completionsettings import CompletionSettings
from openaiclient.view.frame.settings.settingsframe import SettingsFrame
from openaiclient.view.frame.input.completioninput import CompletionInputFrame
from openaiclient.view.frame.menuframe import *
from openaiclient.view.frame.mainframe import MainFrame
from openaiclient.view.frame.apiframe import APIFrame

if TYPE_CHECKING:
    from openaiclient.controller.controller import Controller


class Window(ABC):
    """
    An abstract constructor class for different windows
    """

    def __init__(self, window: tk.Tk, controller: Controller):
        self._window = window
        self._controller = controller

    @abstractmethod
    def windowConstructor(self):
        """Constructs the frame product"""
        pass

    def draw(self):
        """Packs the product onto the window and starts the window"""
        frame = self.windowConstructor()
        frame.create()
        frame.grid()

        return frame

    @property
    def window(self):
        return self._window

    @property
    def controller(self):
        return self._controller


class CompletionSettingsWindow(Window):
    """
    A concrete factory class for the CompletionSettings window.
    """

    def windowConstructor(self) -> CompletionSettings:
        """Constructs a CompletionSettings frame product"""
        return CompletionSettings(self.window, self.controller)


class SettingsWindow(Window):
    """
    A concrete factory class for the CompletionSettings window.
    """

    def windowConstructor(self) -> SettingsFrame:
        """Constructs a CompletionSettings frame product"""
        return SettingsFrame(self.window, self.controller)


class CompletionInputWindow(Window):
    """
    A concrete factor class for the CompletionInput window.
    """

    def windowConstructor(self) -> CompletionInputFrame:
        menu = CompletionRequestMenu(self.window, self.controller)
        self.window.config(menu=menu.menubar)
        return CompletionInputFrame(self.window, self.controller)


class MainWindow(Window):
    """
    A Concrete factory class for creating the main window on startup.
    """

    def windowConstructor(self) -> MainFrame:
        menu = MainMenu(self.window, self.controller)
        self.window.config(menu=menu.menubar)
        return MainFrame(self.window, self.controller)


class APIWindow(Window):
    """
    A concrete facotry class for creating the API window when no API key is detected
    """

    def windowConstructor(self) -> APIFrame:
        return APIFrame(self.window, self.controller)


if __name__ == "__main__":
    from openaiclient.controller.controller import Controller
    from tests.unit.fixture import api

    controller = Controller(api)

    csw = MainWindow(controller.view._root, controller)

    csw.draw()
