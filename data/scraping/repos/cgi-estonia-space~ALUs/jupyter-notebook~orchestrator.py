import subprocess
from typing import Union

import folium
import ipywidgets
import ipywidgets as widgets
import configuration as config
import shutil
from subprocess import run

import helper_functions as helper
import IPython.display as ipydisplay

from algorithm.gabor_extraction import GaborExtraction
from algorithm.calibration import Calibration
from algorithm.coherence import Coherence
from callback import *


def check_installed_packages() -> None:
    """Runs eio selfcheck to control that all the necessary packages are
    installed.

    Returns:
        None
    """
    process: subprocess.CompletedProcess = subprocess.run(['eio', 'selfcheck'],
                                                          check=True)
    assert process.returncode == 0, 'Not all required packages are installed'


def create_alus_path_prompt() -> None:
    """Checks whether ALUs executable are on the PATH, and creates a widget
    for selecting folder, containing them.

    Returns:
        None.
    """
    info_output: widgets.Output = widgets.Output()

    def choose_alus_dir(button, output_widget):
        helper.select_directory(button, config.parameters,
                                config.ParameterNames.ALUS_DIRECTORY)
        if config.parameters[config.ParameterNames.ALUS_DIRECTORY] != '':
            output_widget.clear_output()
            with output_widget:
                helper.check_if_correct_alus_folder_selected(
                    config.parameters[config.ParameterNames.ALUS_DIRECTORY])

    with info_output:
        alus_exists: bool = helper.check_if_alus_exists()

    if alus_exists:
        print('ALUs was found on PATH.')
        print("You can choose another directory if You wish.")

    alus_chooser_callback: Callback = Callback(choose_alus_dir,
                                               output_widget=info_output)

    alus_chooser: ipywidgets.Button = helper.create_button(
        'Choose ALUs directory', alus_chooser_callback)

    ipydisplay.display(info_output)
    ipydisplay.display(alus_chooser)


def create_algorithm_prompt() -> None:
    """Creates a dropdown prompt for choosing the desired algorithm to run.

    Returns:
        None.
    """

    @widgets.interact
    def choose_algorithm(algorithm=config.supported_algorithms.keys()):
        config.parameters[config.ParameterNames.SELECTED_ALGORITHM] = \
            config.supported_algorithms[algorithm]


def show_algorithm_parameters() -> None:
    """Displays algorithm parameters.

    Returns:
        None
    """
    selected_algorithm = config.parameters[
        config.ParameterNames.SELECTED_ALGORITHM]

    if selected_algorithm is config.AlgorithmName.CALIBRATION_ROUTINE:
        calibration = Calibration()
        config.parameters[config.ParameterNames.ALGORITHM_CLASS] = calibration
        calibration.display_options()
    elif selected_algorithm is config.AlgorithmName.COHERENCE_ROUTINE:
        coherence = Coherence()
        config.parameters[config.ParameterNames.ALGORITHM_CLASS] = coherence
        coherence.display_options()
    elif selected_algorithm is config.AlgorithmName.GABOR_FEATURE:
        gabor_extraction = GaborExtraction()
        config.parameters[
            config.ParameterNames.ALGORITHM_CLASS] = gabor_extraction
        gabor_extraction.display_options()
    else:
        helper.print_error(
            f'Unsupported Algorithm: {selected_algorithm.value}')


def check_necessary_input() -> None:
    """Checks whether all the algorithm necessary parameters were filled and
    displays error messages if they were not.

    Returns:
        None
    """
    algorithm = helper.get_algorithm_class()
    if algorithm is None:
        helper.print_error('No algorithm selected.')
        return
    if not algorithm.check_necessary_input():
        config.parameters[
            config.ParameterNames.ALGORITHM_CLASS].display_options()


def show_first_input_map() -> Union[folium.Map, None]:
    """Visualises the map of the first input dataset.

    Returns:
        Folium map which will be displayed by the Jupyter Notebook.
    """
    helper.get_sentinel_1_files()
    visualised_map: Union[folium.Map, None] = None
    if helper.get_algorithm_name() != config.AlgorithmName.GABOR_FEATURE:
        try:
            visualised_map = helper.get_algorithm_class().input_files[
                0].visualize_webmap()
        except IndexError:
            return visualised_map

    return visualised_map


def show_coherence_secondary_map() -> Union[folium.Map, None]:
    """Visualises the map of the secondary dataset if the selected algorithm
    is a Coherence routine.

    Returns:
        Folium map which will be displayed by the Jupyter Notebook.
    """
    visualised_map: Union[folium.Map, None] = None
    if helper.get_algorithm_name() == config.AlgorithmName.COHERENCE_ROUTINE:
        try:
            visualised_map = helper.get_algorithm_class().input_files[
                1].visualize_webmap()
        except IndexError:
            helper.print_error(
                'No algorithm selected or secondary input not provided.')
            return visualised_map

    return visualised_map


def launch_algorithm() -> int:
    """Launches the algorithm.

    Returns:
        str: Return code of the algorithm execution.
    """
    return helper.get_algorithm_class().launch_algorithm()
