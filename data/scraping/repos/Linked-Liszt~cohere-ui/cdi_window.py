# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
This is GUI that allows user to configure and run experiment.
"""

__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c), UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['select_file',
           'select_dir',
           'msg_window',
           'main']

import sys
import os
import shutil
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import importlib
import convertconfig as conv
import ast
import cohere_core as cohere
import util.util as ut


def select_file(start_dir):
    """
    Shows dialog interface allowing user to select file from file system.
    Parameters
    ----------
    start_dir : str
        directory where to start selecting the file
    Returns
    -------
    str
        name of selected file or None
    """
    start_dir = start_dir.replace(os.sep, '/')
    dialog = QFileDialog(None, 'select dir', start_dir)
    dialog.setFileMode(QFileDialog.ExistingFile)
    dialog.setSidebarUrls([QUrl.fromLocalFile(start_dir)])
    if dialog.exec_() == QDialog.Accepted:
        return str(dialog.selectedFiles()[0]).replace(os.sep, '/')
    else:
        return None


def select_dir(start_dir):
    """
    Shows dialog interface allowing user to select directory from file system.
    Parameters
    ----------
    start_dir : str
        directory where to start selecting
    Returns
    -------
    str
        name of selected directory or None
    """
    start_dir = start_dir.replace(os.sep, '/')
    dialog = QFileDialog(None, 'select dir', start_dir)
    dialog.setFileMode(QFileDialog.DirectoryOnly)
    dialog.setSidebarUrls([QUrl.fromLocalFile(start_dir)])
    if dialog.exec_() == QDialog.Accepted:
        return str(dialog.selectedFiles()[0]).replace(os.sep, '/')
    else:
        return None


def msg_window(text):
    """
    Shows message with requested information (text)).
    Parameters
    ----------
    text : str
        string that will show on the screen
    Returns
    -------
    noting
    """
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Information)
    msg.setText(text)
    msg.setWindowTitle("Info")
    msg.exec_()


class cdi_gui(QWidget):
    def __init__(self, parent=None):
        """
        Constructor, initializes GUI.
        Parameters
        ----------
        none
        Returns
        -------
        noting
        """
        super(cdi_gui, self).__init__(parent)

        self.beamline = None
        self.exp_id = None
        self.experiment_dir = None
        self.working_dir = None
        self.specfile = None

        uplayout = QHBoxLayout()
        luplayout = QFormLayout()
        ruplayout = QFormLayout()
        uplayout.addLayout(luplayout)
        uplayout.addLayout(ruplayout)

        self.set_work_dir_button = QPushButton()
        luplayout.addRow("Working Directory", self.set_work_dir_button)
        self.Id_widget = QLineEdit()
        luplayout.addRow("Experiment ID", self.Id_widget)
        self.scan_widget = QLineEdit()
        luplayout.addRow("scan(s)", self.scan_widget)
        self.beamline_widget = QLineEdit()
        ruplayout.addRow("beamline", self.beamline_widget)
        self.spec_file_button = QPushButton()
        ruplayout.addRow("spec file", self.spec_file_button)

        self.vbox = QVBoxLayout()
        self.vbox.addLayout(uplayout)

        self.t = None
        # self.vbox.addWidget(self.t)

        downlayout = QHBoxLayout()
        downlayout.setAlignment(Qt.AlignCenter)
        self.set_exp_button = QPushButton("load experiment")
        self.set_exp_button.setStyleSheet("background-color:rgb(205,178,102)")
        downlayout.addWidget(self.set_exp_button)
        self.create_exp_button = QPushButton('set experiment')
        self.create_exp_button.setStyleSheet("background-color:rgb(120,180,220)")
        downlayout.addWidget(self.create_exp_button)
        self.run_button = QPushButton('run everything', self)
        self.run_button.setStyleSheet("background-color:rgb(175,208,156)")
        downlayout.addWidget(self.run_button)
        self.vbox.addLayout(downlayout)

        spacer = QSpacerItem(0, 5)
        self.vbox.addItem(spacer)

        self.setLayout(self.vbox)
        self.setWindowTitle("CDI Reconstruction")

        self.set_exp_button.clicked.connect(self.load_experiment)
        self.set_work_dir_button.clicked.connect(self.set_working_dir)
        self.spec_file_button.clicked.connect(self.set_spec_file)
        self.run_button.clicked.connect(self.run_everything)
        self.create_exp_button.clicked.connect(self.set_experiment)


    def set_args(self, args):
        self.args = args


    def set_spec_file(self):
        """
        Calls selection dialog. The selected spec file is parsed.
        The specfile is saved in config.
        Parameters
        ----------
        none
        Returns
        -------
        noting
        """
        self.specfile = select_file(os.getcwd())
        if self.specfile is not None:
            self.spec_file_button.setStyleSheet("Text-align:left")
            self.spec_file_button.setText(self.specfile)
        else:
            self.specfile = None
            self.spec_file_button.setText('')
        if self.is_exp_exists() or self.is_exp_set():
            # this will update configuration when the specfile is updated
            self.save_main()
            self.t.notify(**{'specfile':self.specfile})
        else:
            msg_window('set experiment first and then update spec file')


    def run_everything(self):
        """
        Runs everything.py user script in bin directory.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        if not self.is_exp_exists():
            msg_window('the experiment has not been created yet')
        elif not self.is_exp_set():
            msg_window('the experiment has changed, pres "set experiment" button')
        elif self.t is not None:
            self.t.run_all()


    def reset_window(self):
        self.exp_id = None
        self.experiment_dir = None
        self.working_dir = None
        self.specfile = None
        if self.t is not None:
            self.t.clear_configs()


    def set_working_dir(self):
        """
        It shows the select dialog for user to select working directory. If the selected directory does not exist user will see info message.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        self.working_dir = select_dir(os.getcwd())
        if self.working_dir is not None:
            self.set_work_dir_button.setStyleSheet("Text-align:left")
            self.set_work_dir_button.setText(self.working_dir)
        else:
            self.set_work_dir_button.setText('')
            msg_window('please select valid working directory')
            return


    def is_exp_exists(self):
        """
        Determines if minimum information for creating the experiment space exists, i.e the working directory and experiment id must be set.
        Resolves the experiment name, and create experiment directory if it does not exist.
        Parameters
        ----------
        none
        Returns
        -------
        boolean
            True if experiment exists, False otherwise
        """
        if self.exp_id is None:
            return False
        if self.working_dir is None:
            return False
        exp_id = str(self.Id_widget.text()).strip()
        scan = str(self.scan_widget.text()).replace(' ','')
        if scan != '':
            exp_id = exp_id + '_' + scan
        if not os.path.exists(self.working_dir + '/' + exp_id):
            return False
        return True


    def is_exp_set(self):
        """
        The GUI can be used to load an experiment, and then change the parameters, such id or scan. This function will return True if information in class are the same as in the GUI.
        Parameters
        ----------
        none
        Returns
        -------
        boolean
            True if experiment has been set, False otherwise
        """
        if self.exp_id is None:
            return False
        if self.working_dir is None:
            return False
        if self.id != str(self.Id_widget.text()).strip():
            return False
        return True


    def load_experiment(self):
        """
        It shows a dialog for user to select previously created experiment directory. If no main configuration file is found user will see info message.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        load_dir = select_dir(os.getcwd().replace(os.sep, '/')).replace(os.sep, '/')
        if load_dir is None:
            msg_window('please select valid conf directory')
            return
        if os.path.isfile(load_dir + '/conf/config'):
            need_convert = self.load_main(load_dir)
            # config file could not be parsed
            if need_convert is None:
                return
        else:
            msg_window('missing conf/config file, not experiment directory')
            return

        self.reset_window()
        self.set_experiment(True)
        if self.t is None:
            try:
                self.t = Tabs(self, self.beamline_widget.text())
                self.vbox.addWidget(self.t)
            except:
                pass
        self.t.load_conf(load_dir, need_convert)
        if not self.is_exp_set():
            return
        self.save_main()
        if need_convert:
            self.t.save_conf()


    def load_main(self, load_dir):
        """
        It reads 'config' file from the given directory, parses all parameters, verifies, and sets the display in window and class members to parsed values.
        Parameters
        ----------
        load_dir : str
            a directory to load the main configuration from
        Returns
        -------
        nothing
        """
        load_dir = load_dir.replace(os.sep, '/')
        conf = load_dir + '/conf/config'
        conf_map = ut.read_config(conf)
        if conf_map is None:
            msg_window('please check configuration file ' + conf + '. Cannot parse, ')
            return None

        self.working_dir = None
        need_convert = False
        try:
            working_dir = conf_map['working_dir'].replace(os.sep, '/')
            self.set_work_dir_button.setStyleSheet("Text-align:left")
            self.set_work_dir_button.setText(working_dir)
        except:
            self.set_work_dir_button.setText('')

        # if the converter version in config file is old or none, get the conf with new version
        if 'converter_ver' in conf_map:
            exp_converter_ver = conf_map['converter_ver']
        else:
            exp_converter_ver = None
        if exp_converter_ver is None or exp_converter_ver < conv.get_version():
            conf_map = conv.get_conf_dict(load_dir + '/conf/config', 'config')
            need_convert = True

        try:
            self.Id_widget.setText(conf_map['experiment_id'])
        except:
            self.Id_widget.setText('')
        try:
            self.scan_widget.setText(conf_map['scan'].replace(' ',''))
        except:
            self.scan_widget.setText('')

        try:
            specfile = conf_map['specfile']
            if os.path.isfile(specfile):
                self.spec_file_button.setStyleSheet("Text-align:left")
                self.spec_file_button.setText(specfile)
            else:
                msg_window('The specfile file ' + specfile + ' in config file does not exist')
        except:
            self.spec_file_button.setText('')

        try:
            self.beamline_widget.setText(conf_map['beamline'])
        except:
            self.beamline_widget.setText('')

        return need_convert


    def assure_experiment_dir(self):
        """
        It creates experiment directory, and experiment configuration directory if they dp not exist.
        Parameters
        ----------
        nothing
        Returns
        -------
        nothing
        """
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
        experiment_conf_dir = self.experiment_dir + '/conf'
        if not os.path.exists(experiment_conf_dir):
            os.makedirs(experiment_conf_dir)


    def save_main(self):
        # read the configurations from GUI and write to experiment config files
        # save the main config
        conf_map = {}
        conf_map['working_dir'] = str(self.working_dir)
        conf_map['experiment_id'] = self.id
        if len(self.scan_widget.text()) > 0:
            conf_map['scan'] = str(self.scan_widget.text())
        if self.beamline is not None:
            conf_map['beamline'] = self.beamline
        if self.specfile is not None:
            conf_map['specfile'] = str(self.specfile)
        conf_map['converter_ver'] = conv.get_version()
        er_msg = cohere.verify('config', conf_map)
        if len(er_msg) > 0:
            msg_window(er_msg)
        else:
            ut.write_config(conf_map, self.experiment_dir + '/conf/config')


    def set_experiment(self, loaded=False):
        """
        Reads the parameters in the window, and sets the experiment to this values, i.e. creates experiment directory,
        and saves all configuration files with parameters from window.

        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        working_dir = self.set_work_dir_button.text().replace(os.sep, '/')
        if len(working_dir) == 0:
            msg_window(
                'The working directory is not defined in config file. Select valid working directory and set experiment')
            return
        elif not os.path.isdir(working_dir):
            msg_window(
                'The working directory ' + working_dir + ' from config file does not exist. Select valid working directory and set experiment')
            self.set_work_dir_button.setText('')
            return
        elif not os.access(working_dir, os.W_OK):
            msg_window(
                'The working directory ' + working_dir + ' is not writable. Select valid working directory and set experiment')
            self.set_work_dir_button.setText('')
            return

        id = str(self.Id_widget.text()).strip()
        if id == '':
            msg_window('id must be entered')
            return

        self.working_dir = working_dir
        self.id = id
        if len(self.scan_widget.text()) > 0:
            self.exp_id = self.id + '_' + str(self.scan_widget.text())
        else:
            self.exp_id = self.id
        self.experiment_dir = self.working_dir + '/' + self.exp_id
        self.assure_experiment_dir()

        if len(self.beamline_widget.text().strip()) > 0:
            self.beamline = str(self.beamline_widget.text()).strip()
        else:
            self.beamline = None
        if len(self.spec_file_button.text()) > 0:
            self.specfile = str(self.spec_file_button.text()).strip()
        else:
            self.specfile = None

        if not loaded:
            self.save_main()
            if self.t is None:
                try:
                    self.t = Tabs(self, self.beamline_widget.text())
                    self.vbox.addWidget(self.t)
                except:
                    pass
            self.t.save_conf()
        try:
#            print("notify")
            self.t.notify(specfile=self.specfile)
        except:
            pass


class Tabs(QTabWidget):
    """
    The main window contains four tabs, each tab holding parameters for different part of processing.
    The tabs are as follows: prep (prepare data), data (format data), rec (reconstruction), disp (visualization).
    This class holds holds the tabs.
    """
    def __init__(self, main_win, beamline, parent=None):
        """
        Constructor, initializes the tabs.
        """
        super(Tabs, self).__init__(parent)
        self.main_win = main_win

        if beamline is not None and len(beamline) > 0:
            try:
                beam = importlib.import_module('beamlines.' + beamline + '.beam_tabs')
            except Exception as e:
                print (e)
                msg_window('cannot import beamlines.' + beamline + ' module' )
                raise
            self.prep_tab = beam.PrepTab()
            self.format_tab = DataTab()
            self.rec_tab = RecTab()
            self.display_tab = beam.DispTab()
            self.tabs = [self.prep_tab, self.format_tab, self.rec_tab, self.display_tab]
        else:
            self.format_tab = DataTab()
            self.rec_tab = RecTab()
            self.tabs = [self.format_tab, self.rec_tab]

        for tab in self.tabs:
            self.addTab(tab, tab.name)
            tab.init(self, main_win)


    def notify(self, **args):
        try:
            self.display_tab.update_tab(**args)
            self.prep_tab.update_tab(**args)
        except:
            pass


    def clear_configs(self):
        for tab in self.tabs:
            tab.clear_conf()


    def run_all(self):
        for tab in self.tabs:
            tab.run_tab()

    def run_prep(self):
        import beamline_preprocess as prep

        # this line is passing all parameters from command line to prep script. 
        # if there are other parameters, one can add some code here
        prep.handle_prep(self.main_win.experiment_dir, self.main_win.args)

    def run_viz(self):
        import beamline_visualization as dp

        dp.handle_visualization(self.main_win.experiment_dir)


    def load_conf(self, load_dir, need_convert):
        for tab in self.tabs:
            tab.load_tab(load_dir, need_convert)


    def save_conf(self):
        for tab in self.tabs:
            tab.save_conf()


class DataTab(QWidget):
    def __init__(self, parent=None):
        """
        Constructor, initializes the tabs.
        """
        super(DataTab, self).__init__(parent)
        self.name = 'Data'


    def init(self, tabs, main_window):
        """
        Creates and initializes the 'data' tab.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        self.tabs = tabs
        self.main_win = main_window

        layout = QFormLayout()
        self.alien_alg = QComboBox()
        self.alien_alg.addItem("none")
        self.alien_alg.addItem("block aliens")
        self.alien_alg.addItem("alien file")
        self.alien_alg.addItem("AutoAlien1")
        layout.addRow("alien algorithm", self.alien_alg)
        sub_layout = QFormLayout()
        self.set_alien_layout(sub_layout)
        layout.addRow(sub_layout)
        self.intensity_threshold = QLineEdit()
        layout.addRow("Intensity Threshold", self.intensity_threshold)
        self.center_shift = QLineEdit()
        layout.addRow("center_shift", self.center_shift)
        self.adjust_dimensions = QLineEdit()
        layout.addRow("pad, crop", self.adjust_dimensions)
        self.binning = QLineEdit()
        layout.addRow("binning", self.binning)
        cmd_layout = QHBoxLayout()
        self.set_data_conf_from_button = QPushButton("Load data conf from")
        self.set_data_conf_from_button.setStyleSheet("background-color:rgb(205,178,102)")
        self.config_data_button = QPushButton('format data', self)
        self.config_data_button.setStyleSheet("background-color:rgb(175,208,156)")
        cmd_layout.addWidget(self.set_data_conf_from_button)
        cmd_layout.addWidget(self.config_data_button)
        layout.addRow(cmd_layout)
        self.setLayout(layout)

        self.alien_alg.currentIndexChanged.connect(lambda: self.set_alien_layout(sub_layout))
        # this will create config_data file and run data script
        # to generate data ready for reconstruction
        self.config_data_button.clicked.connect(self.run_tab)
        self.set_data_conf_from_button.clicked.connect(self.load_data_conf)


    def clear_conf(self):
        self.alien_alg.setCurrentIndex(0)
        self.intensity_threshold.setText('')
        self.binning.setText('')
        self.center_shift.setText('')
        self.adjust_dimensions.setText('')


    def load_tab(self, load_from, need_convert):
        """
        It verifies given configuration file, reads the parameters, and fills out the window.
        Parameters
        ----------
        conf : str
            configuration file (config_data)
        Returns
        -------
        nothing
        """
        load_from = load_from.replace(os.sep, '/')
        if os.path.isfile(load_from):
            conf = load_from
        else:
            conf = load_from + '/conf/config_data'
            if not os.path.isfile(conf):
                msg_window('info: the load directory does not contain config_data file')
                return
        if need_convert:
            conf_map = conv.get_conf_dict(conf, 'config_data')
        else:
            conf_map = ut.read_config(conf)
            if conf_map is None:
                msg_window('please check configuration file ' + conf)
                return
        if 'alien_alg' not in conf_map:
            conf_map['alien_alg'] = 'random'
        if conf_map['alien_alg'] == 'random':
            self.alien_alg.setCurrentIndex(0)
        elif conf_map['alien_alg'] == 'block_aliens':
            self.alien_alg.setCurrentIndex(1)
            if 'aliens' in conf_map:
                self.aliens.setText(str(conf_map['aliens']).replace(" ", ""))
        elif conf_map['alien_alg'] == 'alien_file':
            self.alien_alg.setCurrentIndex(2)
            if 'alien_file' in conf_map:
                self.alien_file.setText(str(conf_map['alien_file']).replace(" ", ""))
        elif conf_map['alien_alg'] == 'AutoAlien1':
            self.alien_alg.setCurrentIndex(3)
            if 'AA1_size_threshold' in conf_map:
                self.AA1_size_threshold.setText(str(conf_map['AA1_size_threshold']).replace(" ", ""))
            if 'AA1_asym_threshold' in conf_map:
                self.AA1_asym_threshold.setText(str(conf_map['AA1_asym_threshold']).replace(" ", ""))
            if 'AA1_min_pts' in conf_map:
                self.AA1_min_pts.setText(str(conf_map['AA1_min_pts']).replace(" ", ""))
            if 'AA1_eps' in conf_map:
                self.AA1_eps.setText(str(conf_map['AA1_eps']).replace(" ", ""))
            if 'AA1_amp_threshold' in conf_map:
                self.AA1_amp_threshold.setText(str(conf_map['AA1_amp_threshold']).replace(" ", ""))
            if 'AA1_save_arrs' in conf_map:
                self.AA1_save_arrs.setChecked(conf_map['AA1_save_arrs'])
            else:
                self.AA1_save_arrs.setChecked(False)
            if 'AA1_expandcleanedsigma' in conf_map:
                self.AA1_expandcleanedsigma.setText(str(conf_map['AA1_expandcleanedsigma']).replace(" ", ""))
        if 'intensity_threshold' in conf_map:
            self.intensity_threshold.setText(str(conf_map['intensity_threshold']).replace(" ", ""))
        if 'binning' in conf_map:
            self.binning.setText(str(conf_map['binning']).replace(" ", ""))
        if 'center_shift' in conf_map:
            self.center_shift.setText(str(conf_map['center_shift']).replace(" ", ""))
        if 'adjust_dimensions' in conf_map:
            self.adjust_dimensions.setText(str(conf_map['adjust_dimensions']).replace(" ", ""))


    def get_data_config(self):
        """
        It reads parameters related to formatting data from the window and adds them to dictionary.
        Parameters
        ----------
        none
        Returns
        -------
        conf_map : dict
            contains parameters read from window
        """
        conf_map = {}

        if self.alien_alg.currentIndex() == 1:
            conf_map['alien_alg'] = 'block_aliens'
            if len(self.aliens.text()) > 0:
                conf_map['aliens'] = str(self.aliens.text()).replace('\n', '')
        if self.alien_alg.currentIndex() == 2:
            conf_map['alien_alg'] = 'alien_file'
            if len(self.alien_file.text()) > 0:
                conf_map['alien_file'] = str(self.alien_file.text())
        elif self.alien_alg.currentIndex() == 3:
            conf_map['alien_alg'] = 'AutoAlien1'
            if len(self.AA1_size_threshold.text()) > 0:
                conf_map['AA1_size_threshold'] = ast.literal_eval(str(self.AA1_size_threshold.text()))
            if len(self.AA1_asym_threshold.text()) > 0:
                conf_map['AA1_asym_threshold'] = ast.literal_eval(str(self.AA1_asym_threshold.text()))
            if len(self.AA1_min_pts.text()) > 0:
                conf_map['AA1_min_pts'] = ast.literal_eval(str(self.AA1_min_pts.text()))
            if len(self.AA1_eps.text()) > 0:
                conf_map['AA1_eps'] = ast.literal_eval(str(self.AA1_eps.text()))
            if len(self.AA1_amp_threshold.text()) > 0:
                conf_map['AA1_amp_threshold'] = ast.literal_eval(str(self.AA1_amp_threshold.text()))
            if self.AA1_save_arrs.isChecked():
                conf_map['AA1_save_arrs'] = True
            if len(self.AA1_expandcleanedsigma.text()) > 0:
                conf_map['AA1_expandcleanedsigma'] = ast.literal_eval(str(self.AA1_expandcleanedsigma.text()))

        if len(self.intensity_threshold.text()) > 0:
            conf_map['intensity_threshold'] = ast.literal_eval(str(self.intensity_threshold.text()))
        if len(self.binning.text()) > 0:
            conf_map['binning'] = ast.literal_eval(str(self.binning.text()).replace('\n', ''))
        if len(self.center_shift.text()) > 0:
            conf_map['center_shift'] = ast.literal_eval(str(self.center_shift.text()).replace('\n', ''))
        if len(self.adjust_dimensions.text()) > 0:
            conf_map['adjust_dimensions'] = ast.literal_eval(str(self.adjust_dimensions.text()).replace('\n', ''))

        return conf_map


    def set_alien_layout(self, layout):
        for i in reversed(range(layout.count())):
            layout.itemAt(i).widget().setParent(None)
        if self.alien_alg.currentIndex() == 1:
            self.aliens = QLineEdit()
            layout.addRow("aliens", self.aliens)
        elif self.alien_alg.currentIndex() == 2:
            self.alien_file = QPushButton()
            layout.addRow("alien file", self.alien_file)
            self.alien_file.clicked.connect(self.set_alien_file)
        elif self.alien_alg.currentIndex() == 3:
            self.AA1_size_threshold = QLineEdit()
            layout.addRow("relative size threshold", self.AA1_size_threshold)
            self.AA1_asym_threshold = QLineEdit()
            layout.addRow("average asymmetry threshold", self.AA1_asym_threshold)
            self.AA1_min_pts = QLineEdit()
            layout.addRow("min pts in cluster", self.AA1_min_pts)
            self.AA1_eps = QLineEdit()
            layout.addRow("cluster alg eps", self.AA1_eps)
            self.AA1_amp_threshold = QLineEdit()
            layout.addRow("alien alg amp threshold", self.AA1_amp_threshold)
            self.AA1_save_arrs = QCheckBox()
            layout.addRow("save analysis arrs", self.AA1_save_arrs)
            self.AA1_save_arrs.setChecked(False)
            self.AA1_expandcleanedsigma = QLineEdit()
            layout.addRow("expand cleaned sigma", self.AA1_expandcleanedsigma)


    def set_alien_file(self):
        """
        It display a select dialog for user to select an alien file.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        self.alien_filename = select_file(os.getcwd())
        if self.alien_filename is not None:
            self.alien_file.setStyleSheet("Text-align:left")
            self.alien_file.setText(self.alien_filename)
        else:
            self.alien_file.setText('')


    def run_tab(self):
        """
        Reads the parameters needed by format data script. Saves the config_data configuration file with parameters from the window and runs the format script.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        import standard_preprocess as run_dt

        if not self.main_win.is_exp_exists():
            msg_window('the experiment has not been created yet')
        elif not self.main_win.is_exp_set():
            msg_window('the experiment has changed, pres "set experiment" button')
        elif len(self.intensity_threshold.text()) == 0:
            msg_window('Please, enter Intensity Threshold parameter')
        else:
            found_file = False
            for p, d, f in os.walk(self.main_win.experiment_dir):
                if 'prep_data.tif' in f:
                    found_file = True
                    break
            if found_file:
                conf_map = self.get_data_config()
                # verify that data configuration is ok
                er_msg = cohere.verify('config_data', conf_map)
                if len(er_msg) > 0:
                    msg_window(er_msg)
                    return
                ut.write_config(conf_map, self.main_win.experiment_dir + '/conf/config_data')
                run_dt.format_data(self.main_win.experiment_dir)
            else:
                msg_window('Please, run data preparation in previous tab to activate this function')


    def save_conf(self):
        # save data config
        conf_map = self.get_data_config()
        if len(conf_map) > 0:
            er_msg = cohere.verify('config_data', conf_map)
            if len(er_msg) > 0:
                msg_window(er_msg)
                return
            ut.write_config(conf_map, self.main_win.experiment_dir + '/conf/config_data')


    def load_data_conf(self):
        """
        It display a select dialog for user to select a configuration file. When selected, the parameters from that file will be loaded to the window.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        data_file = select_file(os.getcwd())
        if data_file is not None:
            self.load_tab(data_file)
        else:
            msg_window('please select valid data config file')


class RecTab(QWidget):
    def __init__(self, parent=None):
        """
        Constructor, initializes the tabs.
        """
        super(RecTab, self).__init__(parent)
        self.name = 'Reconstruction'


    def init(self, tabs, main_window):
        """
        Creates and initializes the 'reconstruction' tab.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        self.tabs = tabs
        self.main_win = main_window
        self.old_conf_id = ''

        layout = QVBoxLayout()
        ulayout = QFormLayout()
        mlayout = QHBoxLayout()

        self.init_guess = QComboBox()
        self.init_guess.InsertAtBottom
        self.init_guess.addItem("random")
        self.init_guess.addItem("continue")
        self.init_guess.addItem("AI algorithm")
        ulayout.addRow("initial guess", self.init_guess)
        sub_layout = QFormLayout()
        self.set_init_guess_layout(sub_layout)
        ulayout.addRow(sub_layout)

        self.add_conf_button = QPushButton('add configuration', self)
        ulayout.addWidget(self.add_conf_button)
        self.rec_id = QComboBox()
        self.rec_id.InsertAtBottom
        self.rec_id.addItem("main")
        ulayout.addWidget(self.rec_id)
        self.rec_id.hide()
        self.proc = QComboBox()
        self.proc.addItem("auto")
        if sys.platform != 'darwin':
            self.proc.addItem("cp")
        self.proc.addItem("np")
        self.proc.addItem("af")
        if sys.platform != 'darwin':
            self.proc.addItem("cuda")
        self.proc.addItem("opencl")
        self.proc.addItem("cpu")
        ulayout.addRow("processor type", self.proc)
        self.device = QLineEdit()
        ulayout.addRow("device(s)", self.device)
        self.reconstructions = QLineEdit()
        ulayout.addRow("number of reconstructions", self.reconstructions)
        self.alg_seq = QLineEdit()
        ulayout.addRow("algorithm sequence", self.alg_seq)
        # TODO add logic to show this only if HIO is in sequence
        self.hio_beta = QLineEdit()
        ulayout.addRow("HIO beta", self.hio_beta)
        self.initial_support_area = QLineEdit()
        ulayout.addRow("initial support area", self.initial_support_area)
        self.rec_default_button = QPushButton('set to defaults', self)
        ulayout.addWidget(self.rec_default_button)

        self.features = Features(self, mlayout)

        llayout = QHBoxLayout()
        self.set_rec_conf_from_button = QPushButton("Load rec conf from")
        self.set_rec_conf_from_button.setStyleSheet("background-color:rgb(205,178,102)")
        self.config_rec_button = QPushButton('run reconstruction', self)
        self.config_rec_button.setStyleSheet("background-color:rgb(175,208,156)")
        llayout.addWidget(self.set_rec_conf_from_button)
        llayout.addWidget(self.config_rec_button)

        spacer = QSpacerItem(0, 3)
        llayout.addItem(spacer)

        layout.addLayout(ulayout)
        layout.addLayout(mlayout)
        layout.addLayout(llayout)

        self.setAutoFillBackground(True)
        self.setLayout(layout)

        self.config_rec_button.clicked.connect(self.run_tab)
        self.init_guess.currentIndexChanged.connect(lambda: self.set_init_guess_layout(sub_layout))
        self.rec_default_button.clicked.connect(self.set_defaults)
        self.add_conf_button.clicked.connect(self.add_rec_conf)
        self.rec_id.currentIndexChanged.connect(self.toggle_conf)
        self.set_rec_conf_from_button.clicked.connect(self.load_rec_conf_dir)


    def load_tab(self, load_dir, need_convert):
        """
        It verifies given configuration file, reads the parameters, and fills out the window.
        Parameters
        ----------
        conf : str
            configuration file (config_rec)
        Returns
        -------
        nothing
        """
        load_dir = load_dir.replace(os.sep, '/')
        conf = load_dir + '/conf/config_rec'
        if not os.path.isfile(conf):
            msg_window('info: the load directory does not contain config_rec file')
            return
        if need_convert:
            conf_dict = conv.get_conf_dict(conf, 'config_rec')
            # if experiment set, save the config_rec
            try:
                ut.write_config(conf_dict, conf)
            except:
                pass
        else:
            conf_map = ut.read_config(conf)
            if conf_map is None:
                msg_window('please check configuration file ' + conf)
                return
        self.load_tab_common(conf_map)


    def load_tab_common(self, conf_map, update_rec_choice=True):
        if 'init_guess' not in conf_map:
            conf_map['init_guess'] = 'random'
        if conf_map['init_guess'] == 'random':
            self.init_guess.setCurrentIndex(0)
        elif conf_map['init_guess'] == 'continue':
            self.init_guess.setCurrentIndex(1)
            if 'continue_dir' in conf_map:
                self.cont_dir_button.setText(str(conf_map['continue_dir'].replace(os.sep, '/')).replace(" ", ""))
        elif conf_map['init_guess'] == 'AI_guess':
            self.init_guess.setCurrentIndex(2)
            if 'AI_trained_model' in conf_map:
                self.AI_trained_model.setText(str(conf_map['AI_trained_model'].replace(os.sep, '/')).replace(" ", ""))
                self.AI_trained_model.setStyleSheet("Text-align:left")

        # this will update the configuration choices by reading configuration files names
        # do not update when doing toggle
        if update_rec_choice:
            self.update_rec_configs_choice()

        if 'processing' in conf_map:
            self.proc.setCurrentText(str(conf_map['processing']))
        if 'device' in conf_map:
            self.device.setText(str(conf_map['device']).replace(" ", ""))
        if 'reconstructions' in conf_map:
            self.reconstructions.setText(str(conf_map['reconstructions']).replace(" ", ""))
        if 'algorithm_sequence' in conf_map:
            self.alg_seq.setText(str(conf_map['algorithm_sequence']))
        if 'hio_beta' in conf_map:
            self.hio_beta.setText(str(conf_map['hio_beta']).replace(" ", ""))
        if 'initial_support_area' in conf_map:
            self.initial_support_area.setText(str(conf_map['initial_support_area']).replace(" ", ""))

        for feat_id in self.features.feature_dir:
            self.features.feature_dir[feat_id].init_config(conf_map)

        self.notify()


    def clear_conf(self):
        self.init_guess.setCurrentIndex(0)
        self.device.setText('')
        self.proc.setCurrentIndex(0)
        self.reconstructions.setText('')
        self.alg_seq.setText('')
        self.hio_beta.setText('')
        self.initial_support_area.setText('')
        for feat_id in self.features.feature_dir:
            self.features.feature_dir[feat_id].active.setChecked(False)


    def get_rec_config(self):
        """
        It reads parameters related to reconstruction from the window and adds them to dictionary.
        Parameters
        ----------
        none
        Returns
        -------
        conf_map : dict
            contains parameters read from window
        """
        conf_map = {}
        if len(self.reconstructions.text()) > 0:
            try:
                conf_map['reconstructions'] = ast.literal_eval(self.reconstructions.text())
            except:
                msg_window('reconstructions parameter should be int')
                return {}
        if len(self.proc.currentText()) > 0:
            conf_map['processing'] = str(self.proc.currentText())
        if len(self.device.text()) > 0:
            try:
                conf_map['device'] = ast.literal_eval(str(self.device.text()).replace('\n',''))
            except:
                msg_window('device parameter should be a list of int')
                return {}
        if len(self.alg_seq.text()) > 0:
            conf_map['algorithm_sequence'] = str(self.alg_seq.text()).strip()
        if len(self.hio_beta.text()) > 0:
            try:
                conf_map['hio_beta'] = ast.literal_eval(str(self.hio_beta.text()))
            except:
                msg_window('hio_beta parameter should be float')
                return {}
        if len(self.initial_support_area.text()) > 0:
            try:
                conf_map['initial_support_area'] = ast.literal_eval(str(self.initial_support_area.text()).replace('\n',''))
            except:
                msg_window('initial_support_area parameter should be a list of floats')
                return {}
        if self.init_guess.currentIndex() == 1:
            conf_map['init_guess'] = 'continue'
            if len(self.cont_dir_button.text().strip()) > 0:
                conf_map['continue_dir'] = str(self.cont_dir_button.text()).replace(os.sep, '/').strip()
        elif self.init_guess.currentIndex() == 2:
            conf_map['init_guess'] = 'AI_guess'
            if len(self.AI_trained_model.text()) > 0:
                conf_map['AI_trained_model'] = str(self.AI_trained_model.text()).replace(os.sep, '/').strip()
        for feat_id in self.features.feature_dir:
            self.features.feature_dir[feat_id].add_config(conf_map)

        return conf_map


    def save_conf(self):
        conf_map = self.get_rec_config()
        if len(conf_map) == 0:
            return
        er_msg = cohere.verify('config_rec', conf_map)
        if len(er_msg) > 0:
            msg_window(er_msg)
            return
        if len(conf_map) > 0:
            ut.write_config(conf_map, self.main_win.experiment_dir + '/conf/config_rec')


    def set_init_guess_layout(self, layout):
        for i in reversed(range(layout.count())):
            layout.itemAt(i).widget().setParent(None)
        if self.init_guess.currentIndex() == 1:
            self.cont_dir_button = QPushButton()
            layout.addRow("continue directory", self.cont_dir_button)
            self.cont_dir_button.clicked.connect(self.set_cont_dir)
        elif self.init_guess.currentIndex() == 2:
            self.AI_trained_model = QPushButton()
            layout.addRow("AI trained model file", self.AI_trained_model)
            self.AI_trained_model.clicked.connect(self.set_aitm_file)


    def set_cont_dir(self):
        """
        It display a select dialog for user to select a directory with raw data file.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        cont_dir = select_dir(os.getcwd().replace(os.sep, '/')).replace(os.sep, '/')
        if cont_dir is not None:
            self.cont_dir_button.setStyleSheet("Text-align:left")
            self.cont_dir_button.setText(cont_dir)
        else:
            self.cont_dir_button.setText('')


    def set_aitm_file(self):
        AI_trained_model = select_file(os.getcwd().replace(os.sep, '/')).replace(os.sep, '/')
        if AI_trained_model is not None:
            self.AI_trained_model.setStyleSheet("Text-align:left")
            self.AI_trained_model.setText(AI_trained_model)
        else:
            self.AI_trained_model.setText('')


    def add_rec_conf(self):
        id, ok = QInputDialog.getText(self, '', "enter configuration id")
        if id in self.rec_ids:
            msg_window('the ' + id + ' is alredy used')
            return
        if ok and len(id) > 0:
            if len(self.rec_ids) > 1:
                self.rec_id.addItem(id)
            else:
                self.rec_id.show()
                self.rec_id.addItem(id)
        else:
            return

        # copy the config_rec into <id>_config_rec
        conf_file = self.main_win.experiment_dir + '/conf/config_rec'
        new_conf_file = self.main_win.experiment_dir + '/conf/config_rec_' + id
        shutil.copyfile(conf_file, new_conf_file)
        self.rec_id.setCurrentIndex(self.rec_id.count() - 1)


    def toggle_conf(self):
        """
        Invoked when the configuration to use in the reconstruction was changed. This will bring the parameters from
        the previous config to be saved, and the new ones retrieved and showed in window.
        Parameters
        ----------
        layout : QFormLayout
            a layout to add the continue dir

        Returns
        -------
        nothing
        """
        # save the configuration file before updating the incoming config
        if self.old_conf_id == '':
            conf_file = 'config_rec'
        else:
            conf_file =  'config_rec_' + self.old_conf_id

        conf_map = self.get_rec_config()
        if len(conf_map) == 0:
            return
        conf_dir = self.main_win.experiment_dir + '/conf'

        ut.write_config(conf_map, conf_dir + '/' + conf_file)
        if str(self.rec_id.currentText()) == 'main':
            self.old_conf_id = ''
        else:
            self.old_conf_id = str(self.rec_id.currentText())
        # if a config file corresponding to the rec id exists, load it
        # otherwise read from base configuration and load
        if self.old_conf_id == '':
            conf_file = conf_dir + '/config_rec'
        else:
            conf_file = conf_dir +  '/config_rec_' + self.old_conf_id

        conf_map = ut.read_config(conf_file)
        if conf_map is None:
            msg_window('please check configuration file ' + conf_file)
            return
        self.load_tab_common(conf_map, False)
        self.notify()


    def load_rec_conf_dir(self):
        """
        It display a select dialog for user to select a configuration file. When selected, the parameters from that file will be loaded to the window.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        rec_file = select_file(os.getcwd())
        if rec_file is not None:
            conf_map = ut.read_config(rec_file)
            if conf_map is None:
                msg_window('please check configuration file ' + rec_file)
                return

            self.load_tab_common(conf_map)
        else:
            msg_window('please select valid rec config file')


    def run_tab(self):
        """
        Reads the parameters needed by reconstruction script. Saves the config_rec configuration file with parameters from the window and runs the reconstruction script.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        import run_reconstruction as run_rc

        if not self.main_win.is_exp_exists():
            msg_window('the experiment has not been created yet')
        elif not self.main_win.is_exp_set():
            msg_window('the experiment has changed, pres "set experiment" button')
        else:
            found_file = False
            for p, d, f in os.walk(self.main_win.experiment_dir):
                if 'data.tif' in f:
                    found_file = True
                    break
                if 'data.npy' in f:
                    found_file = True
                    break
            if found_file:
                # find out which configuration should be saved
                if self.old_conf_id == '':
                    conf_file = 'config_rec'
                    conf_id = None
                else:
                    conf_file =  'config_rec_' + self.old_conf_id
                    conf_id = self.old_conf_id

                conf_map = self.get_rec_config()
                if len(conf_map) == 0:
                    return

                # verify that reconstruction configuration is ok
                er_msg = cohere.verify('config_rec', conf_map)
                if len(er_msg) > 0:
                    msg_window(er_msg)
                    return
                ut.write_config(conf_map, self.main_win.experiment_dir + '/conf/' + conf_file)
                run_rc.manage_reconstruction(self.main_win.experiment_dir, conf_id)
                self.notify()
            else:
                msg_window('Please, run format data in previous tab to activate this function')


    def set_defaults(self):
        """
        Sets the basic parameters in the reconstruction tab main part to hardcoded defaults.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        if self.main_win.working_dir is None or self.main_win.id is None or \
                        len(self.main_win.working_dir) == 0 or len(self.main_win.id) == 0:
            msg_window('Working Directory or Reconstruction ID not configured')
        else:
            self.reconstructions.setText('1')
            self.proc.setCurrentIndex(0)
            self.device.setText('[0,1]')
            self.alg_seq.setText('3*(20*ER+180*HIO)+20*ER')
            self.hio_beta.setText('.9')
            self.initial_support_area.setText('[0.5, 0.5, 0.5]')


    def update_rec_configs_choice(self):
        """
        Looks for alternate reconstruction configurations, and updates window with that information.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        # this will update the configuration choices in reconstruction tab
        # fill out the config_id choice bar by reading configuration files names
        if not self.main_win.is_exp_set():
            return
        self.rec_ids = []
        for file in os.listdir(self.main_win.experiment_dir + '/conf'):
            if file.startswith('config_rec_'):
                self.rec_ids.append(file[len('config_rec_') : len(file)])
        if len(self.rec_ids) > 0:
            self.rec_id.addItems(self.rec_ids)
            self.rec_id.show()


    def notify(self):
        generations = 0
        if self.features.feature_dir['GA'].active.isChecked():
            generations = int(self.features.feature_dir['GA'].generations.text())
        if len(self.reconstructions.text()) > 0:
            rec_no = int(self.reconstructions.text())
        else:
            rec_no = 1
        self.tabs.notify(**{'rec_id':self.old_conf_id, 'generations':generations, 'rec_no':rec_no})


class Feature(object):
    """
    This is a parent class to concrete feature classes.
    """
    def __init__(self):
        """
        Constructor, each feature object contains QWidget.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        self.stack = QWidget()


    def stackUI(self, item, feats):
        """
        Used by all sub-classes (features) when initialized.
        Parameters
        ----------
        item : item from QListWidget
            item represents a feature
        feats : Features object
            Features object is a composition of features
        Returns
        -------
        nothing
        """
        layout = QFormLayout()
        self.active = QCheckBox("active")
        layout.addWidget(self.active)
        self.toggle(layout, item, feats)
        self.stack.setLayout(layout)
        self.active.stateChanged.connect(lambda: self.toggle(layout, item, feats))


    def toggle(self, layout, item, feats):
        """
        Used by sub-classes (features) when a feature is activated or deactivated.
        Parameters
        ----------
        item : item from QListWidget
            item represents a feature
        feats : Features object
            Features object is a composition of features
        Returns
        -------
        nothing
        """
        if self.active.isChecked():
            self.fill_active(layout)

            self.default_button = QPushButton('set to defaults', feats)
            layout.addWidget(self.default_button)
            self.default_button.clicked.connect(self.rec_default)

            item.setForeground(QColor('black'));
        else:
            self.clear_params(layout, item)


    def clear_params(self, layout, item):
        for i in reversed(range(1, layout.count())):
            layout.itemAt(i).widget().setParent(None)
        item.setForeground(QColor('grey'));


    def fill_active(self, layout):
        """
        This function is overriden in concrete class. It displays the feature's parameters when the feature becomes active.
        Parameters
        ----------
        layout : Layout widget
            a layout with the feature
        Returns
        -------
        nothing
        """
        pass


    def rec_default(self):
        """
        This function is overriden in concrete class. It sets feature's parameters to hardcoded default values.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        pass


    def add_config(self, conf_map):
        """
        This function calls all of the subclasses to add feature's parameters to dictionary.
        Parameters
        ----------
        conf_map : dict
            contains parameters for reconstruction
        Returns
        -------
        nothing
        """
        if self.active.isChecked():
            self.add_feat_conf(conf_map)


    def add_feat_conf(self, conf_map):
        """
        This function is overriden in concrete class. It adds feature's parameters to dictionary.
        Parameters
        ----------
        conf_map : dict
            contains parameters for reconstruction
        Returns
        -------
        nothing
        """
        pass


    def init_config(self, conf_map):
        """
        This function is overriden in concrete class. It sets feature's parameters to parameters in dictionary and displays in the window.
        Parameters
        ----------
        conf_map : dict
            contains parameters for reconstruction
        Returns
        -------
        nothing
        """
        pass


class GA(Feature):
    """
    This class encapsulates GA feature.
    """
    def __init__(self):
        super(GA, self).__init__()
        self.id = 'GA'

    # override setting the active to set it False
    def stackUI(self, item, feats):
        super(GA, self).stackUI(item, feats)


    def init_config(self, conf_map):
        """
        This function sets GA feature's parameters to parameters in dictionary and displays in the window.
        Parameters
        ----------
        conf_map : dict
            contains parameters for reconstruction
        Returns
        -------
        nothing
        """
        if 'ga_generations' in conf_map:
            gens = conf_map['ga_generations']
            self.active.setChecked(True)
            self.generations.setText(str(gens).replace(" ", ""))
        else:
            self.active.setChecked(False)
            return
        if 'ga_fast' in conf_map:
            ga_fast = conf_map['ga_fast']
            if ga_fast:
                self.ga_fast.setChecked(True)
            else:
                self.ga_fast.setChecked(False)
        else:
            self.ga_fast.setChecked(False)
        if 'ga_metrics' in conf_map:
            self.metrics.setText(str(conf_map['ga_metrics']).replace(" ", ""))
        if 'ga_breed_modes' in conf_map:
            self.breed_modes.setText(str(conf_map['ga_breed_modes']).replace(" ", ""))
        if 'ga_cullings' in conf_map:
            self.removes.setText(str(conf_map['ga_cullings']).replace(" ", ""))
        if 'ga_shrink_wrap_thresholds' in conf_map:
            self.ga_shrink_wrap_thresholds.setText(str(conf_map['ga_shrink_wrap_thresholds']).replace(" ", ""))
        if 'ga_shrink_wrap_gauss_sigmas' in conf_map:
            self.ga_shrink_wrap_gauss_sigmas.setText(str(conf_map['ga_shrink_wrap_gauss_sigmas']).replace(" ", ""))
        if 'ga_lowpass_filter_sigmas' in conf_map:
            self.lr_sigmas.setText(str(conf_map['ga_lowpass_filter_sigmas']).replace(" ", ""))
        if 'ga_gen_pc_start' in conf_map:
            self.gen_pc_start.setText(str(conf_map['ga_gen_pc_start']).replace(" ", ""))


    def fill_active(self, layout):
        """
        This function displays the feature's parameters when the feature becomes active.
        Parameters
        ----------
        layout : Layout widget
            a layout with the feature
        Returns
        -------
        nothing
        """
        self.ga_fast = QCheckBox("fast processing, size limited")
        self.ga_fast.setChecked(False)
        layout.addWidget(self.ga_fast)
        self.generations = QLineEdit()
        layout.addRow("generations", self.generations)
        self.metrics = QLineEdit()
        layout.addRow("fitness metrics", self.metrics)
        self.breed_modes = QLineEdit()
        layout.addRow("breed modes", self.breed_modes)
        self.removes = QLineEdit()
        layout.addRow("cullings", self.removes)
        self.ga_shrink_wrap_thresholds = QLineEdit()
        layout.addRow("after breed support thresholds", self.ga_shrink_wrap_thresholds)
        self.ga_shrink_wrap_gauss_sigmas = QLineEdit()
        layout.addRow("after breed shrink wrap sigmas", self.ga_shrink_wrap_gauss_sigmas)
        self.lr_sigmas = QLineEdit()
        layout.addRow("low resolution sigmas", self.lr_sigmas)
        self.gen_pc_start = QLineEdit()
        layout.addRow("gen to start pcdi", self.gen_pc_start)


    def rec_default(self):
        """
        This function sets GA feature's parameters to hardcoded default values.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        self.generations.setText('5')
        self.metrics.setText('["chi","area"]')
        self.breed_modes.setText('["sqrt_ab"]')
        self.removes.setText('[2,2,1]')
        self.ga_shrink_wrap_thresholds.setText('[.1]')
        self.ga_shrink_wrap_gauss_sigmas.setText('[1.0]')
        self.lr_sigmas.setText('[2.0,1.5]')
        self.gen_pc_start.setText('3')
        self.active.setChecked(True)


    def add_feat_conf(self, conf_map):
        """
        This function adds GA feature's parameters to dictionary.
        Parameters
        ----------
        conf_map : dict
            contains parameters for reconstruction
        Returns
        -------
        nothing
        """
        if self.ga_fast.isChecked():
            conf_map['ga_fast'] = True
        if len(self.generations.text()) > 0:
            conf_map['ga_generations'] = ast.literal_eval(str(self.generations.text()))
        if len(self.metrics.text()) > 0:
         conf_map['ga_metrics'] = ast.literal_eval(str(self.metrics.text()).replace('\n',''))
        if len(self.breed_modes.text()) > 0:
          conf_map['ga_breed_modes'] = ast.literal_eval(str(self.breed_modes.text()).replace('\n',''))
        if len(self.removes.text()) > 0:
           conf_map['ga_cullings'] = ast.literal_eval(str(self.removes.text()).replace('\n',''))
        if len(self.ga_shrink_wrap_thresholds.text()) > 0:
            conf_map['ga_shrink_wrap_thresholds'] = ast.literal_eval(str(self.ga_shrink_wrap_thresholds.text()).replace('\n',''))
        if len(self.ga_shrink_wrap_gauss_sigmas.text()) > 0:
            conf_map['ga_shrink_wrap_gauss_sigmas'] = ast.literal_eval(str(self.ga_shrink_wrap_gauss_sigmas.text()).replace('\n',''))
        if len(self.lr_sigmas.text()) > 0:
            conf_map['ga_lowpass_filter_sigmas'] = ast.literal_eval(str(self.lr_sigmas.text()).replace('\n',''))
        if len(self.gen_pc_start.text()) > 0:
            conf_map['ga_gen_pc_start'] = ast.literal_eval(str(self.gen_pc_start.text()))


class low_resolution(Feature):
    """
    This class encapsulates low resolution feature.
    """
    def __init__(self):
        super(low_resolution, self).__init__()
        self.id = 'low resolution'


    def init_config(self, conf_map):
        """
        This function sets low resolution feature's parameters to parameters in dictionary and displays in the window.
        Parameters
        ----------
        conf_map : dict
            contains parameters for reconstruction
        Returns
        -------
        nothing
        """
        if 'resolution_trigger' in conf_map:
            triggers = conf_map['resolution_trigger']
            self.active.setChecked(True)
            self.res_triggers.setText(str(triggers).replace(" ", ""))
        else:
            self.active.setChecked(False)
            return
        if 'lowpass_filter_sw_sigma_range' in conf_map:
            self.sigma_range.setText(str(conf_map['lowpass_filter_sw_sigma_range']).replace(" ", ""))
        if 'lowpass_filter_range' in conf_map:
            self.det_range.setText(str(conf_map['lowpass_filter_range']).replace(" ", ""))


    def fill_active(self, layout):
        """
        This function displays the feature's parameters when the feature becomes active.
        Parameters
        ----------
        layout : Layout widget
            a layout with the feature
        Returns
        -------
        nothing
        """
        self.res_triggers = QLineEdit()
        layout.addRow("low resolution triggers", self.res_triggers)
        self.sigma_range = QLineEdit()
        layout.addRow("sigma range", self.sigma_range)
        self.det_range = QLineEdit()
        layout.addRow("det range", self.det_range)


    def rec_default(self):
        """
        This function sets low resolution feature's parameters to hardcoded default values.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        self.res_triggers.setText('[0, 1, 320]')
        self.sigma_range.setText('[2.0]')
        self.det_range.setText('[.7]')


    def add_feat_conf(self, conf_map):
        """
        This function adds low resolution feature's parameters to dictionary.
        Parameters
        ----------
        conf_map : dict
            contains parameters for reconstruction
        Returns
        -------
        nothing
        """
        if len(self.res_triggers.text()) > 0:
            conf_map['resolution_trigger'] = ast.literal_eval(str(self.res_triggers.text()).replace('\n',''))
        if len(self.sigma_range.text()) > 0:
            conf_map['lowpass_filter_sw_sigma_range'] = ast.literal_eval(str(self.sigma_range.text()).replace('\n',''))
        if len(self.det_range.text()) > 0:
            conf_map['lowpass_filter_range'] = ast.literal_eval(str(self.det_range.text()).replace('\n',''))


class shrink_wrap(Feature):
    """
    This class encapsulates support feature.
    """
    def __init__(self):
        super(shrink_wrap, self).__init__()
        self.id = 'shrink wrap'


    def init_config(self, conf_map):
        """
        This function sets support feature's parameters to parameters in dictionary and displays in the window.
        Parameters
        ----------
        conf_map : dict
            contains parameters for reconstruction
        Returns
        -------
        nothing
        """
        if 'shrink_wrap_trigger' in conf_map:
            triggers = conf_map['shrink_wrap_trigger']
            self.active.setChecked(True)
            self.shrink_wrap_triggers.setText(str(triggers).replace(" ", ""))
        else:
            self.active.setChecked(False)
            return
        if 'shrink_wrap_type' in conf_map:
            self.shrink_wrap_type.setText(str(conf_map['shrink_wrap_type']).replace(" ", ""))
        if 'shrink_wrap_threshold' in conf_map:
            self.shrink_wrap_threshold.setText(str(conf_map['shrink_wrap_threshold']).replace(" ", ""))
        if 'shrink_wrap_gauss_sigma' in conf_map:
            self.shrink_wrap_gauss_sigma.setText(str(conf_map['shrink_wrap_gauss_sigma']).replace(" ", ""))


    def fill_active(self, layout):
        """
        This function displays the feature's parameters when the feature becomes active.
        Parameters
        ----------
        layout : Layout widget
            a layout with the feature
        Returns
        -------
        nothing
        """
        self.shrink_wrap_triggers = QLineEdit()
        layout.addRow("shrink wrap triggers", self.shrink_wrap_triggers)
        self.shrink_wrap_type = QLineEdit()
        layout.addRow("shrink wrap algorithm", self.shrink_wrap_type)
        self.shrink_wrap_threshold = QLineEdit()
        layout.addRow("shrink wrap threshold", self.shrink_wrap_threshold)
        self.shrink_wrap_gauss_sigma = QLineEdit()
        layout.addRow("shrink wrap Gauss sigma", self.shrink_wrap_gauss_sigma)


    def rec_default(self):
        """
        This function sets support feature's parameters to hardcoded default values.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        self.shrink_wrap_triggers.setText('[1,1]')
        self.shrink_wrap_type.setText('GAUSS')
        self.shrink_wrap_gauss_sigma.setText('1.0')
        self.shrink_wrap_threshold.setText('0.1')


    def add_feat_conf(self, conf_map):
        """
        This function adds support feature's parameters to dictionary.
        Parameters
        ----------
        conf_map : dict
            contains parameters for reconstruction
        Returns
        -------
        nothing
        """
        if len(self.shrink_wrap_triggers.text()) > 0:
            conf_map['shrink_wrap_trigger'] = ast.literal_eval(str(self.shrink_wrap_triggers.text()).replace('\n',''))
        if len(self.shrink_wrap_type.text()) > 0:
            conf_map['shrink_wrap_type'] = str(self.shrink_wrap_type.text())
        if len(self.shrink_wrap_threshold.text()) > 0:
            conf_map['shrink_wrap_threshold'] = ast.literal_eval(str(self.shrink_wrap_threshold.text()))
        if len(self.shrink_wrap_gauss_sigma.text()) > 0:
            conf_map['shrink_wrap_gauss_sigma'] = ast.literal_eval(str(self.shrink_wrap_gauss_sigma.text()))


class phase_support(Feature):
    """
    This class encapsulates phase support feature.
    """
    def __init__(self):
        super(phase_support, self).__init__()
        self.id = 'phase support'


    def init_config(self, conf_map):
        """
        This function sets phase support feature's parameters to parameters in dictionary and displays in the window.
        Parameters
        ----------
        conf_map : dict
            contains parameters for reconstruction
        Returns
        -------
        nothing
        """
        if 'phase_support_trigger' in conf_map:
            triggers = conf_map['phase_support_trigger']
            self.active.setChecked(True)
            self.phase_triggers.setText(str(triggers).replace(" ", ""))
        else:
            self.active.setChecked(False)
            return
        if 'phm_phase_min' in conf_map:
            self.phm_phase_min.setText(str(conf_map['phm_phase_min']).replace(" ", ""))
        if 'phm_phase_max' in conf_map:
            self.phm_phase_max.setText(str(conf_map['phm_phase_max']).replace(" ", ""))


    def fill_active(self, layout):
        """
        This function displays the feature's parameters when the feature becomes active.
        Parameters
        ----------
        layout : Layout widget
            a layout with the feature
        Returns
        -------
        nothing
        """
        self.phase_triggers = QLineEdit()
        layout.addRow("phase support triggers", self.phase_triggers)
        self.phm_phase_min = QLineEdit()
        layout.addRow("phase minimum", self.phm_phase_min)
        self.phm_phase_max = QLineEdit()
        layout.addRow("phase maximum", self.phm_phase_max)


    def rec_default(self):
        """
        This function sets phase support feature's parameters to hardcoded default values.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        self.phase_triggers.setText('[0,1,320]')
        self.phm_phase_min.setText('-1.57')
        self.phm_phase_max.setText('1.57')


    def add_feat_conf(self, conf_map):
        """
        This function adds phase support feature's parameters to dictionary.
        Parameters
        ----------
        conf_map : dict
            contains parameters for reconstruction
        Returns
        -------
        nothing
        """
        if len(self.phase_triggers.text()) > 0:
            conf_map['phase_support_trigger'] = ast.literal_eval(str(self.phase_triggers.text()).replace('\n',''))
        if len(self.phm_phase_min.text()) > 0:
            conf_map['phm_phase_min'] = ast.literal_eval(str(self.phm_phase_min.text()))
        if len(self.phm_phase_max.text()) > 0:
            conf_map['phm_phase_max'] = ast.literal_eval(str(self.phm_phase_max.text()))


class pcdi(Feature):
    """
    This class encapsulates pcdi feature.
    """
    def __init__(self):
        super(pcdi, self).__init__()
        self.id = 'pcdi'


    def init_config(self, conf_map):
        """
        This function sets pcdi feature's parameters to parameters in dictionary and displays in the window.
        Parameters
        ----------
        conf_map : dict
            contains parameters for reconstruction
        Returns
        -------
        nothing
        """
        if 'pc_interval' in conf_map:
            self.active.setChecked(True)
            self.pc_interval.setText(str(conf_map['pc_interval']).replace(" ", ""))
        else:
            self.active.setChecked(False)
            return
        if 'pc_type' in conf_map:
            self.pc_type.setText(str(conf_map['pc_type']).replace(" ", ""))
        if 'pc_LUCY_iterations' in conf_map:
            self.pc_iter.setText(str(conf_map['pc_LUCY_iterations']).replace(" ", ""))
        if 'pc_normalize' in conf_map:
            self.pc_normalize.setText(str(conf_map['pc_normalize']).replace(" ", ""))
        if 'pc_LUCY_kernel' in conf_map:
            self.pc_LUCY_kernel.setText(str(conf_map['pc_LUCY_kernel']).replace(" ", ""))


    def fill_active(self, layout):
        """
        This function displays the feature's parameters when the feature becomes active.
        Parameters
        ----------
        layout : Layout widget
            a layout with the feature
        Returns
        -------
        nothing
        """
        self.pc_interval = QLineEdit()
        layout.addRow("pc interval", self.pc_interval)
        self.pc_type = QLineEdit()
        layout.addRow("partial coherence algorithm", self.pc_type)
        self.pc_iter = QLineEdit()
        layout.addRow("LUCY iteration number", self.pc_iter)
        self.pc_normalize = QLineEdit()
        layout.addRow("normalize", self.pc_normalize)
        self.pc_LUCY_kernel = QLineEdit()
        layout.addRow("LUCY kernel area", self.pc_LUCY_kernel)


    def rec_default(self):
        """
        This function sets pcdi feature's parameters to hardcoded default values.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        self.pc_interval.setText('50')
        self.pc_type.setText('LUCY')
        self.pc_iter.setText('20')
        self.pc_normalize.setText('True')
        self.pc_LUCY_kernel.setText('[16, 16, 16]')


    def add_feat_conf(self, conf_map):
        """
        This function adds pcdi feature's parameters to dictionary.
        Parameters
        ----------
        conf_map : dict
            contains parameters for reconstruction
        Returns
        -------
        nothing
        """
        if len(self.pc_interval.text()) > 0:
            conf_map['pc_interval'] = ast.literal_eval(str(self.pc_interval.text()))
        if len(self.pc_type.text()) > 0:
            conf_map['pc_type'] = str(self.pc_type.text())
        if len(self.pc_iter.text()) > 0:
            conf_map['pc_LUCY_iterations'] = ast.literal_eval(str(self.pc_iter.text()))
        pc_normalize_txt = str(self.pc_normalize.text()).strip()
        if pc_normalize_txt == 'False':
            conf_map['pc_normalize'] = False
        else:
            conf_map['pc_normalize'] = True
        if len(self.pc_LUCY_kernel.text()) > 0:
            conf_map['pc_LUCY_kernel'] = ast.literal_eval(str(self.pc_LUCY_kernel.text()).replace('\n',''))


class twin(Feature):
    """
    This class encapsulates twin feature.
    """
    def __init__(self):
        super(twin, self).__init__()
        self.id = 'twin'


    def init_config(self, conf_map):
        """
        This function sets twin feature's parameters to parameters in dictionary and displays in the window.
        Parameters
        ----------
        conf_map : dict
            contains parameters for reconstruction
        Returns
        -------
        nothing
        """
        if 'twin_trigger' in conf_map:
            self.active.setChecked(True)
            self.twin_triggers.setText(str(conf_map['twin_trigger']).replace(" ", ""))
        else:
            self.active.setChecked(False)
            return
        if 'twin_halves' in conf_map:
            self.twin_halves.setText(str(conf_map['twin_halves']).replace(" ", ""))


    def fill_active(self, layout):
        """
        This function displays the feature's parameters when the feature becomes active.
        Parameters
        ----------
        layout : Layout widget
            a layout with the feature
        Returns
        -------
        nothing
        """
        self.twin_triggers = QLineEdit()
        layout.addRow("twin triggers", self.twin_triggers)
        self.twin_halves = QLineEdit()
        layout.addRow("twin halves", self.twin_halves)


    def rec_default(self):
        """
        This function sets twin feature's parameters to hardcoded default values.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        self.twin_triggers.setText('[2]')
        self.twin_halves.setText('[0,0]')


    def add_feat_conf(self, conf_map):
        """
        This function adds twin feature's parameters to dictionary.
        Parameters
        ----------
        conf_map : dict
            contains parameters for reconstruction
        Returns
        -------
        nothing
        """
        if len(self.twin_triggers.text()) > 0:
            conf_map['twin_trigger'] = ast.literal_eval(str(self.twin_triggers.text()).replace('\n',''))
        if len(self.twin_halves.text()) > 0:
            conf_map['twin_halves'] = ast.literal_eval(str(self.twin_halves.text()).replace('\n',''))


class average(Feature):
    """
    This class encapsulates average feature.
    """
    def __init__(self):
        super(average, self).__init__()
        self.id = 'average'


    def init_config(self, conf_map):
        """
        This function sets average feature's parameters to parameters in dictionary and displays in the window.
        Parameters
        ----------
        conf_map : dict
            contains parameters for reconstruction
        Returns
        -------
        nothing
        """
        if 'average_trigger' in conf_map:
            self.active.setChecked(True)
            self.average_triggers.setText(str(conf_map['average_trigger']).replace(" ", ""))
        else:
            self.active.setChecked(False)
            return


    def fill_active(self, layout):
        """
        This function displays the feature's parameters when the feature becomes active.
        Parameters
        ----------
        layout : Layout widget
            a layout with the feature
        Returns
        -------
        nothing
        """
        self.average_triggers = QLineEdit()
        layout.addRow("average triggers", self.average_triggers)


    def rec_default(self):
        """
        This function sets average feature's parameters to hardcoded default values.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        self.average_triggers.setText('[-50,1]')


    def add_feat_conf(self, conf_map):
        """
        This function adds average feature's parameters to dictionary.
        Parameters
        ----------
        conf_map : dict
            contains parameters for reconstruction
        Returns
        -------
        nothing
        """
        conf_map['average_trigger'] = ast.literal_eval(str(self.average_triggers.text()).replace('\n',''))


class progress(Feature):
    """
    This class encapsulates progress feature.
    """
    def __init__(self):
        super(progress, self).__init__()
        self.id = 'progress'


    def init_config(self, conf_map):
        """
        This function sets progress feature's parameters to parameters in dictionary and displays in the window.
        Parameters
        ----------
        conf_map : dict
            contains parameters for reconstruction
        Returns
        -------
        nothing
        """
        if 'progress_trigger' in conf_map:
            self.active.setChecked(True)
            self.progress_triggers.setText(str(conf_map['progress_trigger']).replace(" ", ""))
        else:
            self.active.setChecked(False)
            return


    def fill_active(self, layout):
        """
        This function displays the feature's parameters when the feature becomes active.
        Parameters
        ----------
        layout : Layout widget
            a layout with the feature
        Returns
        -------
        nothing
        """
        self.progress_triggers = QLineEdit()
        layout.addRow("progress triggers", self.progress_triggers)


    def rec_default(self):
        """
        This function sets progress feature's parameters to hardcoded default values.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        self.progress_triggers.setText('[0,20]')


    def add_feat_conf(self, conf_map):
        """
        This function adds progress feature's parameters to dictionary.
        Parameters
        ----------
        conf_map : dict
            contains parameters for reconstruction
        Returns
        -------
        nothing
        """
        conf_map['progress_trigger'] = ast.literal_eval(str(self.progress_triggers.text()).replace('\n',''))


class Features(QWidget):
    """
    This class is composition of all feature classes.
    """
    def __init__(self, tab, layout):
        """
        Constructor, creates all concrete feature objects, and displays in window.
        """
        super(Features, self).__init__()
        feature_ids = ['GA', 'low resolution', 'shrink wrap', 'phase support', 'pcdi', 'twin', 'average', 'progress']
        self.leftlist = QListWidget()
        self.feature_dir = {'GA' : GA(),
                            'low resolution' : low_resolution(),
                            'shrink wrap' : shrink_wrap(),
                            'phase support' : phase_support(),
                            'pcdi' : pcdi(),
                            'twin' : twin(),
                            'average' : average(),
                            'progress' : progress()}
        self.Stack = QStackedWidget(self)
        for i in range(len(feature_ids)):
            id = feature_ids[i]
            self.leftlist.insertItem(i, id)
            feature = self.feature_dir[id]
            feature.stackUI(self.leftlist.item(i), self)
            self.Stack.addWidget(feature.stack)

        layout.addWidget(self.leftlist)
        layout.addWidget(self.Stack)

        self.leftlist.currentRowChanged.connect(self.display)


    def display(self, i):
        self.Stack.setCurrentIndex(i)


def main(args):
    """
    Starts GUI application.
    """
    app = QApplication(args)
    ex = cdi_gui()
    ex.set_args(args)
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main(sys.argv[1:])
