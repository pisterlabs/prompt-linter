import os
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from xrayutilities.io import spec as spec
import convertconfig as conv
import ast
import cohere_core as cohere
import util.util as ut

def get_det_from_spec(specfile, scan):
    """
    Reads detector area and detector name from spec file for given scan.
    Parameters
    ----------
    specfile : str
        spec file name
         
    scan : int
        scan number to use to recover the saved measurements
    Returns
    -------
    detector_name : str
        detector name
    det_area : list
        detector area
    """
    try:
    # Scan numbers start at one but the list is 0 indexed
        ss = spec.SPECFile(specfile)[scan - 1]
    # Stuff from the header
        detector_name = str(ss.getheader_element('UIMDET'))
        det_area = [int(n) for n in ss.getheader_element('UIMR5').split()]
        return detector_name, det_area
    except  Exception as ex:
        print(str(ex))
        print ('Could not parse ' + specfile )
        return None, None


def parse_spec(specfile, scan):
    """
    Reads parameters necessary to run visualization from spec file for given scan.
    Parameters
    ----------
    specfile : str
        spec file name
         
    scan : int
        scan number to use to recover the saved measurements
    Returns
    -------
    delta, gamma, theta, phi, chi, scanmot, scanmot_del, detdist, detector_name, energy
    """
    # Scan numbers start at one but the list is 0 indexed
    specfile = specfile.replace(os.sep, '/')
    try:
        ss = spec.SPECFile(specfile)[scan - 1]
    except  Exception as ex:
        print(str(ex))
        print ('Could not parse ' + specfile )
        return None,None,None,None,None,None,None,None,None,None

    # Stuff from the header
    try:
        detector_name = str(ss.getheader_element('UIMDET'))
    except:
        detector_name = None
    try:
        command = ss.command.split()
        scanmot = command[1]
        scanmot_del = (float(command[3]) - float(command[2])) / int(command[4])
    except:
        scanmot = None
        scanmot_del = None

    # Motor stuff from the header
    try:
        delta = ss.init_motor_pos['INIT_MOPO_Delta']
    except:
        delta = None
    try:
        gamma = ss.init_motor_pos['INIT_MOPO_Gamma']
    except:
        gamma = None
    try:
        theta = ss.init_motor_pos['INIT_MOPO_Theta']
    except:
        theta = None
    try:
        phi = ss.init_motor_pos['INIT_MOPO_Phi']
    except:
        phi = None
    try:
        chi = ss.init_motor_pos['INIT_MOPO_Chi']
    except:
        chi = None
    try:
        detdist = ss.init_motor_pos['INIT_MOPO_camdist']
    except:
        detdist = None
    try:
        energy = ss.init_motor_pos['INIT_MOPO_Energy']
    except:
        energy = None

    # returning the scan motor name as well.  Sometimes we scan things
    # other than theta.  So we need to expand the capability of the display
    # code.
    return delta, gamma, theta, phi, chi, scanmot, scanmot_del, detdist, detector_name, energy


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


def set_overriden(item):
    """
    Helper function that will set the text color to black.
    Parameters
    ----------
    item : widget
    Returns
    -------
    nothing
    """
    item.setStyleSheet('color: black')


class PrepTab(QWidget):
    def __init__(self, parent=None):
        """
        Constructor, initializes the tabs.
        """
        super(PrepTab, self).__init__(parent)
        self.name = 'Prep Data'


    def init(self, tabs, main_window):
        """
        Creates and initializes the 'prep' tab.
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
        scan_layout = QHBoxLayout()
        self.separate_scans = QCheckBox('separate scans')
        self.separate_scans.setChecked(False)
        scan_layout.addWidget(self.separate_scans)
        self.separate_scan_ranges = QCheckBox('separate scan ranges')
        self.separate_scan_ranges.setChecked(False)
        scan_layout.addWidget(self.separate_scan_ranges)
        layout.addRow(scan_layout)
        self.data_dir_button = QPushButton()
        layout.addRow("data directory", self.data_dir_button)
        self.dark_file_button = QPushButton()
        layout.addRow("darkfield file", self.dark_file_button)
        self.white_file_button = QPushButton()
        layout.addRow("whitefield file", self.white_file_button)
        self.roi = QLineEdit()
        self.Imult = QLineEdit()
        layout.addRow("Imult", self.Imult)
        self.detector = QLineEdit()
        layout.addRow("detector", self.detector)
        layout.addRow("detector area (roi)", self.roi)
        self.min_files = QLineEdit()
        layout.addRow("min files in scan", self.min_files)
        self.exclude_scans = QLineEdit()
        layout.addRow("exclude scans", self.exclude_scans)

        cmd_layout = QHBoxLayout()
        self.set_prep_conf_from_button = QPushButton("Load prep conf from")
        self.set_prep_conf_from_button.setStyleSheet("background-color:rgb(205,178,102)")
        self.prep_button = QPushButton('prepare', self)
        self.prep_button.setStyleSheet("background-color:rgb(175,208,156)")
        cmd_layout.addWidget(self.set_prep_conf_from_button)
        cmd_layout.addWidget(self.prep_button)
        layout.addRow(cmd_layout)
        self.setLayout(layout)

        self.prep_button.clicked.connect(self.run_tab)
        self.data_dir_button.clicked.connect(self.set_data_dir)
        self.dark_file_button.clicked.connect(self.set_dark_file)
        self.white_file_button.clicked.connect(self.set_white_file)
        self.set_prep_conf_from_button.clicked.connect(self.load_prep_conf)


    def load_tab(self, load_from, need_convert):
        """
        It verifies given configuration file, reads the parameters, and fills out the window.
        Parameters
        ----------
        conf : str
            configuration file (config_prep)
        Returns
        -------
        nothing
        """
        load_from = load_from.replace(os.sep, '/')
        if os.path.isfile(load_from):
            conf = load_from
        else:
            conf = load_from + '/conf/config_prep'
            if not os.path.isfile(conf):
                msg_window('info: the load directory does not contain config_prep file')
                return
        if need_convert:
            conf_map = conv.get_conf_dict(conf, 'config_prep')
            # if experiment set, save the config_prep
            ut.write_config(conf_map, conf)
        else:
            conf_map = ut.read_config(conf)
            if conf_map is None:
                msg_window('please check configuration file ' + conf )
                return

        self.parse_spec()
        if 'separate_scans' in conf_map:
            separate_scans = conf_map['separate_scans']
            if separate_scans:
                self.separate_scans.setChecked(True)
            else:
                self.separate_scans.setChecked(False)
        else:
            self.separate_scans.setChecked(False)
        if 'separate_scan_ranges' in conf_map:
            separate_scan_ranges = conf_map['separate_scan_ranges']
            if separate_scan_ranges:
                self.separate_scan_ranges.setChecked(True)
            else:
                self.separate_scan_ranges.setChecked(False)
        else:
            self.separate_scan_ranges.setChecked(False)
        # the separate_scan and separate_scan_ranges parameters affects other tab (results_dir in dispaly tab)
        # this tab has to notify observer about the initial setup
        self.notify()
        if 'data_dir' in conf_map:
            if os.path.isdir(conf_map['data_dir']):
                self.data_dir_button.setStyleSheet("Text-align:left")
                self.data_dir_button.setText(conf_map['data_dir'])
            else:
                msg_window('The data_dir directory in config_prep file  ' + conf_map['data_dir'] + ' does not exist')
        else:
            self.data_dir_button.setText('')
        if 'darkfield_filename' in conf_map:
            if os.path.isfile(conf_map['darkfield_filename']):
                self.dark_file_button.setStyleSheet("Text-align:left")
                self.dark_file_button.setText(conf_map['darkfield_filename'])
            else:
                msg_window('The darkfield file ' + conf_map['darkfield_filename'] + ' in config_prep file does not exist')
                self.dark_file_button.setText('')
        else:
            self.dark_file_button.setText('')
        if 'whitefield_filename' in conf_map:
            if os.path.isfile(conf_map['whitefield_filename']):
                self.white_file_button.setStyleSheet("Text-align:left")
                self.white_file_button.setText(conf_map['whitefield_filename'])
            else:
                self.white_file_button.setText('')
                msg_window('The whitefield file ' + conf_map['whitefield_filename'] + ' in config_prep file does not exist')
        else:
            self.white_file_button.setText('')
        if 'Imult' in conf_map:
            self.Imult.setText(str(conf_map['Imult']).replace(" ", ""))
        if 'detector' in conf_map:
            self.detector.setText(str(conf_map['detector']).replace(" ", ""))
            self.detector.setStyleSheet('color: black')
        if 'min_files' in conf_map:
            self.min_files.setText(str(conf_map['min_files']).replace(" ", ""))
        if 'exclude_scans' in conf_map:
            self.exclude_scans.setText(str(conf_map['exclude_scans']).replace(" ", ""))
        if 'roi' in conf_map:
            self.roi.setText(str(conf_map['roi']).replace(" ", ""))
            self.roi.setStyleSheet('color: black')


    def clear_conf(self):
        self.separate_scans.setChecked(False)
        self.separate_scan_ranges.setChecked(False)
        self.data_dir_button.setText('')
        self.dark_file_button.setText('')
        self.white_file_button.setText('')
        self.Imult.setText('')
        self.detector.setText('')
        self.min_files.setText('')
        self.exclude_scans.setText('')
        self.roi.setText('')


    def load_prep_conf(self):
        """
        TODO: combine all load conf files in one function
        It display a select dialog for user to select a configuration file for preparation. When selected, the parameters from that file will be loaded to the window.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        prep_file = select_file(os.getcwd().replace(os.sep, '/')).replace(os.sep, '/')
        if prep_file is not None:
            self.load_tab(prep_file)
        else:
            msg_window('please select valid prep config file')


    def get_prep_config(self):
        """
        It reads parameters related to preparation from the window and adds them to dictionary.
        Parameters
        ----------
        none
        Returns
        -------
        conf_map : dict
            contains parameters read from window
        """
        conf_map = {}
        if len(self.data_dir_button.text().strip()) > 0:
            conf_map['data_dir'] = str(self.data_dir_button.text()).strip()
        if len(self.dark_file_button.text().strip()) > 0:
            conf_map['darkfield_filename'] = str(self.dark_file_button.text().strip())
        if len(self.white_file_button.text().strip()) > 0:
            conf_map['whitefield_filename'] = str(self.white_file_button.text().strip())
        if len(self.Imult.text()) > 0:
            conf_map['Imult'] = ast.literal_eval(str(self.Imult.text()).replace('\n',''))
        if len(self.detector.text()) > 0:
            conf_map['detector'] = str(self.detector.text().strip())
        if self.separate_scans.isChecked():
            conf_map['separate_scans'] = True
        if self.separate_scan_ranges.isChecked():
            conf_map['separate_scan_ranges'] = True
        if len(self.min_files.text()) > 0:
            min_files = ast.literal_eval(str(self.min_files.text()))
            conf_map['min_files'] = min_files
        if len(self.exclude_scans.text()) > 0:
            conf_map['exclude_scans'] = ast.literal_eval(str(self.exclude_scans.text()).replace('\n',''))
        if len(self.roi.text()) > 0:
            conf_map['roi'] = ast.literal_eval(str(self.roi.text()).replace('\n',''))

        return conf_map


    def run_tab(self):
        """
        Reads the parameters needed by prep script. Saves the config_prep configuration file with parameters from
        the window and runs the prep script.

        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        if not self.main_win.is_exp_exists():
            msg_window('the experiment has not been created yet')
            return
        elif not self.main_win.is_exp_set():
            msg_window('the experiment has changed, press "set experiment" button')
            return
        else:
            conf_map = self.get_prep_config()
        # verify that prep configuration is ok
        er_msg = cohere.verify('config_prep', conf_map)
        if len(er_msg) > 0:
            msg_window(er_msg)
            return
        # for 34idc prep data directory is needed
        if len(self.data_dir_button.text().strip()) == 0:
            msg_window('cannot prepare data for 34idc, need data directory')
            return
        scan = str(self.main_win.scan_widget.text())
        if len(scan) == 0:
            msg_window('cannot prepare data for 34idc, scan not specified')

        ut.write_config(conf_map, self.main_win.experiment_dir + '/conf/config_prep')
        # the separate_scan and separate_scan_ranges parameters affects other tab (results_dir in dispaly tab)
        # this tab has to notify observer about the initial setup
        self.notify()

        if len(self.main_win.scan_widget.text()) == 0:
            msg_window('cannot prepare data for 34idc, scan not specified')
        else:
            self.tabs.run_prep()


    def set_dark_file(self):
        """
        It display a select dialog for user to select a darkfield file.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        darkfield_filename = select_file(os.getcwd().replace(os.sep, '/')).replace(os.sep, '/')
        if darkfield_filename is not None:
            self.dark_file_button.setStyleSheet("Text-align:left")
            self.dark_file_button.setText(darkfield_filename)
        else:
            self.dark_file_button.setText('')


    def set_white_file(self):
        """
        It display a select dialog for user to select a whitefield file.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        whitefield_filename = select_file(os.getcwd().replace(os.sep, '/')).replace(os.sep, '/')
        if whitefield_filename is not None:
            self.white_file_button.setStyleSheet("Text-align:left")
            self.white_file_button.setText(whitefield_filename)
        else:
            self.white_file_button.setText('')


    def set_data_dir(self):
        """
        It display a select dialog for user to select a directory with raw data file.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        data_dir = select_dir(os.getcwd().replace(os.sep, '/')).replace(os.sep, '/')
        if data_dir is not None:
            self.data_dir_button.setStyleSheet("Text-align:left")
            self.data_dir_button.setText(data_dir)
        else:
            self.data_dir_button.setText('')


    def save_conf(self):
        conf_map = self.get_prep_config()
        if len(conf_map) > 0:
            er_msg = cohere.verify('config_prep', conf_map)
            if len(er_msg) > 0:
                msg_window(er_msg)
            else:
                ut.write_config(conf_map, self.main_win.experiment_dir + '/conf/config_prep')


    def parse_spec(self):
        """
        Calls utility function to parse spec file. Displas the parsed parameters in the window with blue text.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        if self.main_win.specfile is None:
            return
        if not self.main_win.is_exp_exists():
            # do not parse on initial assignment
            return
        scan = str(self.main_win.scan_widget.text())
        if len(scan) == 0:
            msg_window('scan number is needed to parse spec file')
        else:
            try:
                last_scan = int(scan.split('-')[-1].split(',')[-1])
                detector_name, roi = get_det_from_spec(self.main_win.specfile, last_scan)
                self.roi.setText(str(roi))
                self.roi.setStyleSheet('color: blue')

                if detector_name is not None:
                    self.detector.setText(str(detector_name)[:-1])
                    self.detector.setStyleSheet('color: blue')
            except Exception as e:
                print(str(e))
                msg_window ('error parsing spec')


    def update_tab(self, **args):
        """
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        if 'specfile' in args:
            self.parse_spec()


    def notify(self):
        self.tabs.notify(**{'separate_scans':self.separate_scans.isChecked(), 'separate_scan_ranges':self.separate_scan_ranges.isChecked()})


class DispTab(QWidget):
    def __init__(self, parent=None):
        """
        Constructor, initializes the tabs.
        """
        super(DispTab, self).__init__(parent)
        self.name = 'Display'


    def init(self, tabs, main_window):
        """
        Creates and initializes the 'disp' tab.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        self.tabs = tabs
        self.main_win = main_window

        self.results_dir = None

        layout = QFormLayout()
        self.result_dir_button = QPushButton()
        layout.addRow("phasing results directory", self.result_dir_button)
        self.make_twin = QCheckBox('make twin')
        self.make_twin.setChecked(False)
        layout.addWidget(self.make_twin)
        self.diffractometer = QLineEdit()
        layout.addRow("diffractometer", self.diffractometer)
        self.crop = QLineEdit()
        layout.addRow("crop", self.crop)
        self.rampups = QLineEdit()
        layout.addRow("ramp upscale", self.rampups)
        self.energy = QLineEdit()
        layout.addRow("energy", self.energy)
        self.delta = QLineEdit()
        layout.addRow("delta (deg)", self.delta)
        self.gamma = QLineEdit()
        layout.addRow("gamma (deg)", self.gamma)
        self.detdist = QLineEdit()
        layout.addRow("detdist (mm)", self.detdist)
        self.theta = QLineEdit()
        layout.addRow("theta (deg)", self.theta)
        self.chi = QLineEdit()
        layout.addRow("chi (deg)", self.chi)
        self.phi = QLineEdit()
        layout.addRow("phi (deg)", self.phi)
        self.scanmot = QLineEdit()
        layout.addRow("scan motor", self.scanmot)
        self.scanmot_del = QLineEdit()
        layout.addRow("scan motor delta", self.scanmot_del)
        self.detector = QLineEdit()
        layout.addRow("detector", self.detector)
        cmd_layout = QHBoxLayout()
        self.set_disp_conf_from_button = QPushButton("Load disp conf from")
        self.set_disp_conf_from_button.setStyleSheet("background-color:rgb(205,178,102)")
        self.config_disp = QPushButton('process display', self)
        self.config_disp.setStyleSheet("background-color:rgb(175,208,156)")
        cmd_layout.addWidget(self.set_disp_conf_from_button)
        cmd_layout.addWidget(self.config_disp)
        layout.addRow(cmd_layout)
        self.setLayout(layout)

        self.result_dir_button.clicked.connect(self.set_res_dir)
        self.config_disp.clicked.connect(self.run_tab)
        self.energy.textChanged.connect(lambda: set_overriden(self.energy))
        self.delta.textChanged.connect(lambda: set_overriden(self.delta))
        self.gamma.textChanged.connect(lambda: set_overriden(self.gamma))
        self.detdist.textChanged.connect(lambda: set_overriden(self.detdist))
        self.theta.textChanged.connect(lambda: set_overriden(self.theta))
        self.chi.textChanged.connect(lambda: set_overriden(self.chi))
        self.phi.textChanged.connect(lambda: set_overriden(self.phi))
        self.scanmot.textChanged.connect(lambda: set_overriden(self.scanmot))
        self.scanmot_del.textChanged.connect(lambda: set_overriden(self.scanmot_del))
        self.detector.textChanged.connect(lambda: set_overriden(self.detector))
        self.set_disp_conf_from_button.clicked.connect(self.load_disp_conf)
        self.layout4 = layout


    def load_tab(self, load_from, need_convert):
        """
        It verifies given configuration file, reads the parameters, and fills out the window.
        Parameters
        ----------
        conf : str
            configuration file (config_disp)
        Returns
        -------
        nothing
        """
        load_from = load_from.replace(os.sep, '/')
        if os.path.isfile(load_from):
            conf = load_from
            conf_dir = os.path.dirname(os.path.abspath(conf).replace(os.sep, '/'))
        else:
            conf_dir = load_from + '/conf'
            conf = conf_dir + '/config_disp'
            if not os.path.isfile(conf):
                msg_window('info: the load directory does not contain config_disp file')
                return
        if need_convert:
            conf_map = conv.get_conf_dict(conf, 'config_disp')
            # if experiment set, save the config_disp
            ut.write_config(conf_map, self.main_win.experiment_dir + '/conf/config_disp')
        else:
            conf_map = ut.read_config(conf)
            if conf_map is None:
                msg_window('please check configuration file ' + conf)
                return

        self.parse_spec()
        self.results_dir = self.main_win.experiment_dir

        self.result_dir_button.setStyleSheet("Text-align:left")
        self.result_dir_button.setText(self.results_dir)
        # if parameters are configured, override the readings from spec file
        if 'make_twin' in conf_map:
            make_twin = conf_map['make_twin']
            if make_twin:
                self.make_twin.setChecked(True)
            else:
                self.make_twin.setChecked(False)
        else:
            self.make_twin.setChecked(False)

        if 'diffractometer' in conf_map:
            self.diffractometer.setText(str(conf_map['diffractometer']).replace(" ", ""))
        if 'crop' in conf_map:
            self.crop.setText(str(conf_map['crop']).replace(" ", ""))
        if 'rampups' in conf_map:
            self.rampups.setText(str(conf_map['rampups']).replace(" ", ""))
        if 'energy' in conf_map:
            self.energy.setText(str(conf_map['energy']).replace(" ", ""))
            self.energy.setStyleSheet('color: black')
        if 'delta' in conf_map:
            self.delta.setText(str(conf_map['delta']).replace(" ", ""))
            self.delta.setStyleSheet('color: black')
        if 'gamma' in conf_map:
            self.gamma.setText(str(conf_map['gamma']).replace(" ", ""))
            self.gamma.setStyleSheet('color: black')
        if 'detdist' in conf_map:
            self.detdist.setText(str(conf_map['detdist']).replace(" ", ""))
            self.detdist.setStyleSheet('color: black')
        if 'theta' in conf_map:
            self.theta.setText(str(conf_map['theta']).replace(" ", ""))
            self.theta.setStyleSheet('color: black')
        if 'chi' in conf_map:
            self.chi.setText(str(conf_map['chi']).replace(" ", ""))
            self.chi.setStyleSheet('color: black')
        if 'phi' in conf_map:
            self.phi.setText(str(conf_map['phi']).replace(" ", ""))
            self.phi.setStyleSheet('color: black')
        if 'scanmot' in conf_map:
            self.scanmot.setText(str(conf_map['scanmot']).replace(" ", ""))
            self.scanmot.setStyleSheet('color: black')
        if 'scanmot_del' in conf_map:
            self.scanmot_del.setText(str(conf_map['scanmot_del']).replace(" ", ""))
            self.scanmot_del.setStyleSheet('color: black')
        if 'detector' in conf_map:
            self.detector.setText(str(conf_map['detector']).replace(" ", ""))
            self.detector.setStyleSheet('color: black')


    def clear_conf(self):
        self.make_twin.setChecked(False)
        self.diffractometer.setText('')
        self.crop.setText('')
        self.rampups.setText('')
        self.energy.setText('')
        self.delta.setText('')
        self.gamma.setText('')
        self.detdist.setText('')
        self.theta.setText('')
        self.chi.setText('')
        self.phi.setText('')
        self.scanmot.setText('')
        self.scanmot_del.setText('')
        self.detector.setText('')


    def load_disp_conf(self):
        """
        It display a select dialog for user to select a configuration file. When selected, the parameters
        from that file will be loaded to the window.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        disp_file = select_file(os.getcwd().replace(os.sep, '/')).replace(os.sep, '/')
        if disp_file is not None:
            self.load_tab(disp_file)
        else:
            msg_window('please select valid disp config file')


    def get_disp_config(self):
        """
        It reads parameters related to visualization from the window and adds them to dictionary.
        Parameters
        ----------
        none
        Returns
        -------
        conf_map : dict
            contains parameters read from window
        """
        conf_map = {}
        if self.results_dir is not None:
            conf_map['results_dir'] = self.results_dir.replace(os.sep, '/')
        if self.make_twin.isChecked():
            conf_map['make_twin'] = True
        if len(self.energy.text()) > 0:
            conf_map['energy'] = ast.literal_eval(str(self.energy.text()))
        if len(self.delta.text()) > 0:
            conf_map['delta'] = ast.literal_eval(str(self.delta.text()))
        if len(self.gamma.text()) > 0:
            conf_map['gamma'] = ast.literal_eval(str(self.gamma.text()))
        if len(self.detdist.text()) > 0:
            conf_map['detdist'] = ast.literal_eval(str(self.detdist.text()))
        if len(self.theta.text()) > 0:
            conf_map['theta'] = ast.literal_eval(str(self.theta.text()))
        if len(self.chi.text()) > 0:
            conf_map['chi'] = ast.literal_eval(str(self.chi.text()))
        if len(self.phi.text()) > 0:
            conf_map['phi'] = ast.literal_eval(str(self.phi.text()))
        if len(self.scanmot.text()) > 0:
            conf_map['scanmot'] = str(self.scanmot.text())
        if len(self.scanmot_del.text()) > 0:
            conf_map['scanmot_del'] = ast.literal_eval(str(self.scanmot_del.text()))
        if len(self.detector.text()) > 0:
            conf_map['detector'] = str(self.detector.text())
        if len(self.diffractometer.text()) > 0:
            conf_map['diffractometer'] = str(self.diffractometer.text())
        if len(self.crop.text()) > 0:
            conf_map['crop'] = ast.literal_eval(str(self.crop.text()).replace('\n', ''))
        if len(self.rampups.text()) > 0:
            conf_map['rampups'] = ast.literal_eval(str(self.rampups.text()).replace('\n', ''))

        return conf_map


    def run_tab(self):
        """
        Reads the parameters needed by format display script. Saves the config_disp configuration file with parameters from the window and runs the display script.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        if not self.main_win.is_exp_exists():
            msg_window('the experiment has not been created yet')
            return
        if not self.main_win.is_exp_set():
            msg_window('the experiment has changed, pres "set experiment" button')
            return
        # check if the results exist
        if self.results_dir is None:
            self.results_dir = self.main_win.experiment_dir
        else:
            self.results_dir = self.results_dir.replace(os.sep, '/')
#        if 'image.npy' in glob.glob1(self.results_dir, recursive=True):
        found_file = False
        for p, d, f in os.walk(self.results_dir):
            if 'image.npy' in f:
                found_file = True
                break
        if not found_file:
            msg_window('No image files found in the results directory tree. Please, run reconstruction in previous tab to activate this function')
            return
        if (self.main_win.specfile is None or not os.path.isfile(self.main_win.specfile)) and \
           (len(self.energy.text()) == 0 or \
            len(self.delta.text()) == 0 or \
            len(self.gamma.text()) == 0 or \
            len(self.detdist.text()) == 0 or \
            len(self.theta.text()) == 0 or \
            len(self.detector.text()) == 0):
                msg_window('Please, enter valid spec file or all detector parameters')
                return

        conf_map = self.get_disp_config()
        # verify that disp configuration is ok
        er_msg = cohere.verify('config_disp', conf_map)
        if len(er_msg) > 0:
            msg_window(er_msg)
            return

        ut.write_config(conf_map, self.main_win.experiment_dir + '/conf/config_disp')
        self.tabs.run_viz()


    def save_conf(self):
        conf_map = self.get_disp_config()
        if len(conf_map) > 0:
            er_msg = cohere.verify('config_disp', conf_map)
            if len(er_msg) > 0:
                msg_window(er_msg)
            else:
                ut.write_config(conf_map, self.main_win.experiment_dir + '/conf/config_disp')


    def parse_spec(self):
        """
        Calls utility function to parse spec file. Displas the parsed parameters in the window with blue text.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        if self.main_win.specfile is None:
            return
        if not self.main_win.is_exp_exists():
            # do not parse on initial assignment
            return
        scan = str(self.main_win.scan_widget.text())
        if len(scan) == 0:
            msg_window('scan number is needed to parse spec file')
        else:
            try:
                last_scan = int(scan.split('-')[-1].split(',')[-1])
                delta, gamma, theta, phi, chi, scanmot, scanmot_del, detdist, detector_name, energy = parse_spec(self.main_win.specfile, last_scan)
                if energy is not None:
                    self.energy.setText(str(energy))
                    self.energy.setStyleSheet('color: blue')
                if delta is not None:
                    self.delta.setText(str(delta))
                    self.delta.setStyleSheet('color: blue')
                if gamma is not None:
                    self.gamma.setText(str(gamma))
                    self.gamma.setStyleSheet('color: blue')
                if theta is not None:
                    self.theta.setText(str(theta))
                    self.theta.setStyleSheet('color: blue')
                if chi is not None:
                    self.chi.setText(str(chi))
                    self.chi.setStyleSheet('color: blue')
                if phi is not None:
                    self.phi.setText(str(phi))
                    self.phi.setStyleSheet('color: blue')
                if detdist is not None:
                    self.detdist.setText(str(detdist))
                    self.detdist.setStyleSheet('color: blue')
                if scanmot is not None:
                    self.scanmot.setText(str(scanmot))
                    self.scanmot.setStyleSheet('color: blue')
                if scanmot_del is not None:
                    self.scanmot_del.setText(str(scanmot_del))
                    self.scanmot_del.setStyleSheet('color: blue')
                if detector_name is not None:
                    self.detector.setText(str(detector_name)[:-1])
                    self.detector.setStyleSheet('color: blue')
            except Exception as e:
                print(str(e))
                msg_window ('error parsing spec')


    def update_tab(self, **args):
        """
        Results directory is a parameter in display tab. It defines a directory tree that the display script will
        search for reconstructed image files and will process them for visualization. This function initializes it in
        typical situation to experiment directory. In case of active genetic algorithm it will be initialized to the
        generation directory with best results, and in case of alternate reconstruction configuration, it will be
        initialized to the last directory where the results were saved.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        if 'specfile' in args:
            self.parse_spec()
        if 'separate_scans' in args or 'separate_scan_ranges' in args:
            if 'separate_scans' in args:
                separate_scans = args['separate_scans']
            if 'separate_scan_ranges' in args:
                separate_scan_ranges = args['separate_scan_ranges']
            if separate_scans or separate_scan_ranges:
                self.results_dir = self.main_win.experiment_dir
            else:
                self.results_dir = self.main_win.experiment_dir + '/results_phasing'
            self.result_dir_button.setText(self.results_dir)
            self.result_dir_button.setStyleSheet("Text-align:left")
            return

        if 'rec_id' in args:
            rec_id = args['rec_id']
            if len(rec_id) > 0:
                self.results_dir = self.main_win.experiment_dir + '/results_phasing_' + rec_id
            else:
                self.results_dir = self.main_win.experiment_dir + '/results_phasing'

        if 'generations' in args:
            generations = args['generations']
            if 'rec_no' in args:
                rec_no = args['rec_no']
            else:
                rec_no = 1
            if generations > 0:
                if rec_no > 1:
                    self.results_dir = self.results_dir + '/g_' + str(generations - 1) + '/0'
                else:
                    self.results_dir = self.results_dir + '/g_' + str(generations - 1)

        self.result_dir_button.setText(self.results_dir)
        self.result_dir_button.setStyleSheet("Text-align:left")


    def set_res_dir(self):
        """
        Results directory is a parameter in display tab. It defines a directory tree that the display script will
        search for reconstructed image files and will process them for visualization. This function displays the
        dialog selection window for the user to select the results directory.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        if self.main_win.is_exp_exists():
            self.results_dir = self.main_win.experiment_dir + '/results_phasing'
            self.results_dir = select_dir(self.results_dir).replace(os.sep, '/')
            if self.results_dir is not None:
                self.result_dir_button.setStyleSheet("Text-align:left")
                self.result_dir_button.setText(self.results_dir)
            else:
                self.result_dir_button.setText('')
        else:
            msg_window('the experiment has not been created yet')
