# TODO: fix vsteps for different numbers of electrodes
# font sizes are different on ylabels
from __future__ import division
import sys, os, copy, traceback
import distutils.sysconfig

import pygtk
pygtk.require("2.0")
import gtk
from gtk import gdk

from scipy import arange, sin, pi, zeros, ones, reshape, \
     greater_equal, transpose, array, arange, resize, \
     absolute, nonzero
from scipy import fromstring, arange, log10
from scipy import minimum, maximum

from matplotlib.cbook import exception_to_str
from pbrainlib.gtkutils import str2num_or_err, simple_msg, error_msg, \
     not_implemented, yes_or_no, FileManager, select_name, get_num_range, Dialog_FileSelection, Dialog_FileChooser, get_num_value

from matplotlib.widgets import Cursor, SpanSelector

from data import EEGWeb, EEGFileSystem, EOI, Amp, Grids
from file_formats import FileFormat_BNI, W18Header, FileFormat_AxonAscii, FileFormat_NeuroscanAscii, FileFormat_AlphaomegaAscii, NeuroscanEpochFile

from dialogs import Dialog_Preferences, Dialog_SelectElectrodes,\
     Dialog_CohstatExport, Dialog_SaveEOI, Dialog_EEGParams, \
     Dialog_Annotate, Dialog_AnnBrowser, \
     Dialog_PhaseSynchrony, Dialog_PhaseSynchronyPlot, \
     AutoPlayDialog, SpecProps, Dialog_EventRelatedSpec

from dialog_filterelectrodes import Dialog_FilterElectrodes

import datetime

import servers
from borgs import Shared
from events import Observer
from shared import fmanager, eegviewrc
from gladewrapper import PrefixWrapper
from utils import filter_grand_mean
from coh_explorer import CohExplorer

from matplotlib import rcParams

from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg as FigureCanvas
import matplotlib.cm as cm
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.transforms import BboxTransform, Bbox, ScaledTranslation, blended_transform_factory #in all, removed unit_bbox, Value, Point, and
#replaced get_bbox_transform with BboxTransform, added ScaledTranslation and blended_transform_factory

from matplotlib.patches import Rectangle

from scipy.signal import buttord, butter, lfilter

from mpl_windows import ChannelWin, AcorrWin, HistogramWin, SpecWin, EventRelatedSpecWin

major, minor1, minor2, s, tmp = sys.version_info
if major<2 or (major==2 and minor1<3):
    True = 1
    False = 0

def load_w18(fullpath):
    assert(os.path.exists(fullpath))
    basename, filename = os.path.split(fullpath)
    fh = file(fullpath, 'rb')
         
    header = W18Header(fh)
    params = {
        'filename'        : filename,
        'date'            : header.currtime,
        'description'     : '',
        'channels'        : 18,
        'freq'            : 200,
        'classification'  : 99,
        'file_type'       : W18,
        'behavior_state'  : 99,
        }

    eeg = EEGFileSystem(fullpath, params)
    return eeg

def load_bmsi(bnipath):

    bni = FileFormat_BNI(bnipath)
    basename, ext = os.path.splitext(bnipath)
    
    if os.path.exists(basename):    
        fullpath = basename
    elif os.path.exists(basename + '.eeg'):
        fullpath = basename + '.eeg'
    else:
        fullpath = fmanager.get_filename(
            title='Select EEG File accompanying this BNI file')

    eeg = bni.get_eeg(fullpath)
    return eeg


def load_epoch(fname):
    epoch = NeuroscanEpochFile(fname)
    return epoch.eeg

def load_params(path):
    params = {}
    for line in file(path):
        line = line.strip()
        if not len(line): continue
        if line.startswith('#'): continue

        k,v = line.split(':',1)
        k = k.strip()
        v = v.strip()
        if k in ('channels', 'pid', 'freq', 'classification', 'file_type', 'behavior_state') :
            v = int(v)
        params[k] = v

    eegfile = params['eegfile']
    if not os.path.exists(eegfile):
        error_msg('Cannot find eeg file "%s"'%eegfile)
        return

    eeg = EEGFileSystem(eegfile, params)
    return eeg


def load_axonascii(path):
    axonascii = FileFormat_AxonAscii(path)
    return axonascii.eeg

def load_alphaomegaascii(path):
    alphaomegascii = FileFormat_AlphaomegaAscii(path)
    return alphaomegascii.eeg

def load_neuroscanascii(path):
    try:
        neuroscanascii = FileFormat_NeuroscanAscii(path)
    except IOError, msg:
        print "load_neuroscanascii(): msg=", msg
        error_msg(msg, title='Error', parent=parent)
                    
    return neuroscanascii.eeg

extmap = { '.w18' : load_w18,
           '.bni' : load_bmsi,
           '.params' : load_params,
           '.epoch' : load_epoch,
           '.axonascii' : load_axonascii,
           '.neuroscanascii' : load_neuroscanascii,
           '.alphaomegaascii' : load_alphaomegaascii
           }

class EEGNavBar(gtk.Toolbar, Observer):
    """
    CLASS: EEGNavBar
    DESCR: toolbar for MainWindow
    """

    def add_toolbutton(self, icon_name, tip_text, tip_private, clicked_function, clicked_param1=None):
        iconSize = gtk.ICON_SIZE_SMALL_TOOLBAR
        iconw = gtk.Image()
        iconw.set_from_stock(icon_name, iconSize)
        
        toolitem = gtk.ToolButton()
        toolitem.set_icon_widget(iconw)
        toolitem.show_all()
        #updated for new tooltip api
        toolitem.set_tooltip_text(tip_text)
        #toolitem.set_tooltip(self.tooltips, tip_text, tip_private)
        toolitem.connect("clicked", clicked_function, clicked_param1)
        toolitem.connect("scroll_event", clicked_function)
        self.insert(toolitem, -1)

    def add_toolitem(self, widget, tip_text):
        toolitem = gtk.ToolItem()
        toolitem.add(widget)
        toolitem.show_all()
        self.insert(toolitem, -1)
        
    def add_separator(self):
        toolitem = gtk.SeparatorToolItem()
        toolitem.set_draw(True)
        #toolitem.set_expand(gtk.TRUE)
        toolitem.show_all()
        self.insert(toolitem, -1)
        
    
    def __init__(self, eegplot=None, win=None):
        """
        eegplot is the EEGPlot instance that the toolboar controls

        win, if not None, is the gtk.Window the Figure is embedded in
        
        """
        gtk.Toolbar.__init__(self)
        Observer.__init__(self)
        self.win = win
        self.eegplot = eegplot
        
        iconSize = gtk.ICON_SIZE_SMALL_TOOLBAR
        self.set_border_width(5)
        self.set_style(gtk.TOOLBAR_ICONS)
        
        self.tooltips = gtk.Tooltip()

        self.add_toolbutton(gtk.STOCK_GOTO_FIRST, 'Move back one page', 'Private', self.panx, -10)
        self.add_toolbutton(gtk.STOCK_GO_BACK, 'Move back in time', 'Private', self.panx, -1)
        self.add_toolbutton(gtk.STOCK_GO_FORWARD, 'Move forward in time', 'Private', self.panx, 1)
        self.add_toolbutton(gtk.STOCK_GOTO_LAST, 'Move forward one page', 'Private', self.panx, 10)

        self.add_separator()

        self.add_toolbutton(gtk.STOCK_ZOOM_IN, 'Shrink the time axis', 'Private', self.zoomx, 1)
        self.add_toolbutton(gtk.STOCK_ZOOM_OUT, 'Expand the time axis', 'Private', self.zoomx, 0)

        self.add_separator()

        self.add_toolbutton(gtk.STOCK_GO_UP, 'Increase the voltage gain', 'Private', self.zoomy, 1)
        self.add_toolbutton(gtk.STOCK_GO_DOWN, 'Decrease the voltage gain', 'Private', self.zoomy, 0)
        
        self.add_toolbutton(gtk.STOCK_REDO, 'Specify time range', 'Private', self.specify_range)
        #self.add_toolbutton(gtk.STOCK_REDO, 'Specify the voltage gain', 'Private', self.specify_range_time)
        #the above was not important enough to keep right now -eli
        self.add_toolbutton(gtk.STOCK_JUMP_TO, 'Automatically page the EEG', 'Private', self.auto_play)
        self.add_toolbutton(gtk.STOCK_SAVE, 'Save the figure', 'Private', self.save_figure)
        
        
        self.add_toolbutton(gtk.STOCK_JUMP_TO, 'Open Coh Explr', 'Private', self.load_cohexplr)
        self.add_separator()

        def toggled(button):
            self.broadcast(Observer.GMTOGGLED, button)

        def lock_trode_toggled(button) :
            self.broadcast(Observer.LOCK_TRODE_TOGGLED, button)

        

        self.buttonGM = gtk.CheckButton('GM')
        self.buttonGM.show()
        self.buttonGM.connect('toggled', toggled)
        self.buttonGM.set_active(True)
        self.buttonGM.set_active(False)
        self.add_toolitem(self.buttonGM, 'Remove grand mean from data if checked')
        #self.append_widget(
        #    self.buttonGM, 'Remove grand mean from data if checked', '')
        
        self.buttonLockTrode = gtk.CheckButton('Lock')
        self.buttonLockTrode.show()
        self.buttonLockTrode.connect('toggled', lock_trode_toggled)
        self.add_toolitem(self.buttonLockTrode, 'Lock Selected Electrode')
        #self.append_widget(
        #    self.buttonLockTrode, 'Lock Selected Electrode', '')
        
        self.add_separator()
        #adding a decimate toggle here, without an icon
        toolitem = gtk.ToolButton()
        toolitem.show_all()
        toolitem.set_label("Dec.")
        toolitem.connect("clicked", self.set_decimate, None)
        self.insert(toolitem, -1)
        #/decimate toggle

    def set_decimate(self, *args):
        decfactor = self.eegplot.find_decimate_factor()
        dec_input = get_num_value(labelStr='Choose a decimation factor (1 for none)', title='Enter value', parent=None,
                  default=decfactor) #the default value will be the optimal dec factor
        if dec_input < self.eegplot.decimateFactor:
            self.eegplot.decimateFactor = copy.deepcopy(dec_input)
            self.eegplot.set_time_lim(updateData=True, broadcast=True)
            return
        else:
            self.eegplot.decimateFactor = dec_input
            self.eegplot.plot()
            self.eegplot.draw()

    def auto_play(self, *args):

        tmin, tmax = self.eegplot.get_time_lim()
        twidth = tmax-tmin

        dlg = AutoPlayDialog(0, self.eegplot.eeg.get_tmax(), twidth)
        dlg.show()
        
    def load_cohexplr(self, *args):
        self.eegplot.load_cohexplr()
            
    def specify_range(self, *args):

        response = get_num_range()
        if response is None: return

        tmin, tmax = response
        
        self.eegplot.set_time_lim(tmin, tmax, updateData=False)
        self.eegplot.plot()
        self.eegplot.draw()

    def save_figure(self, button):
                
        def print_ok(button):
            fname = fs.get_filename()
            fmanager.set_lastdir(fname)
            fs.destroy()
            try: self.eegplot.canvas.print_figure(fname)
            except IOError, msg:                
                err = '\n'.join(map(str, msg))
                msg = 'Failed to save %s: Error msg was\n\n%s' % (
                    fname, err)
                try: parent = Shared.windowMain.widget
                except AttributeError: parent = None
                simple_msg(msg, title='Error', parent=parent)

        fs = gtk.FileSelection(title='Save the figure')
        if self.win is not None:
            fs.set_transient_for(self.win)
        fs.set_filename(fmanager.get_lastdir() + os.sep)

        fs.ok_button.connect("clicked", print_ok)
        fs.cancel_button.connect("clicked", lambda b: fs.destroy())
        fs.show()

    def set_eegplot(self, eegplot):
        self.eegplot = eegplot
        
    def panx(self, button, arg):

        if self.eegplot is None: return 
        try: arg.direction
        except AttributeError: right = arg
        else:
            if arg.direction == gdk.SCROLL_UP: right=1
            else: right=0
	
        self.eegplot.pan_time(right)
        self.eegplot.plot() #redraw the traces -eli
        self.eegplot.draw()
        return False

    def zoomx(self, button, arg):
        if self.eegplot is None: return 
        try: arg.direction
        except AttributeError: direction = arg
        else:            
            if arg.direction == gdk.SCROLL_UP: direction=1
            else: direction=0

        self.eegplot.change_time_gain(direction)
        self.eegplot.plot() #redraw the traces -eli
        self.eegplot.draw()
        return False

    def zoomy(self, button, arg):
        if self.eegplot is None: return 
        try: arg.direction
        except AttributeError: direction = arg
        else:
            if arg.direction == gdk.SCROLL_UP: direction=1
            else: direction=0

        self.eegplot.change_volt_gain(direction)
        self.eegplot.plot() #redraw the traces -eli
        self.eegplot.draw()
        return False

class EEGPlot(Observer):
    """
    CLASS: EEGPlot
    DESCR: controls MainWindow's canvas
    """
    timeSets = ((1.,.1), (2.,.2), (5.,.5), (10.,1.), (20.,2.),
                (50.,5.), (100., 10.), (200., 20.))

    voltSets = (.1, .2, .5,  .75, 1., 2., 5., 7.5,
                10., 20., 50., 75., 100., 200., 500., 750,
                1000., 1250., 1500. , 1750., 2000., 2100., 2300., 2500., 3000., 3500., 4000., 4500., 5000., 7500.,
                10000., 20000., 50000., 75000., 150000., 300000.)

    colorOrder = ('b','k','g','c','m')

    def __init__(self, eeg, canvas):
        Observer.__init__(self)
        eeg.load_data()
        self.canvas = canvas
        self.figure = canvas.figure
        self.axes = self.figure.axes[0]
        self.axes.cla()
        self.eeg = eeg
        self.cnumDict = self.eeg.get_amp().get_channel_num_dict()

        #self.annman = AnnotationManager(self)

        amp = eeg.get_amp()
        eoi = amp.to_eoi()
        
        self.colord = {}
        colorInd = 0

        for gname, gnum in eoi:
            gname = gname.lower()
            color = self.colord.get(gname.lower())
            if color is None:
                color = self.colorOrder[colorInd % len(self.colorOrder)]
                self.colord[gname] = color
                colorInd += 1
        
        self._selected = eoi[0]
        self.set_eoi(eoi)

        self.timeInd = 3
        self.voltInd = 27
        self.maxLabels = 36
        
        self.decimateFactor = 1 #this is set when toggled by the user
        self.filterGM = Shared.windowMain.toolbar.buttonGM.get_active()
        # mcc XXX: turning off cache
        #self._selectedCache = None, None

        # Lock the selected electrode.
        self.lock_trode = False

        # Create a vertical cursor.
        self.cursor = Cursor(self.axes, useblit=True, linewidth=1, color='red')
        if eegviewrc.horizcursor == 'True' :
            self.cursor.horizOn = True
        else :
            self.cursor.horizOn = False
        if eegviewrc.vertcursor == 'True' :
            self.cursor.vertOn = True
        else :
            self.cursor.vertOn = False

        # mcc XXX: map for whether or not to rectify/DC offset/lowpass filter a given (e.g. EMG) channel
        #self.rectifyChannels = Set()

    def get_color(self, trode):
        gname, gnum = trode
        gname = gname.lower()
        return self.colord[gname]

    def recieve(self, event, *args):

        #if event in (Observer.SET_TIME_LIM,):
        #    tmin, tmax = args
        #    print "EEGVIEW.EEGVIEW.recieve: set_time_lim"
        #    self.set_time_lim(tmin, tmax, updateData=False, broadcast=False)
        #    self.plot()
        #    self.draw()

        if event==Observer.SAVE_FRAME:
            fname = args[0] + '.png'
            width, height = self.canvas.get_width_height()
            # matplotlib needs to have get_pixmap() (in backends/FigureCanvasGTKAgg)
            pixmap = self.canvas._pixmap
            pixbuf = gdk.Pixbuf(gdk.COLORSPACE_RGB, 0, 8,
                                width, height)
            pixbuf.get_from_drawable(pixmap, pixmap.get_colormap(),
                                     0, 0, 0, 0, width, height)
        
            pixbuf.save(fname, 'png')
            try:
                Shared.windowMain.update_status_bar(
                    'Saved frame: %s' % fname)
            except AttributeError: pass
        elif event == Observer.SELECT_CHANNEL:
            trode = args[0]
            gname, gnum = trode
            self.set_selected((gname, gnum))
        elif event == Observer.GMTOGGLED:
            button = args[0]
            self.filterGM = button.get_active()
            tmin, tmax = self.get_time_lim()
            t, data, freq = self.filter(tmin, tmax)        

            for ind, line in zip(self.indices, self.lines):
                line.set_data(t, data[:,ind])
            self.draw()
        elif event == Observer.LOCK_TRODE_TOGGLED :
            button = args[0]
            self.lock_trode = button.get_active()
            
    def draw(self):
        self.canvas.draw()

    def load_cohexplr(self):
        ce = CohExplorer(self.eoi, self.eeg.freq)
        ce.show()

    def get_selected(self, filtergm=False):
        'return t, data[ind], trode'
        print "EEGPlot.get_selected()"
        tmin, tmax = self.get_time_lim()

        key = (tmin, tmax, self._selected, filtergm)

        #keycache, retcache = self._selectedCache
        #if keycache==key: return retcache
        
        t, data = self.eeg.get_data(tmin, tmax)
        # mccXXX : why does this line exist?
        data = -data

        if filtergm:
            data = filter_grand_mean(data)

        ind = self.eoiIndDict[self._selected]

        print "EEGPlot.get_selected(): data.shape is ", data.shape, " and we are about to index it like data[:,%d]" % self.indices[ind]
        ret = t, data[:,self.indices[ind]], self._selected
        #self._selectedCache = key, ret
        return ret
        
    def get_selected_window(self, filtergm=False, extraTime=0):
        'return t, data[ind], trode'
        tmin, tmax = self.get_time_lim()
        print "get_selected_window: ", tmin, tmax
        # XXX mcc, taking this out for neuroscanascii format which doesn't handle negative vals well
        #tmin -= extraTime/2.
        #tmax += extraTime/2.

        #key = (tmin, tmax, self._selected, filtergm)
        #keycache, retcache = self._selectedCache
        #if keycache==key: return retcache

        print "get_selected_window(tmin=",tmin,"tmax=",tmax,")"
        t, data = self.eeg.get_data(tmin, tmax)
        # mcc XXX : why does this line exist?
        #data = -data

        if filtergm:
            print "EEGPlot.get_selected(): filtering grand mean"
            data = filter_grand_mean(data)

        ind = self.eoiIndDict[self._selected]

        ret = t, data[:,self.indices[ind]], self._selected
        #self._selectedCache = key, ret
        return ret
        
        
    def get_eoi(self):
        # XXX mcc: we want to return a copy here, because otherwise view3 can
        # remove our EOIs!!
        #return list(self.eoi)
        return self.eoi

    def set_eoi(self, eoi):
        print "eegview.set_eoi(",eoi,")"
        try:
            #print self.eeg.get_amp()
            self.indices = eoi.to_data_indices(self.eeg.get_amp())
        except KeyError:
            msg = exception_to_str('Could not get amplifier indices for EOI')
            try: parent = Shared.windowMain.widget
            except AttributeError: parent = None
            error_msg(msg, title='Error', parent=parent)
            return 0

        self.eoi = eoi
        self.eoiIndDict = dict([ (trode, i) for i, trode in enumerate(self.eoi)])

        if not self.eoiIndDict.has_key(self._selected):
            self._selected = self.eoi[0]

        # Remove annotation rects, so they will get redrawn on the next 
        # update_annotations()
        #self.annman.remove_rects()
            
        return True
        
    def get_eeg(self):
        return self.eeg
    

    def find_decimate_factor(self, lpcf = 40):
        print "EEGPlot.find_decimate_factor(): calculating decimation factor"
        print "EEGPlot.find_decimate_factor(): eeg.freq is ", self.eeg.freq
        Nyq = self.eeg.freq/2
       
        self.decimateFactor = int(Nyq/lpcf) #a decimation factor has to be an integer as it turns out-eli
        if self.decimateFactor == 0:
            self.decimateFactor = 1 #take care of dividebyzero errors - this shouldn't happen anyway when Nyq is high enough (ie when freq is high enough ~500)
        print "EEGPlot.find_decimate_factor: ", self.decimateFactor
        return self.decimateFactor

    def filter(self, tmin, tmax, lpcf=40, lpsf=55, hpcf=None, hpsf=None):
        """
        lpcf: low pass corner freq=40 (Hz)
        lpsf: low pass stop freq=55 (Hz)
        hpcf: high pass corner freq=None
        hpsf: high pass stop freq=None
        a lowpass decimate filter first uses a lowpass to smooth out the data, and then takes chunks out of it according to the decimation factor
        in order to speed up processing. Here we use a butterworth lowpass - this was here before I got here, but I think there must be simpler 
        options -eli
        """
        print "\n========\nEEGPlot.filter(%f, %f, ...)" % (tmin, tmax)

        try: t, data = self.eeg.get_data(tmin, tmax)
        
        except KeyError, msg:
            msg = exception_to_str('Could not get data')
            error_msg(exception_to_str('Could not get data'))
            return None

        #data = -data  # invert neg up #why?

        if self.filterGM:
            data = filter_grand_mean(data)
        Nyq = self.eeg.freq/2
        #as of now we do a lowpass filter regardless of whether the decimation factor is > 1. -eli
        Rp, Rs = 2, 20
        
        Wp = lpcf/Nyq
        Ws = lpsf/Nyq

        [n,Wn] = buttord(Wp,Ws,Rp,Rs)
        print "EEGPlot.filter(): [n,Wn] = buttord(Wp= ", Wp, ",Ws=", Ws, ",Rp=", Rp, ",Rs=", Rs, ") = [", n, "," , Wn, "]"
        [b,a] = butter(n,Wn)
        print "EEGPlot.filter(): [b,a] = butter(n=" , n , " , Wn=", Wn, ") = [", b, ",", a, "]" 
        print "EEGPlot.filter(): doing transpose(lfilter(b,a,transpose(data)))"

        data = transpose( lfilter(b,a,transpose(data)))

        decfreq = self.eeg.freq/self.decimateFactor
        self.decfreq = decfreq

        #print "EEGPlot.filter(): decimateFactor  = int(Nyq=%f/lpcf=%d) = " % (Nyq, lpcf), decimateFactor, "self.decfreq=(eeg.freq=%f)/(%d) = " %     (self.eeg.freq, decimateFactor), self.decfreq
        #are all of the above commented lines really not needed anymore? -eli
        print "EEGPlot.filter(): returning decimated data t[::%d], data[::%d], %f" % (self.decimateFactor, self.decimateFactor, decfreq)
        return t[::self.decimateFactor], data[::self.decimateFactor], decfreq #the "::" takes every decimateFactorth value from each array!


    def plot(self):
        print "EEGPlot.plot()"
        
        self.axes.cla()
        tmin, tmax = self.get_time_lim() #it turns out hardcoding 0,10 in this function was ahem counterproductive -eli 
        #print "EEGPLOT.plot(): tmn, tmax ", tmin, tmax        
        
        #let's take out filtering for some tests
        #t, data, freq = self.filter(tmin, tmax)
        try: t, data = self.eeg.get_data(tmin, tmax)
        except KeyError, msg:
            msg = exception_to_str('Could not get data')
            error_msg(exception_to_str('Could not get data'))
            return None
        freq = self.eeg.freq    
        
        #print "EEGplot filtertest: ", data[0:10] 

        dt = 1/freq

        self.lines = []

        skip = max(1, len(self.indices)//self.maxLabels)
        count = 0
        amp = self.eeg.get_amp()
        labels = []
        locs = []


        maxo = 0.975
        mino = 0.025

        N = len(self.indices)
        offsets = 1.0-((maxo-mino)/N*arange(N) + mino)
        self.offsets = offsets
        
        vset = self.voltSets[self.voltInd]
        
        #old transformation block
        """
        boxin = Bbox(
            Point(self.axes.viewLim.ll().x(), Value(-vset)),
            Point(self.axes.viewLim.ur().x(), Value(vset)))



        boxout = Bbox(
            Point(self.axes.bbox.ll().x(), Value(-72)),
            Point(self.axes.bbox.ur().x(), Value(72)))


        transOffset = get_bbox_transform(
            unit_bbox(),
            Bbox( Point( Value(0), self.axes.bbox.ll().y()),
                  Point( Value(1), self.axes.bbox.ur().y())
                  ))
        """ 
        #new transformation block
        #updated by removing Point and Value methods and simply passing four points #to Bbox() this may be a bad idea... I tried passing them to Bbox.set_points#() but this method seems to be either not working or badly documented.
        #also, viewLim is deprecated from what I can tell, so I'll try to use axes.g#et_xlim
        viewLimX=self.axes.get_xlim() #this returns a list of min and max x points, which is what we want to pass below
	    #print "************", viewLimX        
        boxin = Bbox(
            [[viewLimX[0], -vset], #replaced self.axes.viewLim.ll().x() with viewLimX
            [viewLimX[1], vset]])

	    #does this work? yes! there actually is a bbox living in axes, for whatever reason, and this method returns all four points as an array of the form [[x0,y0],[x1,y1]]. the bbox that we rebuild below is (hopefully!) taking the x values of the two points.
        axesBboxCoords = self.axes.bbox.get_points()
        boxout = Bbox(
            [[axesBboxCoords[0][0], -72], #see comment above: I replaced self.axes.bbox.ll().x() with axesBboxCoords[0][0]
            [axesBboxCoords[1][0], 72]])


        transOffset = BboxTransform(
            Bbox.unit(), # ([[0,0], [1,1]]), #replaced unit_bbox with unit()
            Bbox(  [[0, axesBboxCoords[0][1]],
                   [1, axesBboxCoords[1][1]]]
                  ))
        
        assert len(self.indices) == len(offsets), 'indices and offsets have different length'
        pairs = zip(self.indices, offsets)

        labeld = amp.get_dataind_dict()

        for ind, offset in pairs:
            trode = labeld[ind]	    
            color = self.get_color(trode)
            if self._selected==trode: color='r'
            trans = BboxTransform(boxin, boxout) #switched to BboxTransform
	    #print "EEGPlot.plot(): " , data.shape, ind, len(pairs), 			self.eeg.channels
	    #set_offset is way deprecated. I'm going to use a tip from the 			newer transforms_tutorial on the matplotlib.sourceforge page.
	    #the basic idea is to use ScaledTranslation, which creates an 			offset that can then be added to the original trans.
	    
	    #trans.set_offset((0, offset), transOffset)
	    #so, these two lines below which I've written seem to work at 			offsetting the lines to where they need to go. -eli 
	    #note: for some reason, in nipy pbrain the original 		trans.set_offset was written _after_ the call to Line2D
	    newtrans = ScaledTranslation(0,offset,transOffset) 
	    trans = trans + newtrans
	    thisLine = Line2D(t, data[:,ind],
                              color=color,
                              linewidth=0.75,
                              linestyle='-',
                              clip_on=True #added this kwarg
                              )
            thisLine.set_transform(trans)
            #thisLine.set_data_clipping(False) #deprecated
               
	    #should the following be commented out?
	    #thisLine.set_lod(on=1)
            self.lines.append(thisLine)
            self.axes.add_line(thisLine)
	    if count % skip == 0:                
                labels.append('%s%d' % trode)
                locs.append(offset)
            count += 1
            #print 'locs', labels[0], locs[0], self.offsets[0]

        #self.set_time_lim(tmin,tmax, updateData=False) #i fixed this and then realized it was reduntant anyway -eli

        self.axes.set_yticks(locs)            

        labels = self.axes.set_yticklabels(labels, fontsize=8)

        for tick in self.axes.yaxis.get_major_ticks():
            tick.label1.set_transform(self.axes.transAxes)
            tick.label2.set_transform(self.axes.transAxes)
            tick.tick1line.set_transform(self.axes.transAxes)
            tick.tick2line.set_transform(self.axes.transAxes)
            tick.gridline.set_transform(self.axes.transAxes)            
        print "EEGPlot.plot(): successful"
        # Update annotation boxes
        #self.annman.update_annotations()

        self.save_excursion()
        self.draw()

    # XXX: mcc: what is this for ?
    def restore_excursion(self):
        try: self.saveExcursion
        except AttributeError: return
        tmin, self.timeInd, self.voltInd = self.saveExcursion 
        self.set_time_lim(tmin, updateData=True)
        

    def save_excursion(self):
        tmin, tmax = self.get_time_lim()
        self.saveExcursion = (tmin, self.timeInd, self.voltInd)
        

    def get_max_labels(self):
        return 25
    

    def change_time_gain(self, magnify=1):
        """Change the time scale.  zoom out with magnify=0, zoom in
        with magnify=1)"""

        # keep the index in bounds
        if magnify and self.timeInd>0:
            self.timeInd -= 1
            
        if not magnify and self.timeInd<(len(self.timeSets)-1):    
            self.timeInd += 1

        origmin, origmax = self.get_time_lim()
        wid, step = self.timeSets[self.timeInd]

        xmin = origmin
        xmax = origmin+wid
        
        self.set_time_lim(xmin, xmax, updateData=False)

    def change_volt_gain(self, magnify=1):
	#note: I had to seriously take this function apart further down. -eli
        """Change the voltage scale.  zoom out with magnify=0, zoom in
        with magnify=1)"""
        #print "EEGPlot.change_volt_gain: magnify=%d, self.voltInd=%d" % (magnify, self.voltInd)

        # keep the index in bounds
        if magnify and self.voltInd>0:
            self.voltInd -= 1
            
        if not magnify and self.voltInd<(len(self.voltSets)-1):    
            self.voltInd += 1

        #print "new self.voltInd=%d" % self.voltInd

        vset = self.voltSets[self.voltInd]

        #print "vset = self.voltSets[%d]" % self.voltInd
	#note: matplotlib had no way of getting at the constructors of a compositeaffine2d object. This use to be done with the get_bbox1 method, but of course this is deprecated and not replaced by ANYTHING. So, using python's built-in hacky __dict__ method below, I extract the in bbox, change the y values to the aboveset vset values, and because python is only using symbolic links and not copying, this does the trick. I am not John and I do not wish to contribute to matplotlib at this time (actually I do but I have other things to do!!) but I wish someone would fix this and then tell me. -eli 
        for line in self.lines:
            trans = line.get_transform()
	    boxin = trans.__dict__['_a'].__dict__['_boxin'].__dict__['_points_orig']
	    #print boxin
	    x0 = boxin[0][0]
	    x1 = boxin[1][0]
	    y0 = -vset
	    y1 = vset
	    boxin = Bbox(
		[[x0,y0],
		[x1,y1]])
	    #print boxin
	    #box1 =  trans.get_bbox1()
            #print "calling line.get_transform().get_bbox1().intervaly().set_bounds(-vset, vset)", box1
            #boxin.intervaly().set_bounds(-vset, vset)

        #print "end of EEGPlot.change_volt_gain()"


    def pan_time(self, right=1):
        """Pan the time axis to the right or left"""

        # keep the index in bounds
        wid, step = self.get_twid_step()
        tmin, tmax = self.get_time_lim()
        #print "pan_time tmin,tmax: ", tmin, tmax        
        step *= right
        #print "pan_time step: ", step
        self.set_time_lim(tmin+step)
        #self.plot() #update the plot! eli

    def get_time_lim(self,):
        return self.axes.get_xlim()


    def get_twid_step(self):
        #print "get_twid_step(): ", self.timeSets[self.timeInd]
        return self.timeSets[self.timeInd] #still not sure exactly why we return twice in this function -eli
        ticks = self.axes.get_xticks()
        wid = ticks[-1] - ticks[0]
        step = ticks[1] - ticks[0]
        #print "get_twid_step(): ", wid, step
        return wid, step
        
    def set_time_lim(self, xmin=None, xmax=None,
                     updateData=False, broadcast=True):
        #make sure xmin keeps some eeg on the screen
        print "EEGPLOT.set_time_lim broadcast=", broadcast, " update data=",updateData
        print "EEGPlot.set_time_lim(xmin=", xmin, "xmax=", xmax, ")"
        
        
        origmin, origmax = self.get_time_lim()
        #print "EEGPlot.set_time_lim(): origmin, origmax = ", origmin, origmax
        if xmin is None: xmin = origmin
        
        if xmax is None:
            wid, step = self.get_twid_step()
            xmax = xmin+wid
        else:
            wid = xmax-xmin
            step = wid/10.0

        print "EEGPlot.set_time_lim(): axes.set_xlim(", [xmin, xmax], ")"
        self.axes.set_xlim([xmin, xmax])
        ticks = arange(xmin, xmax+0.001, step)
        print "EEGPlot.set_time_lim(): axes.set_xticks(", ticks, ")"
        self.axes.set_xticks(ticks)
        def fmt(val):
            if val==int(val): return '%d' % val
            else: return '%1.1f' % val
        #self.axes.set_xticklabels([fmt(val) for val in ticks])
        self.axes.set_xticklabels([])

        
        if updateData:
            print "EEGPlot.set_time_lim(): update data"
            
            # let's take out filtering for some tests
            try: t, data = self.eeg.get_data(xmin, xmax)
            except KeyError, msg:
                msg = exception_to_str('Could not get data')
                error_msg(exception_to_str('Could not get data'))
                return None
            freq = self.eeg.freq    
            
            
            #t, data, freq = self.filter(xmin, xmax)        
            self.axes.set_xlim((xmin, xmax))
            for ind, line in zip(self.indices, self.lines):
                line.set_data(t, data[:,ind])
            self.plot()
            #we'll let the observer take care of this
            #self.axesSpec.set_xlim([xmin,xmax])
            #self.axesSpec.set_xticklabels(ticks)
            
        # recieve the observers
        if broadcast:
            print "EEGPLOT: Broadcasting set time lim"
            self.broadcast(Observer.SET_TIME_LIM, xmin, xmax)

    def get_channel_at_point(self, x, y, select=True):
        "Get the EEG with the voltage trace nearest to x, y (window coords)"

        # avoid a pygtk queue handling error
        if not hasattr(self, 'decfreq'):
            return None
        tmin, tmax = self.get_time_lim()
        dt = 1/self.decfreq

        t, yt = self.axes.transData.inverted().transform( (x,y) ) #replaced inverse_xy_tup with inverted().transform()

        ind = int((t-tmin)/dt)

        ys = zeros( (len(self.lines), ), 'h')

        xdata = self.lines[0].get_xdata()
        if ind>=len(xdata): return None
        thisx = xdata[ind]
        for i, line in enumerate(self.lines):
            thisy = line.get_ydata()[ind]
            trans = line.get_transform()
            xt, yt = trans.transform((thisx, thisy)) #replaced xy_tup with transform
            ys[i] = yt

        ys = absolute(ys-y)
        matches = nonzero(ys==min(ys))

        ind = matches[0]
        labeld = self.eeg.amp.get_dataind_dict()
        # XXX: had to change this for some reason with latest scipy/numpy -- mcc
        trode = labeld[self.indices[ind[0]]]
        #trode = labeld[self.indices[ind]]
        gname, gnum = trode
        if select :
            ok = self.set_selected((gname, gnum))
            if ok: self.broadcast(Observer.SELECT_CHANNEL, trode)
        return trode

    def set_selected(self, trode):
        
        
        lastind = self.eoiIndDict[self._selected]
        ind = self.eoiIndDict[trode]

        lastcolor = self.get_color(self._selected)
        self.lines[lastind].set_color(lastcolor)

        
        self._selected = trode
        self.lines[ind].set_color('r')

        self.canvas.draw()
        Shared.windowMain.update_status_bar('Selected %s %d' % trode)

        return True


class SpecPlot(Observer):
    """
    CLASS: SpecPlot
    DESCR: spectrogram
    """
    propdlg = SpecProps()
    flim = 0, 40    # the defauly yaxis
    clim = None     # the colormap limits

    def __init__(self, axes, canvas, eegplot):
        Observer.__init__(self)
        self.axes = axes
        self.canvas = canvas
        self.eegplot = eegplot
        self.cmap = cm.jet
        # min and max power

    def make_spec(self, *args):
        NFFT, Noverlap = (512, 477)

        selected = self.eegplot.get_selected_window(extraTime=float(NFFT)/float(self.eegplot.eeg.freq))
        #selected = self.eegplot.get_selected()
        print "SpecPlot.make_spec(): selected = ", selected
        if selected is None:
            self.axes.cla()
            t = self.axes.text(
                0.5, 0.5,
                'Click on EEG channel for spectrogram (scroll mouse to expand)',
                verticalalignment='center',
                horizontalalignment='center',
                )
            t.set_transform(self.axes.transAxes)
            xmin, xmax = self.eegplot.get_time_lim()
            self.axes.set_xlim( [xmin, xmax] )
            self.axes.set_xticks( self.eegplot.axes.get_xticks()  )
            return
        flim = SpecPlot.flim
        clim = SpecPlot.clim

        torig, data, trode = selected
        gname, gnum = trode
        label = '%s %d' % (gname, gnum)
        Fs = self.eegplot.eeg.freq

        self.axes.cla()
        xmin, xmax = self.eegplot.get_time_lim()
        xextent = xmin, xmax
        print "make spec: xmin, xmax: ", xmin, xmax
        #try:
        #print "SpecPlot.make_spec(): calling specgram(data=", data.shape, "NFFT=%d, Fs=%d, noverlap=%d, xextent=" % (NFFT, Fs, Noverlap), xextent, ")"
        Pxx, freqs, t, im = self.axes.specgram(
            data, NFFT=NFFT, Fs=Fs, noverlap=Noverlap,
            cmap=self.cmap, xextent=xextent)
        #print "SpecPlot.make_spec(): Pxx.shape is", Pxx.shape, "t is", t
        #except OverflowError, overflowerror:
        #    print "caught overflow error!! bailing: ", overflowerror
        #    f = file("make_spec-%d-%f-%f.overflow.pickle" % (gnum, xmin, xmax), "w")
        #    pickle.dump(data, f)
        #    f.close()
        #    return

            
        if clim is not None:
            im.set_clim(clim[0], clim[1])

        t = t + min(torig)

        Z = 10*log10(Pxx)
        #print "type(Z) is" , type(Z)
        #I fixed this using numpy's min and max but this should work too -eli
        self.pmin = minimum.reduce(minimum.reduce(Z))
        self.pmax = maximum.reduce(maximum.reduce(Z))
        
        #self.eegplot.set_time_lim(xmin=None, xmax=None,
        #             updateData=False, broadcast=False)
        #self.axes.set_xlim( [xmin, xmax] )
        #self.axes.set_xticks( self.eegplot.axes.get_xticks()  )
        print "SpecPlot.make_spec: xticks = ", self.eegplot.axes.get_xticks()
        #self.axes.set_title('Spectrogram for electrode %s' % label)
        #self.axes.set_xlabel('TIME (s)')
        self.axes.set_ylabel('FREQUENCY (Hz)')
        self.axes.set_ylim(flim)

        if flim[1]-flim[0]>=100:
            self.axes.set_yticks(arange(flim[0], flim[1]+1, 20))
        else:
            self.axes.set_yticks(arange(flim[0], flim[1]+1, 10))

    def recieve(self, event, *args):
	#note: this gets called on a timescale update -eli 
        if event in (Observer.SELECT_CHANNEL, Observer.SET_TIME_LIM):
            self.make_spec()
            self.canvas.draw()

    def set_properties(self, *args):
        dlg = SpecPlot.propdlg
        dlg.show()
        if not len(dlg.entryCMin.get_text()) and hasattr(self, 'pmin'):
            dlg.entryCMin.set_text('%1.2f'%self.pmin)
        if not len(dlg.entryCMax.get_text()) and hasattr(self, 'pmax'):
            dlg.entryCMax.set_text('%1.2f'%self.pmax)
            
        while 1:
            response = dlg.run()

            if response in  (gtk.RESPONSE_OK, gtk.RESPONSE_APPLY):
                b = dlg.validate()
                if not b: continue
                SpecPlot.flim = dlg.get_flim()
                SpecPlot.clim = dlg.get_clim()
                self.make_spec()
                self.canvas.draw()
                if response==gtk.RESPONSE_OK:
                    dlg.hide()
                    break
            else:
                dlg.hide()
                break

class MainWindow(PrefixWrapper):
    """
    CLASS: MainWindow
    DESCR: represents XML'd widget tree and other dynamic GUI elements
    """
    prefix = ''
    widgetName = 'windowMain'
    gladeFile = 'main.glade'
    win = None
    def __init__(self):
        if os.path.exists(self.gladeFile):
            #print "opening %s" % self.gladeFile
            theFile=self.gladeFile
        elif os.path.exists(os.path.join('gui', self.gladeFile)):
            #print "opening %s" % os.path.join('gui', self.gladeFile)
            theFile=os.path.join('gui', self.gladeFile)
        else:
            #print "opening %s" % os.path.join(distutils.sysconfig.PREFIX,
            #    'share', 'pbrain', self.gladeFile)

            theFile = os.path.join(
                distutils.sysconfig.PREFIX,
                'share', 'pbrain', self.gladeFile)
            print "MainWindow.__init__(): uhh the file is " , theFile
        
        try: Shared.widgets = gtk.glade.XML(theFile)
        except:
            raise RuntimeError('Could not load glade file %s' % theFile)
        
        PrefixWrapper.__init__(self)
        self._isConfigured = False
        self.patient = None

        figsize = eegviewrc.figsize
        self.fig = Figure(figsize=figsize, dpi=72)
	
        self.canvas = FigureCanvas(self.fig)  # a gtk.DrawingArea
        self.canvas.set_size_request(800, 640)

        self.canvas.connect("scroll_event", self.scroll_event)
        self.canvas.show()

        #self.fig = Figure(figsize=(7,5), dpi=72)
        t = arange(0.0,50.0, 0.01)
        xlim = array([0,10])

        self.axes = self.fig.add_axes([0.075, 0.25, 0.9, 0.725], axisbg='#FFFFCC')

        self.axes.plot(t, sin(2*0.32*pi*t) * sin(2*2.44*pi*t) )
        self.axes.set_xlim([0.0,10.0])
        self.axes.set_xticklabels([])

        self.axesSpec = self.fig.add_axes([0.075, 0.05, 0.9, 0.2])
        t = self.axesSpec.text(
            0.5, 0.5,
            'Click on EEG channel for spectrogram (scroll mouse to expand)',
            verticalalignment='center',
            horizontalalignment='center',
            )
        t.set_transform(self.axes.transAxes)
        self.axesSpec.set_xlim([0.0,10.0])
        self.axesSpec.set_xticklabels([])
        self.axesSpec.set_yticklabels([])
        
        self.win = self['windowMain']
        self.win.move(0,0)

        self['vboxMain'].pack_start(self.canvas, True, True)
        self['vboxMain'].show()
        
        self.toolbar = EEGNavBar( self.canvas, self['windowMain'])
        self.toolbar.show()
        self['vboxMain'].pack_start(self.toolbar, False, False)

        self.statbar = gtk.Statusbar()
        self.statbar.show()
        self.statbarCID = self.statbar.get_context_id('my stat bar')
        self['vboxMain'].pack_start(self.statbar, False, False)
        self.update_status_bar('')
        self.buttonDown = None
        fsize = self.fig.get_size_inches()
        self.fsize = copy.deepcopy(fsize)
	
        self.canvas.mpl_connect('motion_notify_event', self.motion_notify_event)
        self.canvas.mpl_connect('button_press_event', self.button_press_event)
        self.canvas.mpl_connect('button_release_event', self.button_release_event)


    def update_status_bar(self, msg):
        self.statbar.pop(self.statbarCID) 
        mid = self.statbar.push(self.statbarCID, 'Message: ' + msg)

    def menu_select_eeg(self, eeg):
        amp = eeg.get_amp()
        if amp.message is not None:
            simple_msg(amp.message, title='Warning',
                       parent=Shared.windowMain.widget)

        try: self.eegplot
        except AttributeError: pass
        else: Observer.observers.remove(self.eegplot)        

        try: self.specPlot
        except AttributeError: pass
        else: Observer.observers.remove(self.specPlot)        

        self.eegplot = EEGPlot(eeg, self.canvas)
        self.toolbar.set_eegplot(self.eegplot)
        self.specPlot = SpecPlot(self.axesSpec, self.canvas, self.eegplot)
        self.specMenu = self.make_spec_menu()
        eois = eeg.get_associated_files(atype=5, mapped=1)
        self.eoiMenu = self.make_context_menu(eois)
        self.eegplot.plot()
        return False

    def make_patients_menu(self):
        entries = servers.sql.eeg.select(
            where='file_type in (1,4)')
        eegMap = {}
        for entry in entries:
            eegMap.setdefault(entry.pid,[]).append(EEGWeb(entry.get_orig_map()))

        pidList = ','.join(map(str,eegMap.keys()))

        # make a list of eegs and patients so we can pass an index to
        # the callback
        menuItemPatients = self['menuitemPatients']
        menuPatients = gtk.Menu()
        patients = servers.sql.patients.select(
            where='pid in (%s) ORDER BY last' % pidList)
        for patient in patients:
            if not eegMap.has_key(patient.pid): continue

            menuItemPatient = gtk.MenuItem(
                '%s%s' % (patient.first[:2], patient.last[:2]))
            menuItemPatient.show()

            menuEEGs = gtk.Menu()
            for eeg in eegMap[patient.pid]:
                eegLabel = eeg.filename.replace('_', '-')
                item = gtk.MenuItem(label=eegLabel)
                item.show()
                eeg.patient = patient
                item.connect_object(
                    "activate", self.menu_select_eeg, eeg)
                menuEEGs.append(item)
            menuItemPatient.set_submenu(menuEEGs)
            menuPatients.append(menuItemPatient)
        menuItemPatients.set_submenu(menuPatients)

    def load_eoi(self, eoi):
        success = self.eegplot.set_eoi(eoi)
        
        if success:
            tmin, tmax = self.eegplot.get_time_lim()
            self.eegplot.plot()
            self.eegplot.set_time_lim(tmin, tmax, updateData=True)
            self.eegplot.draw()
        else:
            #TODO: popup edit window for eoi
            pass
        
    def new_eoi(self, menuitem):
        self.edit_eoi()

    def make_context_menu(self, eois):
        contextMenu = gtk.Menu()

        label = "Load EOI"
        menuItemLoad = gtk.MenuItem(label)
        contextMenu.append(menuItemLoad)
        menuItemLoad.show()

        menuEOIS = gtk.Menu()
        for eoi in eois:
            eoiLabel = eoi.filename.replace('_', '-')
            item = gtk.MenuItem(label=eoiLabel)
            item.show()
            item.connect_object(
                "activate", self.load_eoi, eoi)
            menuEOIS.append(item)
        menuItemLoad.set_submenu(menuEOIS)

        label = "Save EOI"
        menuItemSave = gtk.MenuItem(label)
        contextMenu.append(menuItemSave)
        menuItemSave.connect("activate", self.save_eoi, 0)
        menuItemSave.show()

        label = "Save As EOI"
        menuItemSaveAs = gtk.MenuItem(label)
        contextMenu.append(menuItemSaveAs)
        menuItemSaveAs.connect("activate", self.save_eoi, 1)
        menuItemSaveAs.show()

        label = "Edit EOI"
        menuItemEdit = gtk.MenuItem(label)
        contextMenu.append(menuItemEdit)
        menuItemEdit.connect("activate", self.edit_eoi)
        menuItemEdit.show()

        label = "New EOI"
        menuItemNew = gtk.MenuItem(label)
        contextMenu.append(menuItemNew)
        menuItemNew.connect("activate", self.new_eoi)
        menuItemNew.show()

        menuItemSep = gtk.MenuItem()
        contextMenu.append(menuItemSep)
        menuItemSep.show()
        menuItemSep = gtk.MenuItem()
        contextMenu.append(menuItemSep)
        menuItemSep.show()

        label = "Edit Channel Filter"
        menuItemEdit = gtk.MenuItem(label)
        menuItemEdit.connect("activate", self.edit_filter)
        menuItemEdit.show()
        contextMenu.append(menuItemEdit)
       
        return contextMenu

    def make_spec_menu(self):
        contextMenu = gtk.Menu()

        label = "Set limits"
        menuItemSave = gtk.MenuItem(label)
        contextMenu.append(menuItemSave)
        menuItemSave.connect("activate", self.specPlot.set_properties, 0)
        menuItemSave.show()
        return contextMenu

    def edit_eoi(self, *args):
        def ok_callback(eoi):
            success = self.eegplot.set_eoi(eoi)
            if success:
                tmin, tmax = self.eegplot.get_time_lim()
                self.eegplot.plot()
                self.eegplot.set_time_lim(tmin, tmax,updateData=True)
                self.eegplot.draw()

            d.destroy_dialog()
            return
        
        eoiActive = self.eegplot.get_eoi()
        eoiAll = self.eegplot.get_eeg().get_amp().to_eoi()
        d = Dialog_SelectElectrodes(trodes=eoiAll,
                                    ok_callback=ok_callback,
                                    selected=eoiActive
                                    )
        d.set_transient_for(self.widget)

    def edit_filter(self, *args):
        """
        This brings up the prefiltering window, which allows one to rectify/hilbert-xform the data
        before sending it to external mpl_windows.
        """
        def ok_callback(filters):
            print "in MainWindow.edit_filter.ok_callback(): filters=", filters

            rectifiedChannels = {}
            hilbertedChannels = {}
            for channel, params in filters.iteritems():
                print "filter f is ", channel, params['rectify']
                rectifiedChannels[channel]= params['rectify']
                hilbertedChannels[channel]= params['hilbert']

            self.eegplot.get_eeg().set_rectified(rectifiedChannels)
            self.eegplot.get_eeg().set_hilberted(hilbertedChannels)
            
            tmin, tmax = self.eegplot.get_time_lim()
            self.eegplot.plot()
            self.eegplot.set_time_lim(tmin, tmax,updateData=True)
            self.eegplot.draw()
            
            d.destroy_dialog()
            return
        
        eoiActive = self.eegplot.get_eoi()
        #print "eoiActive is " , eoiActive
        eoiAll = self.eegplot.get_eeg().get_amp().to_eoi()
        #print "eoiAll is ", eoiAll

        rectify_selected = self.eegplot.get_eeg().get_rectified()
        hilbert_selected = self.eegplot.get_eeg().get_hilberted()
        
        d = Dialog_FilterElectrodes(trodes=eoiActive,
                                    ok_callback=ok_callback,
                                    rectify_selected=rectify_selected,
                                    hilbert_selected=hilbert_selected
                                    )
        d.set_transient_for(self.widget)

    def save_eoi(self, menuitem, saveas):
        eoi = self.eegplot.get_eoi()
        if not self['dlgPref_radiobuttonUseWebOn'].get_active():
            # not using the web, write to local filesystem
            fname = fmanager.get_filename(
                    title='Enter filename for EOI')
            if not os.path.exists(fname):
                basepath, ext = os.path.splitext(fname)
                if ext.lower() != '.eoi':
                    fname += '.eoi'
            try:
                fh = file(fname, 'w')
                fh.write(eoi.to_conf_file())
            except IOError:
                error_msg('Could not write EOI to %s' % fname,
                          parent=self.widget)
            return

        #TODO: handle same filename vs different filename; add a save as?
        def ok_callback(m):
            pid=self.eegplot.get_eeg().get_pid()
            newName = m['filename']

            eoiNew = EOI()
            eoiNew.extend(eoi)
            
            def new_eoi_success():
                eeg = self.eegplot.get_eeg()
                success = self.eegplot.set_eoi(eoiNew)

                eoiNew.update_map(eeg.get_filename())
                eois = eeg.get_associated_files(atype=5, mapped=1)
                self.eoiMenu = self.make_context_menu(eois)
                dlgSave.hide_widget()
                simple_msg('%s successfully uploaded' % newName,
                              title='Congratulations',
                              parent=self.widget)
                if success: self.eegplot.plot()

            # make a new file
            try:
                eoiNew.new_web(pid, newName)
            except NameError:
                # fname already exists
                def response_callback(dialog, response):
                    if response==gtk.RESPONSE_YES:
                        eoiNew.set_exists_web(pid, newName)
                        eoiNew.update_web()                            
                        new_eoi_success()
                    else: dialog.destroy()
                msg = '%s already exists.  Overwrite?' % newName
                yes_or_no(msg=msg, title='Warning!',
                          responseCallback=response_callback,
                          parent=dlgSave.widget)
            else: new_eoi_success()

        if not saveas and eoi.is_web_file():
            eoi.update_web()
            simple_msg('%s updated' % eoi.filename,
                          title='You did it!',
                          parent=self.widget)
            return
        
        dlgSave = Dialog_SaveEOI(eoiActive=self.eegplot.get_eoi(),
                           eoisAll=self.eegplot.get_eeg().get_eois(),
                           ok_callback=ok_callback)
        dlgSave.get_widget().set_transient_for(self.widget)
        dlgSave.show_widget()

    def expose_event(self, widget, event):
        #now the traces resize themselves on window resize - hurrah! eli
        #I had more trouble with this than I care to admit, which explains the messiness of the code
        try: self.eegplot 
        except AttributeError: return False
        newsize = self.fig.get_size_inches()
        fsize = self.fsize
        #print newsize.all(), fsize.all() #why didn't .all() work??
        if (fsize[1] != newsize[1]) or (fsize[0] != newsize[0]) :
            self.eegplot.plot() #added these two lines -eli
            self.eegplot.draw()
            self.fsize = copy.deepcopy(newsize) #why didn't regular copy work?
	    return False    
	
    def configure_event(self, widget, event):
	
        return False

    def realize(self, widget):
        return False

    def motion_notify_event(self, event):
        try: self.eegplot
        except : return False

        if not event.inaxes: return
        
        # Motion within EEG axes
        if event.inaxes == self.axes:
            t, yt = event.xdata, event.ydata
            #t = float('%1.1f' % t)
            # Update status bar with time and electrode name and number
            trode = self.eegplot.get_channel_at_point(event.x, event.y, False)
            if trode is not None:
                gname, gnum = trode
                currdate = self.eegplot.eeg.get_date()
                timedelta = datetime.timedelta(0, event.xdata)
                
                if (currdate != None):
                    self.update_status_bar(
                        'Time  = %1.1f (s), %s, Electrode %s%d' % (t, str(currdate + timedelta), gname, gnum))
                else:
                    self.update_status_bar(
                        'Time  = %1.1f (s), Electrode %s%d' % (t, gname, gnum))

        # Motion within spectrum axes
        elif event.inaxes == self.axesSpec:
            t, f = event.xdata, event.ydata
            self.update_status_bar(
                'Time  = %1.1f (s), Freq = %1.1f (Hz)' % (t, f))

        return False

    def scroll_event(self, widget, event):
        "If in specgram resize"
        if event.direction == gdk.SCROLL_UP:
            direction = 1
        else:
            direction = -1

        l1,b1,w1,h1 = self.axes.get_position()
        l2,b2,w2,h2 = self.axesSpec.get_position()

        deltay = direction*0.1*h2
        h1 -= deltay
        h2 += deltay
        
        self.axes.set_position([l1, b2+h2, w1, h1])
        self.axesSpec.set_position([l2, b2, w2, h2])
        self.eegplot.plot() #added these two lines -eli
        self.eegplot.draw()
        self.canvas.draw()
        
    def button_press_event(self, event):
        try: self.eegplot
        except AttributeError: return False
	

        if not event.inaxes: return

        xa, ya = self.axes.transAxes.inverted().transform((event.x, event.y)) #replaced inverse_xy_tup with inverted().transform()
#        print 'axes coords', xa, ya
        
        self.buttonDown = event.button
        #annman = self.eegplot.annman

        if event.button == 1 or event.button == 3 :
            if event.inaxes == self.axes:
                t, yt = event.xdata, event.ydata
        if event.button==1:
            if event.inaxes == self.axes:
                self.eegplot.cursor.visible = False
                t, yt = event.xdata, event.ydata
                # Select an electrode if not locked.
                if not self.eegplot.lock_trode :                        
                    trode = self.eegplot.get_channel_at_point(event.x, event.y)
                    if trode is not None:
                        gname, gnum = trode
                        self.update_status_bar('Electrode: %s%d' % (gname, gnum))

        if event.button==3:
            # right click brings up the context menu
            if event.inaxes == self.axes:
                menu = self.eoiMenu
            elif event.inaxes == self.axesSpec:
                menu = self.specMenu
        return False

    def button_release_event(self, event):
        try: self.eegplot
        except AttributeError: return False
        self.eegplot.cursor.visible = True
        self.buttonDown = None

        return False

    def on_menuFilePreferences_activate(self, event=None):
        def mysql_callback(dbname, host, user, passwd, port):
            servers.sql.init(dbname, host, user, passwd, port)
            self.make_patients_menu()
            eegviewrc.sqlhost = host
            eegviewrc.sqluser = user
            eegviewrc.sqlpasswd = passwd
            eegviewrc.sqlport = port
            eegviewrc.save()
            
        def datamanager_callback(url, user, passwd, cachedir):
            servers.datamanager.init(url, user, passwd, cachedir)
            eegviewrc.httpurl = url
            eegviewrc.httpuser = user
            eegviewrc.httppasswd = passwd
            eegviewrc.httpcachedir = cachedir
            eegviewrc.save()
            
        d = Dialog_Preferences(
            mysqlCallBack       = mysql_callback,
            dataManagerCallBack = datamanager_callback)

        params = {
            'zopeServer' : eegviewrc.httpurl,
            'zopeUser' : eegviewrc.httpuser,
            'zopePasswd' : eegviewrc.httppasswd,
            'zopeCacheDir' : eegviewrc.httpcachedir,
            
            'mysqlDatabase' : eegviewrc.sqldatabase,
            'mysqlServer' : eegviewrc.sqlhost,
            'mysqlUser' : eegviewrc.sqluser,
            'mysqlPasswd' : eegviewrc.sqlpasswd,
            'mysqlPort' : eegviewrc.sqlport,
            }        
        d.set_params(params)
        d.show_widget()
        d.get_widget().set_transient_for(self.widget)

        return False

    def on_menuFileQuit_activate(self, event):
        update_rc_and_die()


    def get_eeg_params(self, fullpath):
        def callback(pars): pass
            
        dlg = Dialog_EEGParams(fullpath, callback)

        dlg.show_widget()

        response = dlg.widget.run()

        if response == gtk.RESPONSE_OK:
            dlg.hide_widget()
            pars =  dlg.get_params()
            return pars

    def autoload(self, options):
        """DEBUG only"""
        if options.filename != None:
            fullpath = options.filename
            basename, ext = os.path.splitext(fullpath)
            eeg = extmap[ext](fullpath)
            self.load_eeg(eeg)
        self.coh_autoload = options.coh
        self.wavelet_autoload = options.wavelet
        
        if options.eoi is not None:
            eoi = EOI(useFile=options.eoi)
            self.load_eoi(eoi)
        return False

    def on_menuFileOpen_activate(self, event):

        def ok_callback(dlg):
            fname = dlg.get_filename()
                    
            fullpath =  dlg.get_filename()
            fmanager.set_lastdir(fullpath)
            dlg.destroy()

            if not os.path.exists(fullpath):
                error_msg(
                    'Cannot find %s' % fullpath,
                    title='Error',
                    parent=Shared.windowMain.widget)

            basename, ext = os.path.splitext(fullpath)
            if not extmap.has_key(ext.lower()):
                error_msg(
                    'Do not know how to handle extension %s in %s' % (ext, fullpath),
                    title='Error',
                    parent=Shared.windowMain.widget)
                
                return
            else:
                loader = extmap[ext.lower()]
                try: eeg = loader(fullpath)
                except ValueError, msg:
                    msg = exception_to_str('Error loading EEG' )
                    error_msg(msg, title='Error loading EEG',
                              parent=Shared.windowMain.widget)
                    return
                else:
                    if eeg is None: return 

            print "on_menuFileOpen_activate: eeg ext is ", ext
            if (eeg.get_file_type() != 1): # hack -- .bnis do not need .amp files
                if len(eeg.amps)>0:
                    names = [os.path.split(fullname)[-1] for fullname in eeg.amps]
                    name = select_name(names, 'Pick the AMP file')
                    if name is None: return
                    else:
                        amp = eeg.get_amp(name)

                else:
                    amp = eeg.get_amp()
            else:
                amp = eeg.get_amp()
            if amp.message is not None:
                simple_msg(amp.message, title='Warning',
                           parent=Shared.windowMain.widget)

            self.load_eeg(eeg)

            return False
        

	dlg = Dialog_FileChooser(defaultDir=fmanager.get_lastdir(),
                                 okCallback=ok_callback,
                                 title='Select Neuroscanascii file',
                                 parent=self.win,
                                 previous_dirnames=fmanager.get_lastdirs())
	print fmanager.bni
	try: 
	    dlg.set_filename(fmanager.bni) #use the shared filemanager and eegviewrc file to autoload files when set 	
	except:
	    dlg.set_filename("")
	    	
	dlg.run()
        dlg.destroy()

#simple usability hack: chain in the view3 loader here
	try: self.eegplot
        except AttributeError:
            simple_msg(
                'You must first select an EEG from the Patients menu',
                title='Error',
                parent=self.widget)
            return
        from view3 import View3
        print "self.coh_autoload: %s %s"%(self.coh_autoload, self.wavelet_autoload)
        viewWin = View3(eegplot=self.eegplot,coh_autoload=self.coh_autoload,wavelet_autoload=self.wavelet_autoload)
	
        if viewWin.ok:
            viewWin.show()
        else:
            print >>sys.stderr, 'Got an error code from view3'
#/hack


    def load_eeg(self, eeg):
        dlg = gtk.Dialog('Please stand by')
        dlg.show()
        msg = gtk.Label('Loading %s; please hold on' % eeg.filename)
        msg.show()
        dlg.vbox.add(msg)            
        while gtk.events_pending(): gtk.main_iteration()

        try: self.eegplot
        except AttributeError: pass
        else: Observer.observers.remove(self.eegplot)        
        try: self.specPlot
        except AttributeError: pass
        else: Observer.observers.remove(self.specPlot)        


        self.eegplot = EEGPlot(eeg, self.canvas)
        self.specPlot = SpecPlot(self.axesSpec, self.canvas, self.eegplot)
        self.specMenu = self.make_spec_menu()
        dlg.destroy()
        while gtk.events_pending(): gtk.main_iteration()
        self.toolbar.set_eegplot(self.eegplot)
        try: self.eegplot.plot()
        except:
            msg = exception_to_str('Could not read data:')
            error_msg(msg, title='Error',
                      parent=Shared.windowMain.widget)
            return


        eois = eeg.get_associated_files(atype=5, mapped=1)
        self.eoiMenu = self.make_context_menu(eois)

        # change the window title
        self.win = self['windowMain']
        self.win.set_title(eeg.filename)
        self.eegplot.set_time_lim(0, 10, updateData=True)
        
    def on_menuFileSave_activate(self, event):
        not_implemented(self.widget)

    def on_menuFileExport_activate(self, event):
        # dump all the current data to a bunch of .wav files
        tmin, tmax = self.eegplot.get_time_lim()
        eeg = self.eegplot.get_eeg()
        t, data = eeg.get_data(tmin, tmax)
        amp = eeg.get_amp()
        did = amp.get_dataind_dict()
        freq = eeg.get_freq()
        eoi = self.eegplot.get_eoi()

        print "did=", did
        print "eoi=", eoi
        
        for index, chan in did.iteritems():
            if (chan not in eoi):
                continue
            (cname, cnum) = chan
            filename = str("%03d" % index) + "_" + cname + "_" + str(cnum) + "_" +  str(tmin) + "-" + str(tmax) + ".wav"
            print "on_menuFileExport_activate(): saving ", filename
            w = wave.open(filename, 'w')
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(freq)

            #print "data.shape is ", data.shape

            wav_array = data[:,int(index)]
            #print "wav_array length is ", len(wav_array), " with max of ", max(wav_array), "min of ", min(wav_array)

            # not sure how one chooses to "short-ize" this data!
            # arbitrarily make max SHRT_MAX and min SHRT_MIN or
            # something
            
            shrt_max = 32767
            shrt_min = -32768
            wav_max = max(wav_array)
            wav_min = min(wav_array)

            # mcc XXX: This conversion needs fixing... rectified signals
            # wind up with 0 = SHRT_MIN.

            shrt_array = zeros(len(wav_array), 'h')

            wav_max_max = max(wav_max, abs(wav_min))
            
            for i in range(0,len(wav_array)):
                wav_i_0to1 = (wav_max_max - wav_array[i]) / (2 * wav_max_max)
                shrt_array[i] = int(shrt_max - round(wav_i_0to1 * (shrt_max - shrt_min)))
            #print "len(shrt_array) is", len(shrt_array), " type of len(shrt_array) is ", type(len(shrt_array))
            w.writeframes(struct.pack('%dh' % len(shrt_array), *shrt_array))
            w.close()
            
    def on_menuHelpAbout_activate(self, event):
        not_implemented(self.widget)

    def on_menuChannelWindow_activate(self, event):
        try: self.eegplot
        except AttributeError:
            simple_msg(
                'You must first select an EEG from the Patients menu',
                title='Error',
                parent=self.widget)
            return

        win = ChannelWin(eegplot=self.eegplot)
        win.show()

    def on_menuHistogramWindow_activate(self, event):
        try: self.eegplot
        except AttributeError:
            simple_msg(
                'You must first select an EEG from the Patients menu',
                title='Error',
                parent=self.widget)
            return

        win = HistogramWin(eegplot=self.eegplot)
        win.show()

    def on_menuAcorrWindow_activate(self, event):
        try: self.eegplot
        except AttributeError:
            simple_msg(
                'You must first select an EEG from the Patients menu',
                title='Error',
                parent=self.widget)
            return
        win = AcorrWin(eegplot=self.eegplot)
        win.show()

    def on_menuEmbedWindow_activate(self, event):
        try: self.eegplot
        except AttributeError:
            simple_msg(
                'You must first select an EEG from the Patients menu',
                title='Error',
                parent=self.widget)
            return
        from embed import EmbedWin
        embedWin = EmbedWin(eegplot=self.eegplot)
        embedWin.show()

    def on_menuCoherenceWindow_activate(self, event):
        print "on_menuCoherenceWindow_activate"
        try: self.eegplot
        except AttributeError:
            simple_msg(
                'You must first select an EEG from the Patients menu',
                title='Error',
                parent=self.widget)
            return
        from coherence_window import CoherenceWin
        coherenceWin = CoherenceWin(eegplot=self.eegplot)

        coherenceWin.show()

    def on_menuView3DWindow_activate(self, event):
        try: self.eegplot
        except AttributeError:
            simple_msg(
                'You must first select an EEG from the Patients menu',
                title='Error',
                parent=self.widget)
            return
        from view3 import View3
        viewWin = View3(eegplot=self.eegplot)
	
        if viewWin.ok:
            viewWin.show()
        else:
            print >>sys.stderr, 'Got an error code from view3'


    def on_menuPhaseSynchronyPlot_activate(self, event) :
        try : self.eegplot
        except AttributeError :
            simple_msg(
                'You must first select an EEG',
                title='Error',
                parent=self.widget)
            return

        dlgPhaseSynchronyPlot = Dialog_PhaseSynchronyPlot(self.eegplot)
        print dlgPhaseSynchronyPlot
        dlgPhaseSynchronyPlot.show_widget()

    def on_menuSpecWindow_activate(self, event):
        try: self.eegplot
        except AttributeError:
            simple_msg(
                'You must first select an EEG',
                title='Error',
                parent=self.widget)
            return

        specWin = SpecWin(eegplot=self.eegplot)
        specWin.show()                
        
    def on_menuEventRelatedSpecWindow_activate(self, event):

        def ok_callback(erspec_params):
            print "on_menuEventRelatedSpecWindow_activate().ok_callback(): foo=", erspec_params
            win = EventRelatedSpecWin(erspec_params, eegplot=self.eegplot)
            win.show()
        
        try: self.eegplot
        except AttributeError:
            simple_msg(
                'You must first select an EEG',
                title='Error',
                parent=self.widget)
            return

        specWin = Dialog_EventRelatedSpec(ok_callback)
        specWin.show_widget()
        #specWin.show()

        return False
        
    def on_menuComputeExportToCohstat_activate(self, event):
        try: self.eegplot
        except AttributeError:
            simple_msg(
                'You must first select an EEG from the Patients menu',
                title='Error',
                parent=self.widget)
            return
        eoi = self.eegplot.get_eoi()
        if len(eoi)==64: 
            d = Dialog_CohstatExport(self.eegplot.get_eeg(), eoi)
        else:
            d = Dialog_CohstatExport(self.eegplot.get_eeg())
        d.get_widget().set_transient_for(self.widget)
        d.show_widget()
        
        return False

def update_rc_and_die(*args):
    [eegviewrc.lastdir, 
     eegviewrc.lastdir1,    
     eegviewrc.lastdir2,    
     eegviewrc.lastdir3,    
     eegviewrc.lastdir4,    
     eegviewrc.lastdir5,    
     eegviewrc.lastdir6,    
     eegviewrc.lastdir7,    
     eegviewrc.lastdir8,    
     eegviewrc.lastdir9] = fmanager.get_lastdirs()
    #eegviewrc.figsize = Shared.windowMain.fig.get_size_inches()
    eegviewrc.save()
    gtk.main_quit()

if __name__=='__main__':
    __import__('__init__')
    Shared.windowMain = MainWindow()
    Shared.windowMain.show_widget()
    from optparse import OptionParser
    parser = OptionParser()

    parser.add_option("-f", "--file",
                      action="store", type="string", dest="filename",
                      default=None,                      
                      help="Autoload eeg from file", metavar="FILE")

    parser.add_option("-e", "--eoi",
                      action="store", type="string", dest="eoi",
                      default=None,                      
                      help="Autoload eoi from eoi file", metavar="FILE")

    parser.add_option("-c", "--coh_explore",
                      action="store_true", dest="coh",
                      default=False,                      
                      help="run straight to coh explorer")

    parser.add_option("-w", "--wavelet_runner",
                      action="store_true", dest="wavelet",
                      default=False,                      
                      help="run straight to wavelet runner")


    (options, args) = parser.parse_args()
    
    if options.filename is not None:
        Shared.windowMain.autoload(options)
    else:
        Shared.windowMain.autoload(options)
        #No longer load the sql/zope dialog.
        #Shared.windowMain.on_menuFilePreferences_activate(None)
        pass
    Shared.windowMain.widget.connect('expose-event', Shared.windowMain.expose_event) #handle page resizes -eli
    Shared.windowMain.widget.connect('destroy', update_rc_and_die)
    Shared.windowMain.widget.connect('delete_event', update_rc_and_die)
    #Shared.windowMain['menubarMain'].hide()
    # chain in the eeg file loading right away!
    Shared.windowMain.on_menuFileOpen_activate(None)
    try: gtk.main()
    except KeyboardInterrupt:
        update_rc_and_die()
