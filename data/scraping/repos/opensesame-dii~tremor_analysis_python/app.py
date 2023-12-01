#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from logging import root, warning
from os import makedirs, path, name
from io import BytesIO
# from shutil import rmtree
import matplotlib.pyplot as plt
from copy import deepcopy
from sys import exit
from argparse import ArgumentParser

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# from PIL import Image, ImageTk
import numpy as np
from scipy.signal import hamming, detrend, morlet2, cwt, spectrogram, get_window, butter, sosfilt
from matplotlib import use
use('TkAgg')
from matplotlib.mlab import cohere, window_hanning
from matplotlib.pyplot import specgram as pltspectrogram
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import backend_tools as cbook
# import PySimpleGUI as sg
import pandas as pd
from sklearn.decomposition import PCA

plt.rcParams['figure.subplot.bottom'] = 0.18
# number of sensor
# e.g. 3 for "accelerometer, magnetmeter and gyroscope", 2 for "left arm and right arm"
# SENSORS = 3




# figure size settings #白い画像を生成
dpi = 97
figsize_big = (12, 3)
figsize_small = (4, 3)
figsize_pixel_big = (figsize_big[0] * dpi, figsize_big[1] * dpi)
figsize_pixel_small = (figsize_small[0] * dpi, figsize_small[1] * dpi)




def remove_ext(filename):
    return path.splitext(path.basename(filename))[0]  #拡張子を消す

# Tkinter に移行したため不要
# def get_img_data(f, maxsize=(1200, 850)):
#     """
#     Generate image data using PIL
#     """
#     img = Image.open(f)
#     img.thumbnail(maxsize)
#     bio = BytesIO()
#     img.save(bio, format="PNG")
#     del img
#     return bio.getvalue()  #

def detect_data_warning(data):
    """
    detect going off the scale      #変なデータをはじく警告
    """
    max_idx = np.where(data == data.max())[0]
    min_idx = np.where(data == data.min())[0]

    return any([max_idx[i] + 1 == max_idx[i + 1] for i in range(len(max_idx) - 1)]) or any([min_idx[i] + 1 == min_idx[i + 1] for i in range(len(min_idx) - 1)])

class ScrollableFrame(tk.Frame):
# https://water2litter.net/rum/post/python_tkinter_resizeable_canvas/

    def __init__(self, parent, minimal_canvas_size):
        tk.Frame.__init__(self, parent)

        self.minimal_canvas_size = minimal_canvas_size

        # 縦スクロールバー
        self.vscrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL)
        self.vscrollbar.pack(fill=tk.Y, side=tk.RIGHT, expand=False)

        # 横スクロールバー
        self.hscrollbar = ttk.Scrollbar(self, orient=tk.HORIZONTAL)
        self.hscrollbar.pack(fill=tk.X, side=tk.BOTTOM, expand=False)

        # Canvas
        self.canvas = tk.Canvas(self, bd=0, highlightthickness=0,
            yscrollcommand=self.vscrollbar.set, xscrollcommand=self.hscrollbar.set,  bg="#778899")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # スクロールバーをCanvasに関連付け
        self.vscrollbar.config(command=self.canvas.yview)
        self.hscrollbar.config(command=self.canvas.xview)

        self.canvas.configure(yscrollcommand=self.vscrollbar.set)
        self.canvas.configure(xscrollcommand=self.hscrollbar.set)


        # Canvasの位置の初期化
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

        # スクロール範囲の設定
        self.canvas.config(scrollregion=(0, 0, self.minimal_canvas_size[0], self.minimal_canvas_size[1]))

        # Canvas内にフレーム作成
        self.interior = tk.Frame(self.canvas, bg="#778899")
        self.canvas.create_window(0, 0, window=self.interior, anchor='nw')

        # Canvasの大きさを変える関数
        def _configure_interior(event):
            size = (max(self.interior.winfo_reqwidth(), self.minimal_canvas_size[0]),
                max(self.interior.winfo_reqheight(), self.minimal_canvas_size[1]))
            self.canvas.config(scrollregion=(0, 0, size[0], size[1]))
            if self.interior.winfo_reqwidth() != self.canvas.winfo_width():
                self.canvas.config(width = self.interior.winfo_reqwidth())
            if self.interior.winfo_reqheight() != self.canvas.winfo_height():
                self.canvas.config(height = self.interior.winfo_reqheight())

        # 内部フレームの大きさが変わったらCanvasの大きさを変える関数を呼び出す
        self.interior.bind('<Configure>', _configure_interior)
class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()

        #オプション関数取得
        parser =ArgumentParser()
        parser.add_argument("--encoding",default="utf-8")
        parser.add_argument("--row_start",type=int,default=1)
        parser.add_argument("--column_start",type=int,default=1)
        parser.add_argument("--sensors_num",type=int,default=3)
        parser.add_argument("--min_frequency",type=int,default=2)
        parser.add_argument("--max_frequency",type=int,default=20)

        self.args = parser.parse_args()

        # number of sensor
        # e.g. 3 for "accelerometer, magnetmeter and gyroscope", 2 for "left arm and right arm"
        self.SENSORS_NUM = self.args.sensors_num

        self.sampling_rate = 200
        self.segment_duration_sec = 5
        self.frame_range = [0, -1]

        # frequency range
        if (self.args.min_frequency < 0):
            raise ValueError("min_frequency must be greater than or equal to 0")
        if (self.args.min_frequency >= self.args.max_frequency):
            raise ValueError("min_frequency must be less than max_frequency")
        self.min_f = self.args.min_frequency
        self.max_f = self.args.max_frequency


        self.result_value_keys = [
            "sp_peak_amplitude",
            "sp_peak_frequency",
            "sp_peak_time",
            "sa_peak_amplitude",
            "sa_peak_frequency",
            "sa_fwhm",
            "sa_hwp",
            "sa_tsi",
        ]
        self.result_graph_keys = [
            "sa_graph",
            "sp_graph",
        ]

        self.figsize_small = (3.3, 2.5)
        self.figsize_large = (9.9, 3)
        self.warning_width=20
        self.warning_height=10

        self.init_data()

        # exit event
        self.protocol("WM_DELETE_WINDOW", self.app_exit)

        #メインウィンドウの設定
        # root = tkinter.Tk()
        # root.title("tremor")
        self.title("tremor")

        # if name == "nt":
        #     self.state("zoomed")
        # elif name == "posix":
        #     self.attributes("-zoomed", "1")
        self.configure(bg="#778899")


        #scrollbar canvas

        canvas = ScrollableFrame(self, [100, 100])
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        frame_canvas = tk.Frame(canvas.interior, bg="#778899")
        frame_canvas.pack()
        info_frame = tk.Frame(frame_canvas, bg="#778899")
        img_frame = tk.Canvas(frame_canvas, bg="#778899")

        #データを選択するフレーム
        data_input_frame = ttk.Frame(info_frame,relief="groove",borderwidth=5)
        data_label1 = ttk.Label(
            data_input_frame,
            text="Data1:"
            )
        data_label1.grid(row=0, column=0)

        self.browse_button1 = ttk.Button(data_input_frame,text="Browse", command=lambda: self.file_dialog(0))
        # self.brows_button1.bind("<ButtonPress>", self.file_dialog)
        self.browse_button1.grid(row=0, column=1)

        self.clear_button = ttk.Button(data_input_frame,text="Clear", command=lambda: self.reset())
        self.clear_button.grid(row=0, column=2)

        data_label2 = ttk.Label(data_input_frame,text = "Data2:")
        data_label2.grid(row=1, column=0)

        self.browse_button2 = ttk.Button(data_input_frame,text="Browse", command=lambda: self.file_dialog(1))
        # self.brows_button2.bind("<ButtonPress>",self.file_dialog)
        self.browse_button2.grid(row=1, column=1)

        progress = ttk.Label(data_input_frame,text= "progress:")
        progress.grid(row=1, column=2)

        self.progress_bar_text = tk.StringVar()
        self.progress_bar_text.set("--")
        self.progress_bar = ttk.Label(data_input_frame, textvariable=self.progress_bar_text)
        self.progress_bar.grid(row=1, column=3)
        per = ttk.Label(data_input_frame,text = "%")
        per.grid(row=1, column=4)

        #モード選択
        settings_frame = ttk.Frame(info_frame,relief="groove")
        now_showing = ttk.Label(settings_frame,text="Currently displayed:")
        now_showing.grid(row=0, column=0)
        sensor = ttk.Label(settings_frame, text="Sensor")
        sensor.grid(row=1, column=0)
        analysis = ttk.Label(settings_frame, text="Analysis mode:")
        analysis.grid(row=2, column=0)
        #warning
        self.warning_frame=ttk.Frame(info_frame)
        self.warning=ttk.Label(self.warning_frame,text="off-scale warning")
        self.warning_box=tk.Text(self.warning_frame,width=self.warning_width,height=self.warning_height)
        self.warning_box.grid(row=1,column=0)
        self.warning.grid(row=0,column=0)

        #この書き方（moduleを使う）は良くない気がする、、、
        module = ("data_list","mode_list","analysis","sensor_list")
        self.now_showing_box = ttk.Combobox(settings_frame, values=self.data_index, state="readonly")
        self.now_showing_box.set(self.data_index[0])
        self.now_showing_box.bind("<<ComboboxSelected>>", self.onchange_showing)
        self.now_showing_box.grid(row=0, column=1)
        self.sensor_box = ttk.Combobox(settings_frame, values=self.sensors, state="readonly")
        self.sensor_box.set(self.sensors[0])
        self.sensor_box.bind("<<ComboboxSelected>>", self.onchange_sensor)
        self.sensor_box.grid(row=1, column=1)
        self.analysis_box = ttk.Combobox(settings_frame, values=self.modes, state="readonly")
        self.analysis_box.set(self.modes[0])
        self.analysis_box.bind("<<ComboboxSelected>>", self.onchange_analysis)
        self.analysis_box.grid(row=2, column=1)

        #settings
        settings_frame2 = ttk.Frame(info_frame,relief="groove")
        seg = ttk.Label(settings_frame2, text="Segment duration(s):")
        seg.grid(row=0, column=0)
        samp = ttk.Label(settings_frame2, text="Sampling rate(Hz):")
        samp.grid(row=1, column=0)
        frame_range = ttk.Label(settings_frame2, text="Frame range:")
        frame_range.grid(row=2, column=0)

        # setup validation function
        tcl_can_enter_as_number = self.register(self.can_enter_as_number)

        self.seg_txt = tk.Entry(settings_frame2,width=20,validate="key", vcmd=(tcl_can_enter_as_number, "%S"))
        self.seg_txt.grid(row=0, column=1)
        self.samp_txt = tk.Entry(settings_frame2, width=20,validate="key", vcmd=(tcl_can_enter_as_number, "%S"))
        self.samp_txt.grid(row=1, column=1)
        self.range_txt1 = tk.Entry(settings_frame2, width=20,validate="key", vcmd=(tcl_can_enter_as_number, "%S"))
        self.range_txt1.grid(row=2, column=1)
        self.range_txt2 = tk.Entry(settings_frame2,width=20,validate="key", vcmd=(tcl_can_enter_as_number, "%S"))
        self.range_txt2.grid(row=2, column=3)
        self.range_to = ttk.Label(settings_frame2, text="to")
        self.range_to.grid(row=2, column=2)
        self.apply_button = ttk.Button(settings_frame2, text="Apply")
        self.apply_button.bind("<ButtonPress>", self.onchange_settings)
        self.apply_button.grid(row=3, column=0)

        #result
        result_frame = ttk.Frame(info_frame, relief="groove")

        self.clip = ttk.Button(result_frame, text="Copy to clipboard", command=self.copy_to_clipboard)
        self.clip.grid(row=0, column=0,sticky=tk.W)

        #data出力のフレーム
        data1_frame = ttk.Frame(result_frame, relief="groove")
        data2_frame = ttk.Frame(result_frame, relief="groove")

        self.data_frames = [data1_frame,data2_frame]

        for data_ in self.data_frames:
            self.spa_txt = tk.Entry(data_,width=20)
            self.spa_txt.insert(tk.END,"None")
            self.spa_txt.grid(row=0, column=1)
            #self.spa_txt.configure(state="readonly")
            self.spf_txt = tk.Entry(data_, width=20)
            self.spf_txt.insert(tk.END,"None")
            self.spf_txt.grid(row=1, column=1)
            self.spt_txt = tk.Entry(data_,width=20)
            self.spt_txt.insert(tk.END,"None")
            self.spt_txt.grid(row=2, column=1)
            self.wpa_txt = tk.Entry(data_, width=20)
            self.wpa_txt.insert(tk.END,"None")
            self.wpa_txt.grid(row=3, column=1)
            self.wpf_txt = tk.Entry(data_, width=20)
            self.wpf_txt.insert(tk.END,"None")
            self.wpf_txt.grid(row=4, column=1)
            self.fhm_txt = tk.Entry(data_, width=20)
            self.fhm_txt.insert(tk.END,"None")
            self.fhm_txt.grid(row=5, column=1)
            self.hp_txt = tk.Entry(data_, width=20)
            self.hp_txt.insert(tk.END,"None")
            self.hp_txt.grid(row=6, column=1)
            self.tsi_txt = tk.Entry(data_, width=20)
            self.tsi_txt.insert(tk.END,"None")
            self.tsi_txt.grid(row=7, column=1)



            spa = ttk.Label(data_, text="Spectrogram Peak Amplitude:")
            spa.grid(row=0, column=0)
            spf = ttk.Label(data_, text="Spectrogram Peak Frequency(Hz): ")
            spf.grid(row=1, column=0)
            spt = ttk.Label(data_, text = "Spectrogram Peak Time(s): ")
            spt.grid(row=2, column=0)
            wpa = ttk.Label(data_, text = "Whole Peak Amplitude: ")
            wpa.grid(row=3, column=0)
            wpf = ttk.Label(data_, text = "Whole Peak Frequency(Hz): ")
            wpf.grid(row=4, column=0)
            fhm = ttk.Label(data_, text = "Full-width Half Maximum(Hz): ")
            fhm.grid(row=5, column=0)
            hp = ttk.Label(data_, text = "Half-width Power: ")
            hp.grid(row=6, column=0)
            tsi = ttk.Label(data_, text = "Tremor Stability Index: ")
            tsi.grid(row=7, column=0)





        #coherence
        coherence_frame = ttk.Frame(result_frame, relief="groove")
        coh_x = ttk.Label(coherence_frame, text="FT coherence integral(x ):")
        coh_x.grid(row=0, column=0)
        coh_y = ttk.Label(coherence_frame, text="FT coherence integral(y ):")
        coh_y.grid(row=1, column=0)
        coh_z = ttk.Label(coherence_frame, text="FT coherence integral(z ):")
        coh_z.grid(row=2, column=0)
        coh_norm = ttk.Label(coherence_frame, text="FT coherence integral(norm): ")
        coh_norm.grid(row=3, column=0)

        self.coherence_txts = []
        x_txt = tk.Entry(coherence_frame, width=20)
        x_txt.insert(tk.END,"None")
        x_txt.grid(row=0, column=1)
        self.coherence_txts.append(x_txt)
        y_txt = tk.Entry(coherence_frame, width=20)
        y_txt.insert(tk.END,"None")
        y_txt.grid(row=1, column=1)
        self.coherence_txts.append(y_txt)
        z_txt = tk.Entry(coherence_frame, width=20)
        z_txt.insert(tk.END,"None")
        z_txt.grid(row=2, column=1)
        self.coherence_txts.append(z_txt)
        norm_txt = tk.Entry(coherence_frame, width=20)
        norm_txt.insert(tk.END,"None")
        norm_txt.grid(row=3, column=1)
        self.coherence_txts.append(norm_txt)


        #data previewのグラフ


        self.can_preview = ttk.Frame(img_frame)
        fig = Figure(figsize = self.figsize_large, dpi = 100)
        self.canvas = FigureCanvasTkAgg(fig,self.can_preview)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()
        toolbar1 = NavigationToolbar2Tk(self.canvas, self.can_preview)


        self.can2 = ttk.Frame(img_frame)
        self.canvas2 = FigureCanvasTkAgg(fig, self.can2)
        self.canvas2.draw()
        self.canvas2.get_tk_widget().pack()
        toolbar2 = NavigationToolbar2Tk(self.canvas2, self.can2)
        #canvas2.get_tk_widget().grid(row=4, column=1)

        can3 = tk.Canvas(img_frame,width=400, height=300)


        can_x = ttk.Frame(can3)
        can_y = ttk.Frame(can3)
        can_z = ttk.Frame(can3)

        self.can_list = [can_x,can_y,can_z]
        for can_ in self.can_list:
            fig3 = Figure(figsize = self.figsize_small, dpi = 100)
            ax = fig3.add_subplot(1,1,1)
            self.canvas_ = FigureCanvasTkAgg(fig3, can_)
            self.canvas_.draw()
            self.canvas_.get_tk_widget().pack()
            self.toolbar3 = FigureNavigator(self.canvas_, can_)

        """
        #スクロールバー
        hbar = tk.Scrollbar(img_frame, orient=tk.HORIZONTAL)
        hbar.config(command=img_frame.xview)
        hbar.grid(row=3,column=0)
        vbar = tk.Scrollbar(img_frame, orient=tk.VERTICAL)
        vbar.config(command=img_frame.yview)
        vbar.grid(row=0,column=1)
        """

        #フレームの配置
        info_frame.grid(row=0, column=0)
        img_frame.grid(row=0, column=1)
        data_input_frame.grid(row = 0, column=0, padx=10, pady=20,sticky=tk.W)
        settings_frame.grid(row=1, column=0, padx=10, pady=5,sticky=tk.W)
        self.warning_frame.grid(row=0,column=1,columnspan=2,padx=10)
        settings_frame2.grid(row=2, column=0, padx=10, pady=5,sticky=tk.W,columnspan=2)
        result_frame.grid(row=3, column=0,sticky=tk.W, padx=10,columnspan=2)
        data1_frame.grid(row=1, column=0,sticky=tk.W,pady=10)
        data2_frame.grid(row=2, column=0,sticky=tk.W, pady=10)
        coherence_frame.grid(row=3,column=0, sticky=tk.W,pady=10)

        self.can_preview.grid(row=0, column=0)
        self.can2.grid(row=1, column=0)
        can3.grid(row=2, column=0)
        can_x.grid(row=0, column=0)
        can_y.grid(row=0, column=1)
        can_z.grid(row=0, column=2)

        # root.mainloop()

        self.reset(is_init=True)


    def init_data(self):
        plt.close("all")
        empty_fig_small, ax = plt.subplots(figsize=self.figsize_small, dpi=100)
        empty_fig_large, ax = plt.subplots(figsize=self.figsize_large, dpi=100)

        self.filenames = ["", ""]
        self.data = [None, None]
        self.current_data = 0 # showing data index (0 or 1)
        self.data_preview_fig = [[empty_fig_large for i in range(self.SENSORS_NUM)], [empty_fig_large for i in range(self.SENSORS_NUM)]]

        self.modes = ["Spectral Amplitude", "Spectrogram"] # あとで修正(wavelet)
        self.current_mode = 0
        self.sensors = ["sensor" + str(i + 1) for i in range(self.SENSORS_NUM)] # "sensor1", "sensor2", ...
        self.current_sensor = 0
        self.data_index = ["Data1", "Data2"]

        # empty[sensor][axis]
        # axis -> x, y, z, norm
        empty = [[None, None, None, None] for i in range(self.SENSORS_NUM)]

        self.results = {
            0: { # file 1
                # "sa_peak_amplitude" : deepcopy(empty) , # on "spectral amplitude" mode
                # "sa_peak_frequency" : deepcopy(empty) ,
                # "sa_fwhm"           : deepcopy(empty) ,
                # "sa_hwp"            : deepcopy(empty) ,
                # "sa_tsi"            : deepcopy(empty) ,

                # "sp_peak_amplitude" : deepcopy(empty) , # on "Spectrogram" mode
                # "sp_peak_frequency" : deepcopy(empty) ,
                # "sp_peak_time"      : deepcopy(empty) ,

                # "wt_peak_amplitude" : deepcopy(empty) , # on "Wavelet Spectrogram" mode
                # "wt_peak_frequency" : deepcopy(empty) ,
                # "wt_peak_time"      : deepcopy(empty) ,

                # "sa_graph"          : [[empty_fig_small, empty_fig_small, empty_fig_small, empty_fig_large] for i in range(self.SENSORS_NUM)] ,
                # "sp_graph"          : [[empty_fig_small, empty_fig_small, empty_fig_small, empty_fig_large] for i in range(self.SENSORS_NUM)] ,
            },
            1: { # file 2
                # "sa_peak_amplitude" : deepcopy(empty) , # on "spectral amplitude" mode
                # "sa_peak_frequency" : deepcopy(empty) ,
                # "sa_fwhm"           : deepcopy(empty) ,
                # "sa_hwp"            : deepcopy(empty) ,
                # "sa_tsi"            : deepcopy(empty) ,

                # "sp_peak_amplitude" : deepcopy(empty) , # on "Spectrogram" mode
                # "sp_peak_frequency" : deepcopy(empty) ,
                # "sp_peak_time"      : deepcopy(empty) ,

                # "wt_peak_amplitude" : deepcopy(empty) , # on "Wavelet Spectrogram" mode
                # "wt_peak_frequency" : deepcopy(empty) ,
                # "wt_peak_time"      : deepcopy(empty) ,

                # "sa_graph"          : [[empty_fig_small, empty_fig_small, empty_fig_small, empty_fig_large] for i in range(self.SENSORS_NUM)] ,
                # "sp_graph"          : [[empty_fig_small, empty_fig_small, empty_fig_small, empty_fig_large] for i in range(self.SENSORS_NUM)] ,
            },
            -1: { # relational values between file1 and file 2
                "coherence"         : deepcopy(empty) ,
            }
        }
        for i in range(2):
            for key in self.result_value_keys:
                self.results[i][key] = deepcopy(empty)
            for key in self.result_graph_keys:
                self.results[i][key] = [[empty_fig_small, empty_fig_small, empty_fig_small, empty_fig_large] for i in range(self.SENSORS_NUM)]


        plt.close("all")

    def app_exit(self):
        plt.close('all')
        #self.destroy()
        exit()
    def debug(self):
        data = 0
        entry_names = self.data_frames[data].children.keys()
        print(entry_names)

    def gui_update(self, file_update=None, settings_changed=False, recalculation=False, change_target=False):
        self.change_progress("00")
        if (recalculation):
            if (self.data[0] is not None and self.data[1] is not None):
                for sensor_idx in range(self.SENSORS_NUM):
                    for axis_idx in range(3):
                        self.ft_coherence(
                            sensor_idx,
                            axis_idx,
                            self.data[0][:, 3 * sensor_idx + axis_idx],
                            self.data[1][:, 3 * sensor_idx + axis_idx],
                            self.sampling_rate,
                            self.frame_range[0],
                            self.frame_range[1],
                        )
                    self.ft_coherence(
                        sensor_idx,
                        3,
                        np.linalg.norm(self.data[0][:, 3 * sensor_idx: 3 * sensor_idx + 3], axis=1),
                        np.linalg.norm(self.data[1][:, 3 * sensor_idx: 3 * sensor_idx + 3], axis=1),
                        self.sampling_rate,
                        self.frame_range[0],
                        self.frame_range[1],
                    )
                    # self.wt_coherence(
                    #     self.data[0][:, 3 * sensor_idx : 3 * (sensor_idx + 1)].T,
                    #     self.data[1][:, 3 * sensor_idx : 3 * (sensor_idx + 1)].T,
                    #     self.sampling_rate,
                    #     self.frame_range[0],
                    #     self.frame_range[1],
                    # )
                pass

            target_data = []
            if (file_update is None and not settings_changed):
                target_data.append(0)
                target_data.append(1)
            elif (settings_changed):
                for i in range(2):
                    if (self.data[i] is not None):
                        target_data.append(i)
            else:
                target_data.append(file_update)
            for data_idx in range(len(target_data)):
                for sensor_idx in range(self.SENSORS_NUM):
                    progress = (data_idx * self.SENSORS_NUM + sensor_idx) * 100 // (self.SENSORS_NUM * len(target_data))
                    self.change_progress(str(progress))
                    self.spectrogram_analize(
                        target_data[data_idx],
                        sensor_idx,
                        self.data[target_data[data_idx]][:, sensor_idx*self.SENSORS_NUM: sensor_idx*self.SENSORS_NUM + self.SENSORS_NUM].T,
                        self.sampling_rate,
                        self.sampling_rate * self.segment_duration_sec,
                        self.filenames[target_data[data_idx]],
                        self.sensors[sensor_idx],
                        self.frame_range[0],
                        self.frame_range[1],
                    )
                    self.power_density_analize(
                        target_data[data_idx],
                        sensor_idx,
                        self.data[target_data[data_idx]][:, sensor_idx*3: sensor_idx*3 + 3].T,
                        self.sampling_rate,
                        self.sampling_rate * self.segment_duration_sec,
                        self.filenames[target_data[data_idx]],
                        self.sensors[sensor_idx],
                        self.frame_range[0],
                        self.frame_range[1],
                    )
            change_target = True

        if (change_target):
            self.update_all_figure()
        self.change_progress("--")

    def update_all_figure(self):
        # preview update
        self.update_figure(self.can_preview, self.data_preview_fig[self.current_data][self.current_sensor])

        # calculated value update
        self.update_results()

        # graph update
        for i in range(3):
            # self.can_list[i] = FigureCanvasTkAgg(self.)
            self.update_figure(self.can_list[i], self.results[self.current_data][self.result_graph_keys[self.current_mode]][self.current_sensor][i])
        self.update_figure(self.can2, self.results[self.current_data][self.result_graph_keys[self.current_mode]][self.current_sensor][3])

    def update_figure(self, figure_canvas, fig):
        """
        Params:
        figure_canvas: ttk.Frame
            target canvas
        fig: matplotlib.figure.Figure
            figure
        """
        entry_names = list(figure_canvas.children.keys())
        for entry_name in entry_names:
            figure_canvas.children[entry_name].destroy()
        canvas_ = FigureCanvasTkAgg(fig, figure_canvas)
        canvas_.draw()
        canvas_.get_tk_widget().pack()
        FigureNavigator(canvas_, figure_canvas)

    #def mouse_y_scroll(self,event):
       # if event.delta > 0:
       #     self.canvas.yview_scroll(-1,"units")
       # elif event.delta < 0:
       #     self.canvas.yview_scroll(1, "units")

    #コマンドライン引数を設定する関数
    def get_option(self):
        pass
        #print(args.encoding)


    #ファイルを選ぶ関数
    def file_dialog(self, selected):
        ftypes =[('EXCELファイル/CSVファイル', '*.xlsx'),
            ('EXCELファイル/CSVファイル', '*.xlsm'),
            ('EXCELファイル/CSVファイル', '*.csv')]
        fname = filedialog.askopenfilename(filetypes=ftypes)

        if (fname == "" or isinstance(fname, tuple)):
            print("no file selected")
            return

        print(f"loading {fname}")
        self.change_progress("00")
        # print(selected)
        if (self.filenames[selected] != fname):
            plt.close("all")
            self.filenames[selected] = fname

            # Optimize to motion sensor by Logical Product Inc

            df = pd.read_csv(fname, header=None, skiprows=self.args.row_start - 1, usecols=[i + self.args.column_start -1 for i in range(self.args.sensors_num * 3)],encoding=self.args.encoding)
            npdata = np.array(df.values.flatten())
            self.data[selected] = np.reshape(npdata,(df.shape[0],df.shape[1]))

            # テスト用
            print(self.data[selected].shape)

            # warning to go off the scale
            off_scale=[]
            for i in range(self.data[selected].shape[1]):
                if (detect_data_warning(self.data[selected][:,i])):
                    print(f"WARNING: column {i} may go off the scale")
                    off_scale.append(i)

            if len(off_scale)!=0:
                self.warning_box.insert("end","Data"+str(selected+1)+":")

            for i in off_scale:
                self.warning_box.insert("end",str(i)+",")
            self.warning_box.insert("end","\n")
            # update
            for i in range(self.SENSORS_NUM):
                self.data_preview_fig[selected][i], ax = plt.subplots(figsize=self.figsize_large, dpi=100)
                for axis_idx in range(3):
                    #ax.plot(detrend(self.data[selected][:,i * 3 + axis_idx]))
                    ax.plot(self.data[selected][:,i * 3 + axis_idx])
                    ax.set_title(f"{path.basename(fname)} preview")
                    ax.set_xlabel("sample")
                ax.legend(labels=["x", "y", "z"],bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0, fontsize=15)
            plt.close("all")
            self.gui_update(file_update=selected, settings_changed=False, recalculation=True, change_target=False)
        print(f"{fname} was loaded successfully")

    def update_results(self):
        for data in range(2):
            entry_names = list(self.data_frames[data].children.keys())
            for key in range(len(self.result_value_keys)):

                # self.data_frames[data].children[key].text = str(self.results[data][self.result_value_keys[key]][self.current_mode][-1])
                self.data_frames[data].children[entry_names[key]].configure(state = "normal")
                self.data_frames[data].children[entry_names[key]].delete(0, "end")
                self.data_frames[data].children[entry_names[key]].insert(0, str(self.results[data][self.result_value_keys[key]][self.current_sensor][3]))
                self.data_frames[data].children[entry_names[key]].configure(state = "readonly")
        for axis_idx in range(4):
            self.coherence_txts[axis_idx].configure(state = "normal")
            self.coherence_txts[axis_idx].delete(0, "end")
            self.coherence_txts[axis_idx].insert(0, str(self.results[-1]["coherence"][self.current_sensor][axis_idx]))
            self.coherence_txts[axis_idx].configure(state = "readonly")

    def onchange_settings(self, event):
        self.segment_duration_sec = int(self.seg_txt.get())
        self.sampling_rate = int(self.samp_txt.get())
        self.frame_range[0] = int(self.range_txt1.get())
        self.frame_range[1] = int(self.range_txt2.get())
        self.gui_update(file_update=None, settings_changed=True, recalculation=True, change_target=False)

    def onchange_showing(self, event):
        idx = self.data_index.index(self.now_showing_box.get())
        if (self.current_data == idx):
            return
        self.current_data = idx
        self.gui_update(file_update=None, settings_changed=False, recalculation=False, change_target=True)

    def onchange_analysis(self, event):
        idx = self.modes.index(self.analysis_box.get())
        if (self.current_mode == idx):
            return
        self.current_mode = idx
        self.gui_update(file_update=None, settings_changed=False, recalculation=False, change_target=True)


    def onchange_sensor(self, event):
        idx = self.sensors.index(self.sensor_box.get())
        if (self.current_sensor == idx):
            return
        self.current_sensor = idx
        self.gui_update(file_update=None, settings_changed=False, recalculation=False, change_target=True)


    # https://daeudaeu.com/tkinter-validation/
    def can_enter_as_number(self, diff):
        if (diff == "-" or diff.encode('utf-8').isdigit() or str((int(diff)*-1)).encode('utf-8').isdigit()):
            # 妥当（半角数字である）の場合はTrueを返却
            return True
        # 妥当でない（半角数字でない）場合はFalseを返却
        return False

    #パーセント表示する関数
    def change_progress(self,val):
        """
        Params
        val: str
            progress value
        """

        self.progress_bar_text.set(val)

    def copy_to_clipboard(self):
        self.clipboard_clear()
        txt = ""
        for i in range(2):
            for key in self.result_value_keys:
                txt += str(self.results[i][key][self.current_sensor][3])
                txt += "\n"
        for i in range(4):
            txt += str(self.results[-1]["coherence"][self.current_sensor][i])
            txt += "\n"
        self.clipboard_append(txt)
        print("copied to clipboard")

    def reset(self, is_init=False):
        if (not is_init):
            self.init_data()
        self.update_results()
        self.update_all_figure()
        """
        for data in range(2):
            entry_names = list(self.data_frames[data].children.keys())
            for key in range(len(self.result_value_keys)):

                # self.data_frames[data].children[key].text = str(self.results[data][self.result_value_keys[key]][self.current_mode][-1])
                self.data_frames[data].children[entry_names[key]].delete(0, "end")
                self.data_frames[data].children[entry_names[key]].insert(0,"None")
        for axis_idx in range(4):
            self.coherence_txts[axis_idx].delete(0, "end")
            self.coherence_txts[axis_idx].insert(0, "None")
        """

        self.seg_txt.delete(0, "end")
        self.seg_txt.insert(0,self.segment_duration_sec)
        self.samp_txt.delete(0, "end")
        self.samp_txt.insert(0, self.sampling_rate)
        self.range_txt1.delete(0, "end")
        self.range_txt1.insert(0, self.frame_range[0])
        self.range_txt2.delete(0, "end")
        self.range_txt2.insert(0, self.frame_range[1])
        self.warning_box.delete("1.0","end")





    def stft(self, x, fs, nperseg, segment_duration, noverlap=None):
        """
        Params:
        data: array
            signal input
        fs: integer/float
            sampling rate
        segment_duration: integer/float
            stft segment duration(sec)
        nperseg: integer
            sample number per segment
        ---
        return:
        s: ndarray
            stft result
        f: ndarray
            cyclical frequencies
        t: ndarray
            time instants
        """
        # print("----")
        # print(f"fs: {fs}")
        # print(f"nperseg: {nperseg}")
        # print(f"segment_duration: {segment_duration}")
        # print(f"noverlap: {noverlap}")
        # print("----")
        x_length = len(x)
        # print("data length: {}".format(x_length))

        L = np.min((x_length, nperseg))
        nTimesSpectrogram = 500;
        if (noverlap is None):
            noverlap = np.ceil(L - (x_length - L) / (nTimesSpectrogram - 1))
            noverlap = int(np.max((1,noverlap)))
        #nFFTMinimam = 2 ** 12
        nPad = np.max([2 ** int(np.ceil(np.log2(L))), 2 ** 12])
        # print("nPad: {}".format(nPad))
        # print("noverlap: ", noverlap)
        # hamming window を使用
        window = hamming(nperseg)
        sum_window = np.sum(window)

        # セグメントがいくつあるか
        seg = int(np.ceil((x_length - noverlap) / (nperseg - noverlap)))
        #print(seg)
        # print("----")
        # print(f"fs: {fs}")
        # print(f"nperseg: {nperseg}")
        # print(f"segment_duration: {segment_duration}")
        # print(f"noverlap: {noverlap}")
        # print(int(nperseg * seg - noverlap * (seg - 1) - x_length))
        # print("----")
        # データを nperseg, noverlap に合う長さになるようゼロ埋め
        data = np.append(x, np.zeros(int(nperseg * seg - noverlap * (seg - 1) - x_length)))
        # print("padded data length: {}".format(len(data)))

        result = np.empty((0, nPad))
        for iter in range(seg):
            #seg_data = data[(nperseg - noverlap) * iter : (nperseg - noverlap) * iter + nperseg]
            seg_data = data[(nperseg - noverlap) * iter : (nperseg - noverlap) * iter + nperseg] * window

            # ゼロ埋めの方法3パターン→結果は変わらない?
            # どまんなかにデータ
            seg_data = np.append(np.zeros((nPad - nperseg) // 2), seg_data)
            seg_data = np.append(seg_data, np.zeros(nPad - len(seg_data)))

            # result = np.append(result, [np.fft.fft(seg_data)] * window, axis=0)
            result = np.append(result, [np.fft.fft(seg_data)], axis=0)
        # print("spectrogram shape: {}".format(result.shape))

        # print(len(data) / fs - segment_duration / 2)
        sliced_result = result.T[int(nPad / fs * self.min_f):int(nPad / fs * self.max_f), :] * 2 / sum_window
        if (x_length - len(data)) < 0:
            t = np.linspace(segment_duration / 2, len(data) / fs - segment_duration / 2, result.shape[0] + (len(data) - x_length))[0:x_length - len(data)]
        else:
            t = np.linspace(segment_duration / 2, len(data) / fs - segment_duration / 2, result.shape[0] + (len(data) - x_length))

        f = np.linspace(self.min_f, self.max_f, np.shape(sliced_result)[0])

        return sliced_result, f, t


    def spectrogram_analize(self, data_idx, sensor_idx, data_i, fs, nperseg, filename, sensor, start=0, end=-1):
        """
        Params
        data: array(3, n)
            x, y, z data
        fs: int/float
            sampling rate
        nperseg: int
            sample number per stft segment
        filename: str
            filename
        sensor: str
            sensor name
        start: integer
            analysis start frame
        end: integer
            analysis end frame
            -1 means end of input data
        """
        if (not len(data_i[0]) == len(data_i[1]) == len(data_i[2])):
            print("invalid input data")
            return None, None, None
        if (end == -1):
            end = len(data_i[0]) - 1
        elif (start > end):
            print("invalid range setting")
            # sg.Popup("invalid range setting")
            return None, None, None

        data = data_i[:, start: end + 1]

        # print("nperseg: {}".format(nperseg))


        specs = []
        x_length = len(data[0])
        nTimesSpectrogram = 500;
        L = np.min((x_length, nperseg))
        noverlap = np.ceil(L - (x_length - L) / (nTimesSpectrogram - 1))
        noverlap = int(np.max((1,noverlap)))
        for i in range(3):
            # start = time.time()

            # scipy
            f, t, spec = spectrogram(detrend(data[i]), fs, window=get_window("hamming", int(nperseg)), nperseg=int(nperseg), noverlap=noverlap, nfft=2**12, mode="magnitude", )


            # plt
            # spec, f, t, _ = pltspectrogram(detrend(data[i]), Fs=fs, pad_to=int(nperseg), noverlap=noverlap, NFFT=2**12, mode="default", scale="linear")
            # spec = np.sqrt(np.array(spec))

            # self-created
            # spec, f, t = np.abs(self.stft(detrend(data[i]), fs, int(nperseg), self.segment_duration_sec))
            # elapsed_time = time.time() - start
            # print ("elapsed_time:\n{0}".format(elapsed_time))

            specs.append(np.abs(spec))
        # convert to 3-dimensional ndarray
        specs = np.array(specs) #specs.shape: (3, 640, 527)

        # trim into frequency range
        f_range = np.array([self.min_f, self.max_f]) * len(f) * 2 // self.sampling_rate
        specs = specs[:, f_range[0]: f_range[1] , :]
        f = f[f_range[0]: f_range[1]]

        vmin = np.min(specs)
        vmax = np.max(specs)
        # add norm
        specs = np.append(specs, [np.linalg.norm(specs, axis=0)], axis=0)



        for i in range(3):
            self.results[data_idx]["sp_graph"][sensor_idx][i], ax = plt.subplots(figsize=self.figsize_small, dpi=100)
            self.results[data_idx]["sp_graph"][sensor_idx][i].subplots_adjust(left=0.2)
            im = ax.pcolormesh(t, f, specs[i], cmap="jet", vmin=vmin, vmax=vmax)
            ax.set_xlabel("Time [sec]")
            ax.set_ylabel("Frequency [Hz]")
            titles = ["x","y","z"]
            ax.set_title(titles[i])
            # axpos = axes.get_position()
            # cbar_ax = fig.add_axes([0.87, axpos.y0, 0.02, axpos.height])
            #cbar = self.results[data_idx]["sp_graph"][sensor_idx][i].colorbar(im,ax=ax)
            #cbar.set_label("Amplitude")
            # plt.show()
            # input("aa")

        self.results[data_idx]["sp_graph"][sensor_idx][3], ax = plt.subplots(figsize=self.figsize_large, dpi=100)
        im = ax.pcolormesh(t, f, specs[3], cmap="jet", vmin=vmin, vmax=vmax)
        ax.set_title("Norm")
        ax.set_xlabel("Time [sec]")
        ax.set_ylabel("Frequency [Hz]")

        # axpos = axes.get_position()
        # cbar_ax = fig.add_axes([0.87, axpos.y0, 0.02, axpos.height])
        cbar = self.results[data_idx]["sp_graph"][sensor_idx][3].colorbar(im,ax=ax)
        cbar.set_label("Amplitude")

        recording = len(data[0]) / fs
        #f_offset = int(specs.shape[2] * 2 / 20)
        # print("s t {} {}".format(t[0], t[-1]))


        peak_amp = np.max(specs[3, :, :])
        peak_idx = np.where(specs[3] == peak_amp)
        peak_freq = f[peak_idx[0][0]]
        peak_time = t[peak_idx[1][0]]

        # print("recording(s): {}".format(recording))
        # print("peak amplitude: {}  {}".format(peak_amp, peak_idx))
        # print("peak frequency(Hz): {}".format(peak_freq))
        # print("peaktime(s): {}".format(peak_time))


        self.results[data_idx]["sp_peak_amplitude"][sensor_idx][3] = peak_amp
        self.results[data_idx]["sp_peak_frequency"][sensor_idx][3] = peak_freq
        self.results[data_idx]["sp_peak_time"][sensor_idx][3] = peak_time

        plt.close("all")
        return peak_amp, peak_freq, peak_time

    def power_density_analize(self, data_idx, sensor_idx, data_i, fs, nperseg, filename, sensor, start=0, end=-1):
        """
        Params
        data: array(3, n)
            x, y, z data
        fs: int/float
            sampling rate
        nperseg: int
            sample number per stft segment
        filename: str
            filename
        sensor: str
            sensor name
        start: integer
            analysis start frame
        end: integer
            analysis end frame
            -1 means end of input data
        """

        if (not len(data_i[0]) == len(data_i[1]) == len(data_i[2])):
            print("invalid input data")
            return None, None, None
        if (end == -1):
            end = len(data_i[0]) - 1
        elif (start > end):
            # sg.Popup("invalid range setting")
            print("invalid range setting")
            return None, None, None

        data = data_i[:, start: end + 1]
        # print("nperseg: {}".format(nperseg))
        # print(data.shape)

        specs = []
        for i in range(3):
            #spec, f, t = self.stft(detrend(data[i]), fs, int(nperseg), self.segment_duration_sec, int(nperseg * 0.75))
            f, t, spec = spectrogram(detrend(data[i]), fs, window=get_window("hamming", int(nperseg)), nperseg=int(nperseg), noverlap=int(nperseg * 0.75), nfft=2**12, mode="complex", ) # scipy
            # spec, f, t, _ = pltspectrogram(detrend(data[i]), Fs=fs, pad_to=int(nperseg), noverlap=int(nperseg * 0.75), NFFT=2**12, mode="magnitude", scale="linear") #plt
            specs.append(np.sum(np.power(np.abs(spec), 1), axis=1) / (len(t)))

        # convert to 3-dimensional ndarray
        specs = np.array(specs) #specs.shape: (3, 640)

        # trim into frequency range
        f_range = np.array([self.min_f, self.max_f]) * len(f) * 2 // self.sampling_rate
        specs = specs[:, f_range[0]: f_range[1]]
        f = f[f_range[0]: f_range[1]]

        #specs /= np.sum(np.power(signal.tukey(int(nperseg)), 2)) / np.power(np.sum(signal.tukey(int(nperseg))), 2)
        vmin = np.min(specs)
        vmax = np.max(specs)

        # add norm
        specs = np.append(specs, [np.linalg.norm(specs, axis=0)], axis=0)
        for i in range(3):
            self.results[data_idx]["sa_graph"][sensor_idx][i], ax = plt.subplots(figsize=self.figsize_small, dpi=100)
            self.results[data_idx]["sa_graph"][sensor_idx][3].subplots_adjust(left=0.3)
            ax.set_ylim(0, vmax * 1.2)
            ax.plot(f, specs[i])
            ax.set_xlabel("Frequency [Hz]")
            ax.set_ylabel("Amplitude")
            titles = ["x","y","z"]
            ax.set_title(titles[i])

        vmax = np.max(specs[-1])
        self.results[data_idx]["sa_graph"][sensor_idx][3], ax = plt.subplots(figsize=self.figsize_large, dpi=100)
        ax.set_ylim(0, vmax * 1.2)
        ax.plot(f, specs[3])
        ax.set_title("Norm")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Amplitude")

        is_estimated, l, u, lv, uv, hwp = self.full_width_half_maximum(data_idx, sensor_idx, f, specs[3])
        if (uv is None and lv is None):
            fwhm = "None"
        else:
            fwhm = uv - lv
        # print(l, u, lv, uv)
        # print(specs[3, int(l)])
        if (l is not None and u is not None):
            ax.fill_between(f[l:u], specs[3, l:u], color="r", alpha=0.5)
        #plt.show()
        #### plt.savefig(data_dir + "/" + remove_ext(filename) + "norm" + sensor + "am.png")
        recording = len(data[0]) / fs

        peak_amp = np.max(specs[3])
        peak_idx = np.where(specs[3] == peak_amp)
        peak_freq = f[peak_idx[0][0]]
        tsi = self.tremor_stability_index(data_idx, sensor_idx, data, fs)



        # print("recording(s): {}".format(recording))
        # print("peak amplitude: {}  {}".format(peak_amp, peak_idx))
        # print("peak frequency(Hz): {}".format(peak_freq))
        # print("Full-width Half Maximum(Hz): {}".format(fwhm))
        # print("Half-width power: {}".format(hwp))
        # print("Tremor Stability Index: {}".format(tsi))



        self.results[data_idx]["sa_peak_amplitude"][sensor_idx][3] = peak_amp
        self.results[data_idx]["sa_peak_frequency"][sensor_idx][3] = peak_freq
        if (is_estimated):
            self.results[data_idx]["sa_fwhm"][sensor_idx][3] = str(fwhm) + "(estimated)"
            self.results[data_idx]["sa_hwp"][sensor_idx][3] = str(hwp) + "(estimated)"
        else:
            self.results[data_idx]["sa_fwhm"][sensor_idx][3] = fwhm
            self.results[data_idx]["sa_hwp"][sensor_idx][3] = hwp
        self.results[data_idx]["sa_tsi"][sensor_idx][3] = tsi
        plt.close("all")
        return peak_amp, peak_freq, fwhm, hwp, tsi


    def full_width_half_maximum(self, data_idx, sensor_idx, x, y):
        """
        calcurate Full-width Half Maximum and Half-witdh power

        Params
        x: array-like
        y: array-like

        Retuerns
        is_estimated: bool
            whether estimation value is used
        lower: int
            lower limit index
        upper: int
            upper limit index
        lower_v: int/float
            lower limit value (approximate)
        upper_v: int/float
            upper limit value (approximate)
        hwp: int/float
            Half-width power
        """
        y_ndarray = np.array(y)
        length = len(y_ndarray)
        peak_val_half = np.max(y_ndarray) / 2
        peak_idx = y_ndarray.argmax()
        # print(peak_idx)
        lower = peak_idx
        upper = peak_idx
        d = np.abs(x[1] - x[0])
        is_estimated = False

        while (lower > 0 and y_ndarray[lower] > peak_val_half):
            lower -= 1
        if (y_ndarray[lower] != peak_val_half and lower != 0):
            lower_v = x[lower] + d * (peak_val_half - y_ndarray[lower]) / (y_ndarray[lower + 1] - y_ndarray[lower]) # linear interpolation
        else:
            lower_v = x[lower]

        while (upper < length - 1 and y_ndarray[upper] > peak_val_half):
            upper += 1
        if (y_ndarray[upper] != peak_val_half and upper != length - 1):
            upper_v = x[upper] - d * (peak_val_half - y_ndarray[upper]) / (y_ndarray[upper -1] - y_ndarray[upper]) # linear interpolation
        else:
            upper_v = x[upper]

        if (lower == 0 and upper == length -1):
            return (False, None, None, None, None, None)

        # judge whether estimation value is used.
        if (lower == 0):
            is_estimated = True
            upper_v= x[upper]
            lower_v = x[peak_idx] - (x[upper] - x[peak_idx])
            hwp = np.sum(y_ndarray[peak_idx: upper]) * d * 2
        elif(upper == length -1):
            is_estimated = True
            lower_v = x[lower]
            upper_v = x[peak_idx] + (x[peak_idx] - x[lower])
            hwp = np.sum(y_ndarray[lower: peak_idx]) * d * 2
        else:
            # not estimated
            hwp = np.sum(y_ndarray[lower: upper]) * d



        return (is_estimated, lower, upper, lower_v, upper_v, hwp)

    def tremor_stability_index(self, data_idx, sensor_idx, data, fs):
        """
        Tremor Stability Index

        Params
        data: array-like
            data
        fs: int/float
            sampling rate
        """
        # highpass filter
        sos = butter(N=3, Wn=0.1, btype="highpass", fs=fs, output='sos')
        data = sosfilt(sos, data, axis=0)

        # principal component analysis
        pca = PCA(n_components=1)
        x = np.ravel(pca.fit_transform(detrend(data).T))
        length = len(x)

        nperseg = self.sampling_rate * self.segment_duration_sec
        nTimesSpectrogram = 500;
        L = np.min((length, nperseg))
        noverlap = np.ceil(L - (length - L) / (nTimesSpectrogram - 1))
        noverlap = int(np.max((1,noverlap)))
        freq, _, spec = spectrogram(detrend(x), fs, window=get_window("hamming", int(nperseg)), nperseg=int(nperseg), noverlap=int(noverlap), nfft=2**12, mode="complex", )
        max_freq = freq[np.unravel_index(np.argmax(spec, axis=None), spec.shape)[0]]
        spec = np.abs(spec)

        if (max_freq < 2):
            max_freq = 2.001 # to create bandpass filter, max_freq - 2 maust be larger than 0
        elif (max_freq > 9):
            max_freq = 9

        sos = butter(N=3, Wn=(max_freq - 2, max_freq + 2), btype="bandpass", fs=fs, output='sos')
        x= sosfilt(sos, x, axis=0)

        idx = 1
        zero_crossing = np.empty(0)
        while (idx < length):
            if (x[idx - 1] < 0 and x[idx] >= 0):
                zero_crossing = np.append(zero_crossing, idx)
            idx += 1

        f = fs / np.diff(np.array(zero_crossing))
        delta_f = np.diff(f)
        if (len(delta_f) == 0):
            q75, q25 = 0, 0
        else:
            q75, q25 = np.percentile(delta_f, [75, 25], interpolation="nearest")

        # tsi
        return q75 - q25

    def ft_coherence(self, sensor_idx, axis_idx, data1, data2, fs, start=0, end=-1):
        if (len(data1) != len(data2)):
            # sg.Popup("data1 and data2 have different lengths")
            print("data1 and data2 have different lengths")
            return None

        if (end == -1):
            end = len(data1) - 1
        elif (start > end):
            # sg.Popup("invalid range setting")
            print("invalid range setting")
            return None

        x1 = data1[start: end + 1]
        x2 = data2[start: end + 1]

        nfft = 2 ** 8
        noverlap = 2 ** 7
        Cyx, f = cohere(x2, x1, NFFT=nfft, Fs=fs,
                    window=window_hanning, noverlap=noverlap)
        FREQ_LOW = 2
        FREQ_HIGH = 12
        xlim_l = FREQ_LOW
        xlim_r = FREQ_HIGH
        idx = [int(len(f) / f[-1] * xlim_l), int(len(f) / f[-1] * xlim_r)]
        f = f[idx[0] : idx[1]]
        df = f[1] - f[0]
        Cyx = Cyx[idx[0] : idx[1]]

        l = (len(x1) - noverlap) // (nfft - noverlap)
        z = 1 - np.power(0.05, 1 / (l - 1))
        # print("z: ", z)
        # print("significant points rate: ", len(Cyx[Cyx >= z]) / len(Cyx)) # 有意な値の割合
        Cyx = Cyx[Cyx >= z]
        # print(Cyx)
        coh = np.sum(Cyx) * df
        # print("coherence: ", coh)

        self.results[-1]["coherence"][sensor_idx][axis_idx] = coh
        return coh

    def wavelet_analize(self, data_idx, sensor_idx, data_i, fs, nperseg, filename, sensor, start=0, end=-1):
        """
        Params
        data: array(3, n)
            x, y, z data
        fs: int/float
            sampling rate
        nperseg: int
            sample number per stft segment
        filename: str
            filename
        sensor: str
            sensor name
        start: integer
            analysis start frame
        end: integer
            analysis end frame
            -1 means end of input data
        """
        if (not len(data_i[0]) == len(data_i[1]) == len(data_i[2])):
            print("invalid input data")
            return None, None, None
        if (end == -1):
            end = len(data_i[0]) - 1
        elif (start > end):
            print("invalid range setting")
            # sg.Popup("invalid range setting")
            return None, None, None

        data = data_i[:, start: end + 1]

        # print("nperseg: {}".format(nperseg))

        t, dt = np.linspace(0, len(data[0]) // fs, len(data[0]), retstep=True)
        fs = 1/dt
        w = 6. # morlet parameter: being 6 to satisfy the admissibility condition (Farge 1992)
        max_freq = 20
        steps = 100
        freq = np.linspace(0, max_freq, steps + 1)[1:] # avoid zero division
        widths = w*fs / (2*freq*np.pi)

        specs = []
        for i in range(3):
            cwtm = cwt(np.abs(data[i]), morlet2, widths, w=w)
            specs.append(cwtm)

        # convert to 3-dimensional ndarray
        specs = np.array(specs) #specs.shape: (3, 640, 527)
        vmin = np.min(np.abs(specs))
        vmax = np.max(np.abs(specs))
        # add norm
        specs = np.append(specs, [np.linalg.norm(specs, axis=0)], axis=0)

        titles = ["x","y","z"]
        for i in range(3):
            self.results[data_idx]["wavelet"][sensor_idx][i], ax = plt.subplots(figsize=self.figsize_small, dpi=100)
            self.results[data_idx]["wavelet"][sensor_idx][i].subplots_adjust(left=0.2)
            im = ax.pcolormesh(t, freq, np.abs(specs[i]), cmap="jet", vmin=vmin, vmax=vmax)
            ax.set_xlabel("Time [sec]")
            ax.set_ylabel("Frequency [Hz]")
            ax.set_title(titles[i])
            # axpos = axes.get_position()
            # cbar_ax = fig.add_axes([0.87, axpos.y0, 0.02, axpos.height])
            #cbar = self.results[data_idx]["sp_graph"][sensor_idx][i].colorbar(im,ax=ax)
            #cbar.set_label("Amplitude")
            # plt.show()
            # input("aa")

        self.results[data_idx]["wavelet"][sensor_idx][3], ax = plt.subplots(figsize=self.figsize_large, dpi=100)
        im = ax.pcolormesh(t, freq, np.abs(specs[3]), cmap="jet", vmin=vmin, vmax=vmax)
        ax.set_title("Norm")
        ax.set_xlabel("Time [sec]")
        ax.set_ylabel("Frequency [Hz]")

        # axpos = axes.get_position()
        # cbar_ax = fig.add_axes([0.87, axpos.y0, 0.02, axpos.height])
        cbar = self.results[data_idx]["wavelet"][sensor_idx][3].colorbar(im,ax=ax)
        cbar.set_label("Amplitude")


        f_offset = int(specs.shape[1] * 2 / 20)

        # print("s t {} {}".format(t[0], t[-1]))
        peak_amp = np.max(np.abs(specs[3, f_offset:, :]))
        peak_idx = np.where(np.abs(specs[3]) == peak_amp)
        peak_freq = freq[peak_idx[0][0]]
        peak_time = t[peak_idx[1][0]]

        # print("recording(s): {}".format(recording))
        # print("peak amplitude: {}  {}".format(peak_amp, peak_idx))
        # print("peak frequency(Hz): {}".format(peak_freq))
        # print("peaktime(s): {}".format(peak_time))


        self.results[data_idx]["wt_peak_amplitude"][sensor_idx][3] = peak_amp
        self.results[data_idx]["wt_peak_frequency"][sensor_idx][3] = peak_freq
        self.results[data_idx]["wt_peak_time"][sensor_idx][3] = peak_time

        # print(f"wavelet result\npeak amp: {peak_amp}\npeak freq: {peak_freq}\npeak time: {peak_time}")
        plt.close("all")
        return peak_amp, peak_freq, peak_time

    def wt_coherence(self, data1_i, data2_i, fs,  start=0, end=-1):
        """
        wavelet coherence
        """

        if (end == -1):
            end = len(data1_i[0]) - 1
        elif (start > end):
            print("invalid range setting")
            # sg.Popup("invalid range setting")
            return None, None, None

        data = [data1_i[:, start: end + 1], data2_i[:, start: end + 1]]
        # print("nperseg: {}".format(nperseg))
        spec_2 = []

        t, dt = np.linspace(0, len(data[0][0]) // fs, len(data[0][0]), retstep=True)
        fs = 1/dt
        w = 6. # morlet parameter: being 6 to satisfy the admissibility condition (Farge 1992)
        max_freq = 20
        steps = 100
        freq = np.linspace(0, max_freq, steps + 1)[1:] # avoid zero division
        widths = w*fs / (2*freq*np.pi)
        for i in range(2):
            specs = []

            for j in range(3):
                cwtm = cwt(np.abs(data[i][j]), morlet2, widths, w=w)
                specs.append(cwtm)

            # convert to 3-dimensional ndarray
            specs = np.array(specs) #specs.shape: (3, 640, 527)
            vmin = np.min(np.abs(specs))
            vmax = np.max(np.abs(specs))
            # add norm
            specs = np.append(specs, [np.linalg.norm(specs, axis=0)], axis=0)
            spec_2.append(specs)
        spec_2 = np.array(spec_2)

        # specs[axis][freq][time(sample)]
        spec_2_conj = []
        spec_2_conj.append(np.conj(spec_2[0]))
        spec_2_conj.append(np.conj(spec_2[1]))
        spec_2_conj = np.array(spec_2_conj)

        cross_specs = np.average(spec_2[0] * spec_2_conj[1], axis=2)
        power_specs = np.average(spec_2 * spec_2_conj, axis=3)
        coh = np.power(np.abs(cross_specs), 2) / (power_specs[0] * power_specs[1])

        # fig ,ax = plt.subplots()
        # ax.plot(coh[0])
        # fig.savefig("coh.png")


class FigureNavigator(NavigationToolbar2Tk):
    # override to stop displaying mouse coordinate
    def mouse_move(self, event):
        self._set_cursor(event)

        # if event.inaxes and event.inaxes.get_navigate():
        if False:

            try:
                s = event.inaxes.format_coord(event.xdata, event.ydata)
            except (ValueError, OverflowError):
                pass
            else:
                artists = [a for a in event.inaxes._mouseover_set
                           if a.contains(event)[0] and a.get_visible()]

                if artists:
                    a = cbook._topmost_artist(artists)
                    if a is not event.inaxes.patch:
                        data = a.get_cursor_data(event)
                        if data is not None:
                            data_str = a.format_cursor_data(data)
                            if data_str is not None:
                                s = s + ' ' + data_str

                if len(self.mode):
                    self.set_message('%s, %s' % (self.mode, s))
                else:
                    self.set_message(s)
        else:
            self.set_message(self.mode)
if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
