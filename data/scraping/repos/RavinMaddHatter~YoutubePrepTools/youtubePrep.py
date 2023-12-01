import time

t1 = time.time()
from time import sleep
from threading import Thread
import cutter
import openai_translator as Translator
from queue import Queue
from csv import DictReader
from pyperclip import copy
from json import dump, load
from tkinter import Tk, Label, Button, INSERT, Scale, IntVar, Checkbutton, END
from tkinter import filedialog, Entry, DoubleVar, ttk, Toplevel, StringVar, OptionMenu
from os.path import exists, split, join, getmtime
from tkinter.scrolledtext import ScrolledText
from pathlib import Path
from glob import glob

print(time.time() - t1)
data = None
BoilerplateInfo = None
slider_defaults = None
sliders_enabled = None
audioChans = 6
translator = Translator.Translator


def update_save(data, file_name="default_profile.json"):
    with open(file_name, "w+") as file:
        dump(data, file, indent=2)


def load_file(file_name="default_profile.json"):
    print(file_name)
    data = {}

    if exists(file_name):
        with open(file_name) as file:
            data = load(file)

    if "boilerplate" not in data.keys():
        data["boilerplate"] = "Default Test For Your Youtube Description/n"

    if "slider_defaults" not in data.keys():
        data["sliders_enabled"] = []
        data["slider_defaults"] = []

        for i in range(audioChans):
            data["slider_defaults"].append(-24)
            data["sliders_enabled"].append(False)

        data["sliders_enabled"][0] = True

    if "model" not in data.keys():
        data["model"] = "base"

    if "in_space" not in data.keys():
        data["in_space"] = 0.1

    if "out_space" not in data.keys():
        data["out_space"] = 0.1

    if "min_clip" not in data.keys():
        data["min_clip"] = 1

    if "min_silent" not in data.keys():
        data["min_silent"] = 0.1

    if "min_silent" not in data.keys():
        data["min_silent"] = 0.1

    update_save(data)

    return data


data = load_file()


class Markerprocessor:
    def __init__(self, file):
        self.markers = []

        with open(file, newline='') as csvfile:
            reader = DictReader(csvfile, delimiter=',')
            for row in reader:
                time = row["Source In"].split(":")
                time[0] = int(time[0]) - 1
                if time[0] == 0:
                    time.pop(0)
                else:
                    time[0] = "{:02d}".format(time[0])
                time.pop()
                time = ":".join(time)

                self.markers.append(time + " " + row["Notes"])

    def string_to_clipboard(self):
        copy(data["boilerplate"] + "\n\r\n\rChapters: \n\r" + "\n\r".join(self.markers))

    def string_to_file(self, name):
        with open(name, "w+") as text_file:
            text_file.write("\n\r".join(self.markers))


if __name__ == "__main__":
    def progress_bar(operation_name, update_queue):
        popup = Toplevel(height=100, width=500)
        status_text = StringVar()
        popup_description = Label(popup, textvariable=status_text)
        popup_description.grid(row=0, column=0)
        progress_var = DoubleVar()
        progress_bar = ttk.Progressbar(popup, variable=progress_var, maximum=100)
        progress_bar.grid(row=1, column=0)
        complete = False

        while not complete:
            sleep(0.01)
            if not update_queue.empty():
                update = update_queue.get()
                progress_var.set(update["percent"])
                status_text.set(update["state"])
                popup.update()
                popup.focus_force()
                complete = (update["state"] == "done")

        popup.destroy()
        popup.update()


    def find_csv():
        filename = filedialog.askopenfilename(title="Select a CSV File",
                                              filetypes=(("CSV files",
                                                          "*.CSV*"),
                                                         ("all files",
                                                          "*.*")))

        data["boilerplate"] = st.get("1.0", END)
        mk = Markerprocessor(filename)
        mk.string_to_clipboard()
        print("markers in clipboard")

    def transcribe_process(transcribe_queue, filename):
        print("setting moddel")
        trans = translator(transcribe_queue, selected_model.get())
        print("running model")
        trans.audio_to_text(filename)
        print("finished")


    def transcribe_vid():
        filename = filedialog.askopenfilename(title="Select a Media File File",
                                              filetypes=(("Media Files",
                                                          "*.WAV *.MP4 *.MOV *.AVI *.Y4M *.MKV"),
                                                         ("all files",
                                                          "*.*")))
        try:
            transcribe_queue = Queue()
            print("queue sent")
            popup = Thread(target=progress_bar, args=("Transcribing Video", transcribe_queue,))
            popup.start()

            trans = Thread(target=transcribe_process, args=(transcribe_queue, filename,))
            trans.start()
            print("transcribe finished")
        except Exception as e:
            print("failed translation")
            print(e)


    def do_settings(cc):
        levels = []
        chans = []
        for i in range(len(sliders)):
            levels.append(-sliders[i].get())
            chans.append(slider_chks[i].get() == 1)
        cc.set_multi_chan_thres(levels)
        cc.set_lead_in(lead_in.get())
        cc.set_lead_out(lead_out.get())
        cc.set_min_clip_dur(clip_dur.get())
        cc.set_enabled_tracks(chans)
        cc.set_min_silent_dur(min_silent_dur_var.get())


    def cut_clip_process(queue, video_file):
        name = Path(video_file).stem
        head, tail = split(video_file)
        cc = cutter.Clipcutter(queue)
        try:
            do_settings(cc)
            cc.add_cut_video_to_timeline(video_file)
            cc.export_edl(join(head, name + "-cut.edl"))
            cc._cleanup()
        except Exception as e:
            print(e)
            cc._cleanup()


    def cut_clip():
        video_file = filedialog.askopenfilename(title="Select a WAV File",
                                                filetypes=(("video files",
                                                            "*.mkv*"),
                                                           ("all files",
                                                            "*.*")))

        cut_queue = Queue()
        popup = Thread(target=progress_bar, args=("Cutting Video", cut_queue,))
        popup.start()
        trans = Thread(target=cut_clip_process, args=(cut_queue, video_file,))
        trans.start()


    def cut_folder_process(queue, folder):
        cc = cutter.Clipcutter(queue)
        try:
            name = split(folder)[-1]
            do_settings(cc)
            files = glob(join(folder, "*.mkv"))
            files.sort(key=getmtime)
            for file in files:
                print(file)
                cc.add_cut_video_to_timeline(file)
            print(join(folder, (name + "-cut.edl")))
            cc.export_edl(join(folder, (name + "-cut.edl")))
            cc._cleanup()
        except Exception as e:
            print(e)
            cc._cleanup()


    def cut_folder():
        folder = filedialog.askdirectory()
        cut_queue = Queue()
        popup = Thread(target=progress_bar, args=("Cutting Video", cut_queue,))
        popup.start()
        trans = Thread(target=cut_folder_process, args=(cut_queue, folder,))
        trans.start()


    def set_save_data():
        for i in range(audioChans):
            data["slider_defaults"][i] = sliders[i].get()
            data["sliders_enabled"][i] = slider_chks[i].get()
        data["boilerplate"] = st.get("1.0", END)
        data["model"] = selected_model.get()
        data["in_space"] = lead_in.get()
        data["out_space"] = lead_out.get()
        data["min_clip"] = clip_dur.get()
        data["min_silent"] = min_silent_dur_var.get()


    def load_profile():
        settings_file = filedialog.askopenfilename(title="Select a profile",
                                                   filetypes=(("json files",
                                                               "*.json*"),
                                                              ("all files",
                                                               "*.*")))
        load_file(settings_file)
        for i in range(audioChans):
            sliders[i].set(data["slider_defaults"][i])
            slider_chks[i].set(data["sliders_enabled"][i])
        st.delete('1.0', END)
        st.insert(INSERT, data["boilerplate"])
        selected_model.set(data["model"])
        lead_in.set(data["in_space"])
        lead_out.set(data["out_space"])
        clip_dur.set(data["min_clip"])
        min_silent_dur_var.set(data["min_silent"])


    def save_as():
        file_name = filedialog.asksaveasfile(title="Set Profile File Name",
                                             filetypes=(("JSON",
                                                         "*.json*"),)).name
        if not (file_name.endswith(".json") or file_name.endswith(".json")):
            file_name += ".json"
        set_save_data()
        update_save(data, file_name=file_name)


    def save():
        set_save_data()
        update_save(data)


    def exit():
        window.destroy()


    window = Tk()
    window.title('Youtube Video Publishing Tools')
    label_file_explorer = Label(window,
                                text="Video Prep Tools",
                                width=20, height=2)
    csvButton = Button(window,
                       text="Markers to Clipboard",
                       command=find_csv,
                       width=20)
    waveButton = Button(window,
                        text="Transcribe Media",
                        command=transcribe_vid,
                        width=20)
    cut_button = Button(window,
                        text="Cut Clip",
                        command=cut_clip,
                        width=20)
    super_cut_button = Button(window,
                              text="Cut Folder",
                              command=cut_folder,
                              width=20)
    button_exit = Button(window,
                         text="Exit",
                         command=exit,
                         width=20)

    button_save = Button(window,
                         text="Save Default",
                         command=save,
                         width=20)
    button_save_as = Button(window,
                            text="Save as",
                            command=save_as,
                            width=20)
    button_load = Button(window,
                         text="Load",
                         command=load_profile,
                         width=20)
    lbl_entry = Label(window,
                      text="Description Tools",
                      width=50, height=2)
    st = ScrolledText(window, width=75, height=5, relief="raised")

    st.insert(INSERT, data["boilerplate"])
    options = list(Translator._MODELS.keys())
    model_label = Label(window, text="Speach Model Size", width=15, height=2)
    selected_model = StringVar()
    selected_model.set(data["model"])
    model_select = OptionMenu(window, selected_model, *options)
    sliders = []
    sliders_lb = []
    sliders_ch = []
    slider_chks = []

    for i in range(audioChans):
        sliders_lb.append(Label(window,
                                text="ch {}".format(i + 1),
                                height=2))
        sliders.append(Scale(window, from_=0, to=-50))
        sliders[i].set(data["slider_defaults"][i])
        slider_chks.append(IntVar())
        slider_chks[i].set(data["sliders_enabled"][i])
        sliders_ch.append(Checkbutton(window, variable=slider_chks[i]))
    slider_chks[0].set(1)
    lead_in = DoubleVar()

    ld_in_ent = Entry(window, textvariable=lead_in, width=10)
    in_lb = Label(window, text="In Space", width=15, height=2)
    lead_out = DoubleVar()

    ld_out_ent = Entry(window, textvariable=lead_out, width=10)
    out_lb = Label(window, text="Out Space", width=15, height=2)
    clip_dur = DoubleVar()
    clip_dur_ent = Entry(window, textvariable=clip_dur, width=10)
    dur_lb = Label(window, text="Min Clip Length", width=15, height=2)
    min_silent_dur_var = DoubleVar()
    min_silent_dur_ent = Entry(window, textvariable=min_silent_dur_var, width=10)
    silent_lb = Label(window, text="Min Silent Dur", width=15, height=2)
    lead_in.set(data["in_space"])
    lead_out.set(data["out_space"])
    clip_dur.set(data["min_clip"])
    min_silent_dur_var.set(data["min_silent"])
    audio_lb = Label(window, text="Audio Tools", width=15, height=2)
    row = 1
    label_file_explorer.grid(column=1, row=row, columnspan=audioChans)
    row += 1

    cut_button.grid(column=0, row=row, columnspan=3)
    super_cut_button.grid(column=3, row=row, columnspan=3)
    row += 1
    for i in range(len(sliders)):
        sliders_lb[i].grid(column=i + 1, row=row)
        sliders[i].grid(column=i + 1, row=row + 1)
        sliders_ch[i].grid(column=i + 1, row=row + 2)
    row += 3
    in_lb.grid(column=1, row=row)
    out_lb.grid(column=2, row=row)
    dur_lb.grid(column=3, row=row)
    silent_lb.grid(column=4, row=row)
    row += 1
    ld_in_ent.grid(column=1, row=row)
    ld_out_ent.grid(column=2, row=row)
    clip_dur_ent.grid(column=3, row=row)
    min_silent_dur_ent.grid(column=4, row=row)
    row += 1

    audio_lb.grid(column=1, row=row, columnspan=6)
    row += 1
    model_label.grid(column=0, row=row, columnspan=2)
    model_select.grid(column=2, row=row, columnspan=1)
    waveButton.grid(column=3, row=row, columnspan=3)
    row += 1
    lbl_entry.grid(column=1, row=row, columnspan=audioChans)
    row += 1
    st.grid(column=1, row=row, columnspan=audioChans)
    row += 1
    csvButton.grid(column=1, row=row, columnspan=audioChans)
    row += 1
    button_save.grid(column=1, row=row)
    button_save_as.grid(column=2, row=row)
    button_load.grid(column=3, row=row)
    button_exit.grid(column=4, row=row, columnspan=audioChans - 1)
    window.mainloop()
