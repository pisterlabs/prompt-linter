# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 12:52:50 2018

@author: 64942
"""

from scipy import signal
import mne
import numpy as np
import csv
import os
import check_file
from coherence_code import eeg_coherence_anova
from coherence_code import eeg_coherence_anova_plot
from coherence_code import eeg_coherence_plot
from coherence_code import eeg_coherence_plot_difference


def coh_get_Section(f, a, b):
    start = -1
    end = -1
    for i in range(0, len(f)):
        if start == -1 and f[i] > a:
            start = i
        if end == -1 and f[i] > b:
            end = i
            break
    return start, end


start = -1
end = -1


def coh_get_channel_names():
    raw = mne.io.read_raw_brainvision('eegData/health_control/eyeclose/jkdz_cc_20180430_close.vhdr',
                                      preload=True)
    channel_names = []
    for i in raw.info['ch_names']:
        if i != 'Oz':
            if i != 'ECG':
                channel_names.append(i)
    return channel_names[0:-1]


def coherence(x, y, a, b):
    fs = 48
    f, Cxy = signal.coherence(x, y, fs, nperseg=20)
    global start
    global end
    if start == -1:
        start, end = coh_get_Section(f, a, b)
    return (np.mean(Cxy[start:end]))


def eeg_coherence(raw):
    fileread = open('eeg_coherence.csv', 'w', newline='')
    writer = csv.writer(fileread)
    writer.writerow(coh_get_channel_names())
    for i in range(0, 62):
        data = []
        for j in range(0, 62):
            print('channel: ', i, ' ', j)
            data += [coherence(raw[61 - i][0][0], raw[j][0][0], 8, 12)]
        writer.writerow(data)
    fileread.close


def coherence_ofList(raw_list, flag):
    print(raw_list)
    columns = coh_get_channel_names()
    if flag == 'c':
        fileread = open('eeg_coherence_c.csv', 'w', newline='')
        writer = csv.writer(fileread)
        writer.writerow(columns)
        for i in range(0, 62):
            data = []
            for j in range(0, 62):
                t = []
                if (61 - i) <= j:
                    tempread = open('eeg_coh_anova/' + str(columns[61 - i]) + '_' + str(columns[j]) + '_c' + '.csv',
                                    'w', newline='')
                    tempwriter = csv.writer(tempread)
                    tempwriter.writerow(['id', 'coherence'])
                for raw in raw_list:
                    t += [coherence(raw[61 - i][0][0], raw[j][0][0], 8, 12)]
                if (61 - i) <= j:
                    for temp_coh in t:
                        tempwriter.writerow(['0', temp_coh])
                    tempread.close
                s = np.mean(t)
                data += [s]
                print('c' + str(i) + ' ' + str(j))
            writer.writerow(data)
        fileread.close
    else:
        fileread = open('eeg_coherence_p.csv', 'w', newline='')
        writer = csv.writer(fileread)
        writer.writerow(columns)
        for i in range(0, 62):
            data = []
            for j in range(0, 62):
                t = []
                if (61 - i) <= j:
                    tempread = open('eeg_coh_anova/' + str(columns[61 - i]) + '_' + str(columns[j]) + '_p' + '.csv',
                                    'w', newline='')
                    tempwriter = csv.writer(tempread)
                    tempwriter.writerow(['id', 'coherence'])
                for raw in raw_list:
                    t += [coherence(raw[61 - i][0][0], raw[j][0][0], 7.5, 12.5)]
                if (61 - i) <= j:
                    for temp_coh in t:
                        tempwriter.writerow(['1', temp_coh])
                    tempread.close
                s = np.mean(t)
                data += [s]
                print('c' + str(i) + ' ' + str(j))
            writer.writerow(data)
        fileread.close


def coh(control_raw, patient_raw):
    control_raw_temp = []
    patient_raw_temp = []
    count = 0
    for (eid, raw) in control_raw.items():
        raw.load_data()
        raw.drop_channels(['Oz', 'ECG'])
        # raw = eeg_sub_bands.eeg_sub_bands(raw, 'alpha1')
        raw = raw.filter(8, 12)
        raw = raw.resample(48, npad='auto')
        print(count)
        count += 1
        control_raw_temp.append(raw)
    coherence_ofList(control_raw_temp, 'c')
    count = 0
    for (eid, raw) in patient_raw.items():
        raw.load_data()
        raw.drop_channels(['Oz', 'ECG'])
        # raw = eeg_sub_bands.eeg_sub_bands(raw, 'alpha1')
        raw = raw.filter(8, 12)
        raw = raw.resample(48, npad='auto')
        print(count)
        count += 1
        patient_raw_temp.append(raw)
    coherence_ofList(patient_raw_temp, 'p')


def raw_data_info(filePath):
    raw = mne.io.read_raw_brainvision(filePath + '/health_control/eyeclose/jkdz_cc_20180430_close.vhdr',
                                      preload=True)
    # channel_names = raw.info['ch_names']
    print()
    channel_names = []
    for i in raw.info['ch_names']:
        if i != 'Oz':
            if i != 'ECG':
                channel_names.append(i)

    bad_channels = ['Oz', 'ECG']
    return channel_names, bad_channels


def troublesome_data(filePath):
    control_q = []
    patient_q = []
    for dirpath, dirs, files in os.walk(filePath):

        if 'eyeclose' in dirpath and 'health_control' in dirpath:
            # health control group
            for fname in files:
                if '.vhdr' in fname:
                    id_control = fname[:-5]
                    vmrkf, eegf = check_file.get_vhdr_info(dirpath + '/' + fname)
                    if vmrkf == eegf and vmrkf == id_control:
                        print('OK')
                    else:
                        control_q.append(id_control)

        elif 'eyeclose' in dirpath and 'mdd_patient' in dirpath:
            # mdd group
            for fname in files:
                if '.vhdr' in fname:
                    id_patient = fname[:-5]
                    vmrkf, eegf = check_file.get_vhdr_info(dirpath + '/' + fname)
                    if vmrkf == eegf and vmrkf == id_patient:
                        print('OK')
                    else:
                        patient_q.append(id_patient)

    return control_q, patient_q


def read_data(filePath):
    control_q, patient_q = troublesome_data(filePath)
    print(patient_q)
    print('---------===========-----------')
    control_raw = {}
    patient_raw = {}

    for dirpath, dirs, files in os.walk(filePath):

        if 'eyeclose' in dirpath and 'health_control' in dirpath:
            # health control group
            for fname in files:
                if '.vhdr' in fname and fname not in control_q:
                    id_control = fname[:-5]

                    raw = mne.io.read_raw_brainvision(dirpath + '/' + fname, preload=False)
                    if len(raw.info['ch_names']) == 65:
                        raw.set_montage(mne.channels.read_montage("standard_1020"))
                        control_raw[id_control] = raw
                    else:
                        print("Abnormal data with " + str(len(raw.info['ch_names'])) + " channels. id=" + id_control)

        elif 'eyeclose' in dirpath and 'mdd_patient' in dirpath:
            # mdd group
            for fname in files:
                if '.vhdr' in fname and fname[:-5] not in patient_q:
                    id_patient = fname[:-5]

                    raw = mne.io.read_raw_brainvision(dirpath + '/' + fname, preload=False)

                    if len(raw.info['ch_names']) == 65:
                        raw.set_montage(mne.channels.read_montage("standard_1020"))
                        patient_raw[id_patient] = raw
                    else:
                        print("Abnormal data with " + str(len(raw.info['ch_names'])) + " channels. id=" + id_patient)

    return control_raw, patient_raw


def read_file(filePath='/home/rbai/eegData'):
    control_raw, patient_raw = read_data(filePath)
    coh(control_raw, patient_raw)


def start():
    read_file()

    eeg_coherence_plot.coh_plot()

    eeg_coherence_anova.coh_anova_save_csv()
    eeg_coherence_anova.get_coherence_anova()

    eeg_coherence_anova_plot.anova_plot()
    eeg_coherence_plot_difference.brain_plot()
