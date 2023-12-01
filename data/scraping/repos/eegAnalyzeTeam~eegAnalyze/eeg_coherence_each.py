import numpy as np
import mne
import check_file
import os
from scipy import signal
import csv

from coherence_code import pick_eeg_coherence
from coherence_code import eeg_classify_model


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


def coherence(x, y, a, b):
    fs = 48
    f, Cxy = signal.coherence(x, y, fs, nperseg=20)
    global start
    global end
    if start == -1:
        start, end = coh_get_Section(f, a, b)
    return (np.mean(Cxy[start:end]))


def coh_get_channel_names():
    raw = mne.io.read_raw_brainvision('/home/rbai/eegData/health_control/eyeclose/jkdz_cc_20180430_close.vhdr',
                                      preload=True)
    channel_names = []
    for i in raw.info['ch_names']:
        if i != 'Oz':
            if i != 'ECG':
                channel_names.append(i)
    return channel_names[0:-1]


def eeg_coherence(raw, name):
    fileread = open(name, 'w', newline='')
    writer = csv.writer(fileread)
    writer.writerow(coh_get_channel_names())
    for i in range(0, 62):
        data = []
        for j in range(0, 62):
            print('channel: ', i, ' ', j)
            data += [coherence(raw[61 - i][0][0], raw[j][0][0], 8, 12)]
        writer.writerow(data)
    fileread.close


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
    # q contains troublesome eeg files. skip them for now
    control_q, patient_q = troublesome_data(filePath)
    # q = ['njh_after_pjk_20180725_close.vhdr', 'ccs_yb_20180813_close.vhdr', 'njh_before_pjk_20180613_close.vhdr', 'ccs_before_wjy_20180817_close.vhdr', 'ccs_after_csx_20180511_close.vhdr']
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
    print('read success')

    count = 0
    for (eid, raw) in control_raw.items():
        count += 1
        raw.load_data()
        raw.drop_channels(['Oz', 'ECG'])
        # raw = eeg_sub_bands.eeg_sub_bands(raw, 'alpha1')
        raw = raw.filter(8, 12)
        raw = raw.resample(48, npad='auto')
        eeg_coherence(raw, 'coherence/eeg_coherence_c_' + str(count) + '.csv')
        print(count)

    count = 0
    for (eid, raw) in patient_raw.items():
        count += 1
        raw.load_data()
        raw.drop_channels(['Oz', 'ECG'])
        # raw = eeg_sub_bands.eeg_sub_bands(raw, 'alpha1')
        raw = raw.filter(8, 12)
        raw = raw.resample(48, npad='auto')

        eeg_coherence(raw, 'coherence/eeg_coherence_p_' + str(count) + '.csv')
        print(count)


def _start():
    pick_eeg_coherence.get_file()
    eeg_classify_model.init_main()
