import pickle
import numpy as np
import pandas as pd
from scipy.signal import coherence
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster
from scipy.signal import correlate, correlation_lags
from scipy.stats import wilcoxon
from sklearn.cluster import KMeans
import random


def load_data(session):
    file = session + '/ephys_processed/' + session + '_dataset.pkl'
    print(file)
    with open(file, 'rb') as f:
        data = pickle.load(f)
        f.close()
    return data


def get_ch(data):
    return data['info']['ch_names']


def get_lfp(dataset): ## get the lfp data, crop the part before LED_off
    print(dataset['lfp']['amplifier_data'].shape)
    ephys_trigger = dataset['info']['ephys_trigger']
    srate = dataset['info']['frequency_para']['downsample freq/hz']
    crop_from = int(srate * ephys_trigger)

    lfp = dataset['lfp']['amplifier_data'][:, crop_from:]
    ch_list = get_ch(dataset)
    lfp = pd.DataFrame(data=lfp.T, columns=ch_list)

    print(lfp.shape)
    return lfp


def get_power(dataset, brain_area, band='theta', f_ephys=500):
    # print(dataset['lfp']['amplifier_data'].shape)
    ephys_trigger = dataset['info']['ephys_trigger']
    crop_from = int(f_ephys * ephys_trigger)
    if brain_area == 'all':
        ch_list = get_ch(dataset, 'all')
        dat = dataset['bands'][band]['power'][:, crop_from:]
        power = pd.DataFrame(data=dat.T, columns=ch_list)

    if brain_area == 'mpfc':
        ch_list = get_ch(dataset, 'mpfc')
        dat = dataset['bands'][band]['power'][:len(ch_list), crop_from:]
        power = pd.DataFrame(data=dat.T, columns=ch_list)

    if brain_area == 'vhipp':
        ch_list = get_ch(dataset, 'vhipp')
        dat = dataset['bands'][band]['power'][len(get_ch(dataset, 'mpfc')):, crop_from:]
        power = pd.DataFrame(data=dat.T, columns=ch_list)

    # print(power.shape)
    return power


def get_phase(dataset, brain_area, band='theta', f_ephys=500):
    # print(dataset['lfp']['amplifier_data'].shape)
    ephys_trigger = dataset['info']['ephys_trigger']
    crop_from = int(f_ephys * ephys_trigger)

    if brain_area == 'all':
        ch_list = get_ch(dataset, 'all')
        dat = dataset['bands'][band]['phase'][:, crop_from:]
        phase = pd.DataFrame(data=dat.T, columns=ch_list)

    if brain_area == 'mpfc':
        ch_list = get_ch(dataset, 'mpfc')
        dat = dataset['bands'][band]['phase'][:len(ch_list), crop_from:]
        phase = pd.DataFrame(data=dat.T, columns=ch_list)

    if brain_area == 'vhipp':
        ch_list = get_ch(dataset, 'vhipp')
        dat = dataset['bands'][band]['phase'][len(get_ch(dataset, 'mpfc')):, crop_from:]
        phase = pd.DataFrame(data=dat.T, columns=ch_list)

    # print(phase.shape)
    return phase


def sixtyfour_ch_solder_pad_to_zif(zif):  ###
    channel_map = {'T1': 9, 'T2': 10, 'T3': 11, 'T4': 12, 'T5': 13, 'T6': 14, 'T7': 15, 'T8': 16,
                   'T9': 'GND',
                   'T10': 49, 'T11': 50, 'T12': 51, 'T13': 52, 'T14': 53, 'T15': 54, 'T16': 55, 'T17': 56,
                   'T18': 48, 'T19': 47, 'T20': 46, 'T21': 45, 'T22': 44, 'T23': 43, 'T24': 42, 'T25': 41,
                   'T26': 'REF',
                   'T27': 24, 'T28': 23, 'T29': 22, 'T30': 21, 'T31': 20, 'T32': 19, 'T33': 18, 'T34': 17,
                   'B1': 57, 'B2': 58, 'B3': 59, 'B4': 60, 'B5': 61, 'B6': 62, 'B7': 63, 'B8': 64,
                   'B9': 'GND',
                   'B10': 1, 'B11': 2, 'B12': 3, 'B13': 4, 'B14': 5, 'B15': 6, 'B16': 7, 'B17': 8,
                   'B18': 32, 'B19': 31, 'B20': 30, 'B21': 29, 'B22': 28, 'B23': 27, 'B24': 26, 'B25': 25,
                   'B26': 'REF',
                   'B27': 40, 'B28': 39, 'B29': 38, 'B30': 37, 'B31': 36, 'B32': 35, 'B33': 34, 'B34': 33
                   }
    solder_pads = np.zeros(64)
    for ch in range(len(zif)):
        solder_pads[ch] = channel_map[zif[ch]]
    solder_pads = solder_pads.astype('int')
    return solder_pads


zif_connector_to_channel_id = {'T1': 9, 'T2': 10, 'T3': 11, 'T4': 12, 'T5': 13, 'T6': 14, 'T7': 15, 'T8': 16,
                               'T9': 'GND',
                               'T10': 49, 'T11': 50, 'T12': 51, 'T13': 52, 'T14': 53, 'T15': 54, 'T16': 55, 'T17': 56,
                               'T18': 48, 'T19': 47, 'T20': 46, 'T21': 45, 'T22': 44, 'T23': 43, 'T24': 42, 'T25': 41,
                               'T26': 'REF',
                               'T27': 24, 'T28': 23, 'T29': 22, 'T30': 21, 'T31': 20, 'T32': 19, 'T33': 18, 'T34': 17,
                               'B1': 57, 'B2': 58, 'B3': 59, 'B4': 60, 'B5': 61, 'B6': 62, 'B7': 63, 'B8': 64,
                               'B9': 'GND',
                               'B10': 1, 'B11': 2, 'B12': 3, 'B13': 4, 'B14': 5, 'B15': 6, 'B16': 7, 'B17': 8,
                               'B18': 32, 'B19': 31, 'B20': 30, 'B21': 29, 'B22': 28, 'B23': 27, 'B24': 26, 'B25': 25,
                               'B26': 'REF',
                               'B27': 40, 'B28': 39, 'B29': 38, 'B30': 37, 'B31': 36, 'B32': 35, 'B33': 34, 'B34': 33
                               }

intan_array_to_zif_connector = ['B23', 'T1', 'T34', 'B18', 'B19', 'B20', 'T33', 'B17',
                                'T2', 'T32', 'T3', 'B16', 'B13', 'B22', 'T31', 'T4',
                                'B21', 'T30', 'T5', 'B25', 'B14', 'B15', 'T29', 'T6',
                                'B24', 'T28', 'T7', 'B12', 'B10', 'B11', 'T27', 'T8',
                                'B34', 'T25', 'T10', 'B30', 'B2', 'B5', 'T24', 'T11',
                                'B29', 'T23', 'T12', 'B32', 'B1', 'B33', 'T22', 'T13',
                                'B6', 'T21', 'T14', 'B8', 'B27', 'B7', 'T20', 'T15',
                                'B31', 'T19', 'T16', 'B3', 'B28', 'B4', 'T18', 'T17']


### ch_000 ==> B23 ==> pad_27, ch_002 => T1 ...

### pad_1 ==> B10 ==> np.where(intan_array_to_zif_connector['B10'])


def arr_to_pad(el):
    array_to_pad = {}  ### how ephys data array register to the electrode pad. pad1-32 are in the mPFC, with pad1 is the deepest.

    cmap = sixtyfour_ch_solder_pad_to_zif(intan_array_to_zif_connector)
    for i in range(1, 65):
        chan_name = np.where(cmap == i)[0][0]
        if chan_name < 10:
            array_to_pad['A-00' + str(chan_name)] = str(i)
        else:
            array_to_pad['A-0' + str(chan_name)] = str(i)
    return array_to_pad[el]


def pad_to_array():
    pad_to_array = []
    pad_to_array_text = []
    cmap = sixtyfour_ch_solder_pad_to_zif(intan_array_to_zif_connector)
    for i in range(1, 65):
        pad_to_array.append(np.where(cmap == i)[0][0])
        pad_to_array_text.append('pad' + str(i) + '== A-0' + str(np.where(cmap == i)[0][0]))

    return pad_to_array, pad_to_array_text


def get_pad_name(df):
    pad_arr_idx = np.array(pad_to_array()[0])
    pad_list = []
    for col in df.columns:
        col_idx = int(col.split('-')[1])
        pad_list.append(np.where(pad_arr_idx == col_idx)[0][0])

    return pad_list

### rearrange the columns in the order of the electrode depth
def column_by_pad(df):
    col_name = get_pad_name(df)
    df.columns = col_name
    df = df.reindex(sorted(df.columns), axis=1)
    return df



def explore_clusters(dataset, area, cluster_threshold, plot=True, n_clusters=4):
    lfp = get_lfp(dataset, brain_area=area)
    power = get_power(dataset, area)
    chanl_list = get_ch(dataset, brain_area=area)
    coherence_matrix = np.zeros((len(lfp), len(lfp)))
    start = 30
    fps = 500

    # frequs = [5, 25]
    # idxs = [3, 13]

    for x_id, x in tqdm(enumerate(lfp[:, start * fps:(start + 20) * fps])):
        for y_id, y in enumerate(lfp[:, start * fps:(start + 20) * fps]):
            coherence_matrix[x_id, y_id] = coherence(x=x, y=y, fs=500)[1][:20].mean()

    if plot:
        plt.figure(figsize=(10, 10))
        plt.imshow(coherence_matrix, cmap='jet')
        plt.colorbar()
        plt.show()

    corr_linkage = hierarchy.ward(coherence_matrix)
    dendro = hierarchy.dendrogram(corr_linkage, leaf_rotation=90)
    dendro_idx = np.arange(0, len(dendro["ivl"]))
    dendro_idx_pad = [str(arr_to_pad(chanl_list[int(el)])) for el in list(dendro["ivl"])]

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        im = ax.imshow(coherence_matrix[dendro["leaves"], :][:, dendro["leaves"]], cmap="jet")

        ax.set_xticks(dendro_idx)
        ax.set_yticks(dendro_idx)
        ax.set_xticklabels(dendro_idx_pad, rotation="vertical")
        ax.set_yticklabels(dendro_idx_pad)
        fig.tight_layout()
        fig.colorbar(im, orientation="vertical")
        plt.title('Coherence_ch_in_' + area)
        plt.show()

    # clusters based on thresholding
    clusters = fcluster(corr_linkage, cluster_threshold, criterion='distance')
    # sort elements by clusters and put into dictionary
    test1 = [x for _, x in sorted(zip(clusters, dendro_idx))]
    test2 = [x for x, _ in sorted(zip(clusters, dendro_idx))]
    clusts = [str(arr_to_pad(chanl_list[int(el)])) for el in test1]
    clusters_array = {}
    clusters_pad = {}
    for id, cluster in enumerate(test2):
        if cluster not in clusters_array.keys():
            clusters_array[cluster] = []
            clusters_pad[cluster] = []
        else:
            clusters_array[cluster].append(test1[id])
            clusters_pad[cluster].append(clusts[id])

    for cluster in clusters_array.keys():
        means = []
        for channel_1 in clusters_array[cluster]:
            for channel_2 in clusters_array[cluster]:
                if channel_2 == channel_1:
                    continue
                means.append(
                    coherence(x=lfp[channel_1, start * fps:(start + 20) * fps],
                              y=lfp[channel_2, start * fps:(start + 20) * fps], fs=fps)[1][:20].mean())

        print('mean coherence for cluster :' + str(cluster) + '  is:' + str(np.mean(means)) + 'and std: ' + str(
            np.std(means)))
        print(clusters_array[cluster])
        print(clusters_pad[cluster])

    # TODO: check for outliers like channel 41 in vHipp
    # TODO: check smarter solution than just selecting one

    # select channel with highest power
    representative_channels = []
    for cluster in clusters_array.keys():
        cluster_power = power[clusters_array[cluster], :]
        cluster_power = np.nanmean(cluster_power, axis=1)  # cluster_power.mean(axis=1)
        channel_idx = np.where(cluster_power == np.max(cluster_power))[0][0]  ### this line caused error in some dataset
        representative_channels.append(clusters_array[cluster][channel_idx])

    return representative_channels


def explore_clusters2(dataset, area, plot=True, n_clusters=4):
    lfp = get_lfp(dataset, brain_area=area)
    power = get_power(dataset, area)
    channel_list = get_ch(dataset, brain_area=area)
    coherence_matrix = np.zeros((len(lfp), len(lfp)))
    start = 30
    srate = 500

    for x_id, x in tqdm(enumerate(lfp[:, start * srate:(start + 20) * srate])):
        for y_id, y in enumerate(lfp[:, start * srate:(start + 20) * srate]):
            coherence_matrix[x_id, y_id] = coherence(x=x, y=y, fs=500)[1][:20].mean()  # ? take mean or theta band?

    if plot:
        plt.figure(figsize=(10, 10))
        plt.imshow(coherence_matrix, cmap='jet')
        plt.colorbar()
        plt.show()

    kmeans = KMeans(n_clusters=n_clusters, n_jobs=-1)
    kmeans.fit(coherence_matrix)
    kmclust = kmeans.labels_
    # corr_linkage = hierarchy.ward(coherence_matrix)
    # dendro = hierarchy.dendrogram(corr_linkage, leaf_rotation=90)
    # dendro_idx = np.arange(0, len(dendro["ivl"]))
    dendro_idx = np.arange(0, len(kmclust))
    numbering = np.where(kmclust == 0)[0]
    for i in range(1, kmclust.size):
        numbering = np.append(numbering, np.where(kmclust == i)[0])

    # dendro_idx_pad = [str(arr_to_pad(chanl_list[int(el)])) for el in list(dendro["ivl"])]
    dendro_idx_pad = [str(arr_to_pad(channel_list[int(el)])) for el in list(kmclust)]

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        # im = ax.imshow(coherence_matrix[dendro["leaves"], :][:, dendro["leaves"]], cmap="jet")
        im = ax.imshow(coherence_matrix[numbering, :][:, numbering], cmap="jet")

        ax.set_xticks(dendro_idx)
        ax.set_yticks(dendro_idx)
        ax.set_xticklabels(dendro_idx_pad, rotation="vertical")
        ax.set_yticklabels(dendro_idx_pad)
        fig.tight_layout()
        fig.colorbar(im, orientation="vertical")
        plt.title('plot1 ' + area)
        plt.show()

    clusters = kmclust
    # sort elements by clusters and put into dictionary
    test1 = [x for _, x in sorted(zip(clusters, dendro_idx))]
    test2 = [x for x, _ in sorted(zip(clusters, dendro_idx))]
    clusts = [str(arr_to_pad(channel_list[int(el)])) for el in test1]
    clusters_array = {}
    clusters_pad = {}
    for id, cluster in enumerate(test2):
        if cluster not in clusters_array.keys():
            clusters_array[cluster] = []
            clusters_pad[cluster] = []
        else:
            clusters_array[cluster].append(test1[id])
            clusters_pad[cluster].append(clusts[id])

    for cluster in clusters_array.keys():
        means = []
        for channel_1 in clusters_array[cluster]:
            for channel_2 in clusters_array[cluster]:
                if channel_2 == channel_1:
                    continue
                means.append(
                    coherence(x=lfp[channel_1, start * srate:(start + 20) * srate],
                              y=lfp[channel_2, start * srate:(start + 20) * srate], fs=srate)[1][:20].mean())

        print('mean coherence for cluster :' + str(cluster) + '  is:' + str(np.mean(means)) + 'and std: ' + str(
            np.std(means)))
        print(clusters_array[cluster])
        print(clusters_pad[cluster])

    # TODO: check for outliers like channel 41 in vHipp
    # TODO: check smarter solution than just selecting one

    # select channel with highest power
    representative_channels = []
    for cluster in clusters_array.keys():
        cluster_power = power[clusters_array[cluster], :]
        cluster_power = np.nanmean(cluster_power, axis=1)  # cluster_power.mean(axis=1)
        try:
            channel_idx = np.where(cluster_power == np.max(cluster_power))[0][
                0]  ### this line caused error in some dataset
        except ValueError:
            continue
        # representative_channels.append(clusters_array[cluster][channel_idx])
        representative_channels.append(clusters_pad[cluster][channel_idx])

    return representative_channels


def slice_from_arr(arr, events, channels=None, window=1, fps_ephys=500, fps_behavior=50, mean=True):
    window_samples = window * fps_ephys
    f_ratio = fps_ephys / fps_behavior
    ret = []
    if not channels:
        channels = arr.shape[0]
        for channel in range(channels):
            power_per_channel = []
            for idx in events:
                idx_in_ephys = f_ratio * idx
                window_from = np.max([int(idx_in_ephys - window_samples), 0])
                window_to = int(idx_in_ephys + window_samples)
                if window_to > arr.shape[-1] - 1:
                    continue
                # if window_to == 0:
                #     window_to = 1
                bit_power = arr[channel, window_from:window_to]
                if mean:
                    power_per_channel.append(bit_power.mean())
                else:
                    power_per_channel.append(bit_power)
            ret.append(np.vstack(power_per_channel))

        if mean:
            return np.stack(ret)[:, :, 0]
        else:
            return np.stack(ret)

    ret = []
    for channel in channels:
        power_per_channel = []
        for idx in events:
            idx_in_ephys = f_ratio * idx
            window_from = np.max([int(idx_in_ephys - window_samples), 0])
            window_to = int(idx_in_ephys + window_samples)
            if window_to > arr.shape[-1] - 1:
                continue
            # window_to = np.min([int(idx_in_ephys + window_samples), arr.shape[-1] - 1])  # make sure
            # if window_to == 0:
            #     window_to = 1
            bit_power = arr[channel, window_from:window_to]
            if mean:
                power_per_channel.append(bit_power.mean())
            else:
                power_per_channel.append(bit_power)
        ret.append(np.vstack(power_per_channel))

    if mean:
        return np.stack(ret)[:, :, 0]
    else:
        return np.stack(ret)


def epoch_power(power, events, channels=None, window=None):
    idx_in_ephys = [i * 10 for i in events]
    ret = []
    print(events[-1], power.shape[-1])
    for channel in channels:
        power_epoch_per_channel = [power[channel, i] for i in idx_in_ephys]
        ret.append(power_epoch_per_channel)
    return ret


def epoch_data(data_df, channels, events, window=3, f_ephys=500):
    ret = []
    idx_ephys = events * 10
    for i in idx_ephys:
        crop_from = i - window * f_ephys
        crop_to = i + window * f_ephys
        if channels:
            epoch = data_df[channels].iloc[crop_from:crop_to].to_numpy(copy=True).T

        elif channels == None:
            epoch = data_df.iloc[crop_from:crop_to].to_numpy(copy=True).T

        ret.append(epoch)
    return np.array(ret)


def plot_epochs(epochs, average=False):
    t = np.arange(-epochs.shape[-1] / 2, epochs.shape[-1] / 2)
    epoch_mean = epochs.mean(axis=0)
    if average == False:
        for i in range(epoch_mean.shape[0]):
            plt.plot(t, epoch_mean[i, :], alpha=0.4)
    if average == True:
        plt.plot(t, epoch_mean.mean(axis=0))
    plt.show()


def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a
    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den*x/number))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den==1:
            if num==0:
                return r'$0$'
            if num==1:
                return r'$%s$'%latex
            elif num==-1:
                return r'$-%s$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num==1:
                return r'$\frac{%s}{%s}$'%(latex,den)
            elif num==-1:
                return r'$\frac{-%s}{%s}$'%(latex,den)
            else:
                return r'$\frac{%s%s}{%s}$'%(num,latex,den)
    return _multiple_formatter


def get_phase_diffs(data, animal, session, tstart, twin, exclude, nbins, select_idx, band='theta', srate=500, beh_srate=50):
    phase_mpfc = column_by_pad(get_phase(data, 'mpfc', band))
    phase_vhipp = column_by_pad(get_phase(data, 'vhipp', band))
    power_mpfc = column_by_pad(get_power(data, 'mpfc', band))
    power_vhipp = column_by_pad(get_power(data, 'vhipp', band))

    start = int(tstart * srate)
    end = int((tstart + twin) * srate)

    points = []
    for i in range(start, end):
        beh_time_to_start = int(i / srate * beh_srate)
        if select_idx is not None:
            if beh_time_to_start in select_idx:
                points.append(i)
        else:
            points.append(i)
    print(str(len(points)) + ' out of ' + str(end-start) + ' ephys sample points selected (behavior)')

    results = {}
    pairid = 0
    for mpfc_ch in np.array(phase_mpfc.columns):
        for vhipp_ch in np.array(phase_vhipp.columns):
            if (not mpfc_ch in exclude) and (not vhipp_ch in exclude):
                pair_result = {}
                power_vhipp_curr = np.array(power_vhipp[vhipp_ch])
                power_vhipp_mean = np.mean(power_vhipp_curr)
                filtered = []
                for pos in points:
                    if pos < len(power_vhipp_curr):
                        if power_vhipp_curr[pos] > power_vhipp_mean:
                            filtered.append(pos)
                    else:
                        break

                phase_diff = np.unwrap(np.array(phase_mpfc[mpfc_ch])) - np.unwrap(np.array(phase_vhipp[vhipp_ch]))
                phase_diff_filtered = phase_diff[filtered]
                phase_diff_filtered = (phase_diff_filtered + np.pi) % (2.0 * np.pi) - np.pi
                pair_result.update({'mpfc_channel':mpfc_ch, 'vhipp_channel':vhipp_ch, 'phase_diff_filtered': phase_diff_filtered})

                bin_edges = np.linspace(-np.pi, np.pi, num=nbins+1)
                hist, _ = np.histogram(phase_diff_filtered, bins=bin_edges)
                bin_max = np.where(hist == np.max(hist))
                pair_result.update({'peak_position': bin_edges[bin_max][0]})

                difference = np.max(hist) - np.min(hist)
                HM = difference / 2.0
                pos_extremum = hist.argmax()
                nearest_above = (np.abs(hist[pos_extremum:-1] - HM)).argmin()
                nearest_below = (np.abs(hist[0:pos_extremum] - HM)).argmin()
                HM_right = np.mean(bin_edges[nearest_above + pos_extremum])
                HM_left = np.mean(bin_edges[nearest_below])
                pair_result.update({'HM_right': HM_right,'HM_left': HM_left})
                pair_result.update({'FWHM': HM_right - HM_left})

                results.update({pairid: pair_result})
                pairid += 1

    return results


def get_FWHM(phase_diff_result):
    mpfc_channels = []
    vhipp_channels = []

    for pair_id in phase_diff_result:
        pair_result = phase_diff_result[pair_id]
        if not pair_result['mpfc_channel'] in mpfc_channels:
            mpfc_channels.append(pair_result['mpfc_channel'])
        if not pair_result['vhipp_channel'] in vhipp_channels:
            vhipp_channels.append(pair_result['vhipp_channel'])
    mpfc_channels.sort()
    vhipp_channels.sort()

    FWHMs = np.zeros((len(mpfc_channels), len(vhipp_channels)))
    for pair_id in phase_diff_result:
        pair_result = phase_diff_result[pair_id]
        FWHMs[mpfc_channels.index(pair_result['mpfc_channel']), vhipp_channels.index(pair_result['vhipp_channel'])] = \
            pair_result['FWHM']

    FWHMs_return = {
        'mpfc_channels': mpfc_channels,
        'vhipp_channels': vhipp_channels,
        'FWHMs': FWHMs
    }
    return FWHMs_return


def get_lag(sig1, sig2, srate=500):
    corr = correlate(sig1, sig2, mode="full")
    # corr /= np.max(corr)
    lags = correlation_lags(len(sig1), len(sig2), mode="full") / srate * 1000
    lag = lags[np.argmax(corr)]

    return lag, lags, corr


def bootstrap(sig1, sig2, tstart, twin, srate=500, seglen=0.5, offset_min=5, offset_max=10, num=1000):
    start = int(tstart * srate)
    end = int((tstart + twin) * srate)
    corrvalue_max = []

    for i in range(num):
        seg1_cropfrom = random.randint(start + int((offset_max + seglen) * srate),
            end - int((offset_max + seglen) * srate))
        seg1_cropto = int(seg1_cropfrom + seglen * srate)
        offset = random.uniform(offset_min, offset_max)
        backforth = random.choice((-1, 1))
        seg2_cropfrom = seg1_cropfrom + int(backforth * offset * srate)
        seg2_cropto = int(seg2_cropfrom + seglen * srate)

        seg1 = sig1[seg1_cropfrom:seg1_cropto]
        seg2 = sig2[seg2_cropfrom:seg2_cropto]
        lag, lgas, corr = get_lag(seg1, seg2)
        corrvalue_max.append(np.max(corr))
    
    return np.array(corrvalue_max)


def get_seg_lags(data, animal, session, tstart, twin, exclude, seglen, select_idx, band='theta', srate=500, beh_srate=50):
    power_mpfc = column_by_pad(get_power(data, 'mpfc', band))
    power_vhipp = column_by_pad(get_power(data, 'vhipp', band))
    results = {}
    
    segs = []
    maxtime = int((tstart + twin) * beh_srate)

    pos = 0
    segstart = pos

    while pos < maxtime:
        while (not pos in select_idx) and (pos < maxtime):
            pos += 1
        segstart = pos
        count = 0
        if pos in select_idx:
            while count < int(beh_srate * seglen) and pos in select_idx:
                count += 1
                pos += 1
        if count >= int(beh_srate * seglen):
            segs.append(segstart)
        else:
            pos += 1
        pos += random.randint(1,5)
    print('Number of time segments extracted: ', len(segs))
    results.update({'segment_starts':segs})

    pairid = 0
    for mpfc_ch in np.array(power_mpfc.columns):
        print('mPFC-ch-' + str(mpfc_ch))
        for vhipp_ch in np.array(power_vhipp.columns):
            if (not mpfc_ch in exclude) and (not vhipp_ch in exclude):
                power_vhipp_np = np.array(power_vhipp[vhipp_ch])
                vhipp_mean = np.mean(power_vhipp_np)
                power_vhipp_np_nodc = power_vhipp_np - vhipp_mean
                power_mpfc_np = np.array(power_mpfc[mpfc_ch])
                mpfc_mean = np.mean(power_mpfc_np)
                power_mpfc_np_nodc = power_mpfc_np - mpfc_mean

                corr_maxs = bootstrap(power_mpfc_np_nodc, power_vhipp_np_nodc, tstart, twin)
                pass_corr = np.percentile(corr_maxs, 90)

                pair_result = {}
                pair_lags = []
                range_rej = 0
                value_rej = 0
                for seg in segs:
                    segstart = seg / beh_srate
                    segend = segstart + seglen
                    crop_from = int(segstart * srate)
                    crop_to = int(segend * srate)
                    power_mpfc_crop_nodc = power_mpfc_np_nodc[crop_from:crop_to]
                    power_vhipp_crop_nodc = power_vhipp_np_nodc[crop_from:crop_to]
                    power_vhipp_crop = power_vhipp_np[crop_from:crop_to]
                    vhipp_mean_crop = np.mean(power_vhipp_crop)
                    if vhipp_mean_crop > vhipp_mean:
                        lag, lags, corr = get_lag(power_mpfc_crop_nodc, power_vhipp_crop_nodc)
                        if np.max(corr) > pass_corr and -100<=lag<=100:
                            pair_lags.append(lag)
                        if not -100<=lag<=100:
                            range_rej += 1
                        if not np.max(corr) > pass_corr:
                            value_rej += 1
                print('rejected by out-of-100ms-range: ' + str(range_rej))
                print('rejected not-passing-bootstrap: ' + str(value_rej))
                
                pair_result.update({'mpfc_channel':mpfc_ch, 'vhipp_channel':vhipp_ch, 'allseg_lags': pair_lags})
                pair_lags_all = np.array(pair_lags).flatten()
                medianpos = np.median(pair_lags_all)
                meanpos = np.mean(pair_lags_all)
                w, p = wilcoxon(pair_lags_all)
                bin_edges = np.linspace(-int(seglen*srate*2), int(seglen*srate*2), num=int(seglen*srate*2))
                hist, _ = np.histogram(pair_lags_all, bin_edges)
                bin_max = np.where(hist == np.max(hist))
                pair_result.update({'wilcox_p': p, 'mean_lag':meanpos, 'median_lag': medianpos, 'peak_lag': bin_edges[bin_max][0]})

                results.update({pairid: pair_result})
                pairid += 1

    return results


import scipy.stats
import scipy.optimize
import numpy as np


def exg_fun(x, a, b, c, u, s, c2, u2, s2):
    '''
     General Function for exponent + Gaussain
    '''
    return -np.abs(a) * np.exp(-np.abs(b) * x) + c * (1 / ((2 * s * np.pi) ** 0.5)) * np.exp(-((x - u) ** 2) / (2 * s)) \
                                      + c2* (1/  ((2 * s2* np.pi) ** 0.5)) * np.exp(-((x - u2)**2)  / (2*  s2))


def fit_exg(xdata, ydata, bounds=None):
    '''
        General Fitting Function
    '''

    params = [1, 1, 1, 1, 1, 0, 0, 1]

    return scipy.optimize.curve_fit(exg_fun,xdata
                                           ,ydata,
                                            p0=params)


def exg_auc(xdata, ydata):
    '''
    Return Area under the curve (AUC) for Gaussain Function
    Input: xdata: frquency or independent varaible
    Input: ydata: the value of spectrum at the given frequency

    Output: Area Under the Curve
    '''
    popt, pcov = fit_exg(xdata, ydata)
    _, _, auc1, _, _,auc2,_,_ = popt
    return auc1 + auc2


def unit_test():
    '''
        Unit Test
    '''
    xdata = np.linspace(0, 20, 1000)
    y = exg_fun(xdata, 3, .2, 5, 4, .4)
    ydata = y + 0.2 * np.random.normal(size=xdata.size)

    popt, pcov = fit_exg(xdata, ydata, bounds=(0, [3., 1., 0.5]))

    plt.plot(xdata, ydata, 'b-', label='data')
    plt.plot(xdata, exg_fun(xdata, *popt), 'g--',
             label='fit: a=%5.3f, b=%5.3f, c=%5.3f, u=%5.3f, s=%5.3f' % tuple(popt))
    plt.legend()

    print(exg_auc(xdata, ydata))

    plt.show()


def data_to_long(df, mpfc_ch, vhpc_ch, freqs, pad_depth):
    data_list = []
    for col in df.columns:
        if col in mpfc_ch and col in pad_depth.keys():
            data = pd.DataFrame.from_dict({'Freq(Hz)': list(freqs),
                                           'Amp': list(df[col])})
            data['Depth'] = round(pad_depth[col])
            data['Area'] = 'mPFC'
            data_list.append(data)

        if col in vhpc_ch and col in pad_depth.keys():
            data = pd.DataFrame.from_dict({'Freq(Hz)': list(freqs),
                                           'Amp': list(df[col])})
            data['Depth'] = round(pad_depth[col])  ## it fills all the rows with ''
            data['Area'] = 'vHPC'
            data_list.append(data)

    merged_data = pd.concat(data_list)
    return merged_data


def mean_std(data, mpfc_ch):
    mPFC_mean = np.mean(data[:len(mpfc_ch), :], axis=0)
    mPFC_std = np.std(data[:len(mpfc_ch), :], axis=0)
    vHPC_mean = np.mean(data[len(mpfc_ch):, :], axis=0)
    vHPC_std = np.std(data[len(mpfc_ch):, :], axis=0)

    return mPFC_mean, mPFC_std, vHPC_mean, vHPC_std


def get_rms(data):
  data2 = np.power(data,2)
  return np.sqrt(np.sum(data2, axis=0)/data.shape[0])