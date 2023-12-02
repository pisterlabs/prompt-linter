import itertools
import numpy as np
import pandas as pd

from pyqeeg.core import coherence_analysis, Coherence, draw_bands, find_iaf, spectral_analysis
from pyqeeg.output_formatting import get_spectra_dataframe, get_coherence_dataframe
from pyqeeg.plotting import plot_coherence, plot_spectra
from pyqeeg.summary import Summary
from pyqeeg.utils import connection, get_channels_with_bad_spectrum

CHANNELS = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
NETWORKS = {"MF": ["AF3", "AF4", "F3", "F4"],
            "LFT": ["F7", "FC5", "T7"],
            "RFT": ["F8", "FC6", "T8"],
            "LP": ["P7", "O1"],
            "RP": ["P8", "O2"]}
NETWORK_CONNECTIONS = {connection(net1, net2): [connection(ch1, ch2)
                                                for ch1 in NETWORKS[net1] for ch2 in NETWORKS[net2]]
                       for net1, net2 in itertools.combinations(NETWORKS.keys(), 2)}
VERSION = "1.0.0"


def run_analysis(subject, session, filename=None, sampling=128, window=2, sliding=0.75, band_method="FBFW",
                 coherence_plots=False, min_samples_for_inclusion=75, whole_head_iaf=None, return_object=False):
    if not filename:
        filename = f"{subject}_{session}.txt"
    try:
        data = pd.read_csv(filename, sep='\t')
    except FileNotFoundError:
        print(f"File {subject}_{session}.txt doesn't exist")
        return

    summary = Summary(subject, VERSION, session, sampling, window, sliding, len(data) / sampling)
    freq = np.array([i * 1 / window for i in range(1, sampling + 1)])
    blink = np.array(data["Blink"]) if "Blink" in data.columns else np.zeros(len(freq))
    summary.fill_meta_blinks(blink)
    x = np.array(data["GyroX"]) if "GyroX" in data.columns else None
    y = np.array(data["GyroY"]) if "GyroY" in data.columns else None

    all_spectra, iafs = {}, {}
    for channel in CHANNELS:
        all_spectra[channel] = spectral_analysis(series=np.array(data[channel]),
                                                 sampling=sampling,
                                                 length=window,
                                                 sliding=sliding,
                                                 x=x,
                                                 y=y,
                                                 blink=blink,
                                                 quality=np.array(data[f"{channel}_Q"]))
        iafs[channel] = find_iaf(all_spectra[channel].power, freq)

    # Handle excludes
    too_few_samples = [ch for ch, spectrum in all_spectra.items() if spectrum.good_samples <= min_samples_for_inclusion]
    # print(too_few_samples)
    no_peak = [ch for ch, iaf in iafs.items() if not iaf.freq]
    bad_spectrum = get_channels_with_bad_spectrum(all_spectra)
    all_excluded = set(too_few_samples + no_peak + bad_spectrum)
    coherence_excluded = set(too_few_samples + bad_spectrum)
    missing_o1_o2 = "O1" in all_excluded and "O2" in all_excluded

    if not whole_head_iaf:
        data_for_iaf = pd.DataFrame.from_dict({channel: all_spectra[channel].power for channel in all_spectra.keys()
                                               if channel not in all_excluded})
        whole_head_spectrum = np.array(data_for_iaf.mean(axis=1))
        print("[" + ", ".join([str(x) for x in whole_head_spectrum]) + "]")
        whole_head_iaf = find_iaf(whole_head_spectrum, freq).freq
    summary.fill_whole_head_iaf(whole_head_iaf)

    if missing_o1_o2:
        band_method = "FBFW"
    summary.fill_band_method(band_method)
    bands = draw_bands(band_method, whole_head_iaf)

    network_spectra = get_network_spectra(all_spectra, all_excluded)
    summary.fill_spectra_metrics(all_spectra, iafs, bands, freq, network_spectra)
    spectra_df = get_spectra_dataframe(subject, freq, all_spectra, network_spectra)
    spectra_df.to_csv(f"{subject}_{session}_spectra.txt", sep="\t")
    plot_spectra(all_spectra, bands)

    all_cohr = {}
    for channel1, channel2 in itertools.combinations(CHANNELS, 2):
        if (channel1 in coherence_excluded) or (channel2 in coherence_excluded):
            all_cohr[connection(channel1, channel2)] = Coherence(good_samples=0,
                                                                 coherence=np.array([np.nan for _ in freq]))
        else:
            all_cohr[connection(channel1, channel2)] = coherence_analysis(series1=np.array(data[channel1]),
                                                                          series2=np.array(data[channel2]),
                                                                          sampling=sampling,
                                                                          length=window,
                                                                          x=x,
                                                                          y=y,
                                                                          blink=blink,
                                                                          quality1=np.array(data[f"{channel1}_Q"]),
                                                                          quality2=np.array(data[f"{channel2}_Q"]))
    networks_coherence = get_network_coherence(all_cohr)
    summary.fill_coherence_metrics(all_cohr, bands, freq, networks_coherence)
    coherence_df = get_coherence_dataframe(subject, freq, all_cohr, networks_coherence)
    coherence_df.to_csv(f"{subject}_{session}_coherence.txt", sep="\t")
    if coherence_plots:
        plot_coherence(all_cohr, bands)

    summary.write_to_file(f"{subject}_{session}_summary.txt")

    if return_object:
        return all_spectra, network_spectra, all_cohr, networks_coherence, summary


def get_network_spectra(all_spectra, excluded):
    network_spectra = {}
    data_for_networks = pd.DataFrame.from_dict({ch: all_spectra[ch].power for ch in all_spectra.keys()
                                                if ch not in excluded})
    for network, channels in NETWORKS.items():
        present_channels = [ch for ch in channels if ch in data_for_networks.columns]
        network_spectra[network] = data_for_networks[present_channels].mean(axis=1)
    return network_spectra


def get_network_coherence(all_cohr):
    networks_coherence = {}
    data_for_networks = pd.DataFrame.from_dict({con: cohr.coherence for con, cohr in all_cohr.items()})
    for network_connection, channel_connections in NETWORK_CONNECTIONS.items():
        networks_coherence[network_connection] = data_for_networks[channel_connections].mean(axis=1)
    return networks_coherence
