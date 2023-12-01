import bids
import enlighten

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.signal import coherence
from os.path import join, exists

sns.set(style='white', context='talk')

mather = '/Users/katherine.b/Dropbox/Data/ds001242'
musser = '/Users/katherine.b/Dropbox/Data/musserk23'
diva = '/home/data/nbc/Laird_DIVA/dset'

datasets = [musser, mather]

manager = enlighten.get_manager()

for dataset in datasets:
    dset_name = dataset.split('/')[-1]

    dset = bids.BIDSLayout(dataset, derivatives=True)
    files = dset.derivatives['PhysioComb'].get(extension='tsv', desc='filtered', suffix='physio', invalid_filters=True)
    
    tocks = manager.counter(total=len(files), desc=dset_name, unit='files')

    msc_ecg = pd.DataFrame()
    msc_eda = pd.DataFrame()
    for file in files:
        fs = file.get_metadata()['SamplingFrequency']
        dat = pd.read_table(file.path)
        cardiac = dat.filter(regex='cardiac.*').columns
        nperseg = fs * 4
        subject = file.entities['subject']
        base_path =file.path.replace(file.filename, '')
        try:
            session = file.entities['session']
            no_mr_path = f'{base_path}sub-{subject}_ses-{session}_desc-noMR_physio.tsv'
        except:
            no_mr_path = f'{base_path}sub-{subject}_desc-noMR_physio.tsv'
        base_path = file.path.replace(file.filename, '')
        
        if exists(no_mr_path):
                no_mr = pd.read_table(no_mr_path, header=0)
        for col1 in cardiac:
            if exists(no_mr_path):
                # calculate MSC for no-MR here if dset doesn't contain ds0001242 or whatever
                f, Cxy = coherence(dat[col1], 
                                    no_mr['cardiac'], 
                                    fs=fs, 
                                    nperseg=nperseg)
                temp = pd.Series(data=Cxy, 
                                index=f, 
                                name=f'{col1}_no_mr')
                msc_ecg = pd.concat([msc_ecg, temp], 
                                axis=1)
            else:
                pass
            for col2 in cardiac:
                if col1 == col2:
                    pass
                else:
                    f, Cxy = coherence(dat[col1], 
                                    dat[col2], 
                                    fs=fs, 
                                    nperseg=nperseg)
                    temp = pd.Series(data=Cxy, 
                                    index=f, 
                                    name=f'{col1}_{col2}')
                    msc_ecg = pd.concat([msc_ecg, temp], 
                                    axis=1)
        eda = dat.filter(regex='scr.*').columns
        for col1 in eda:
            if exists(no_mr_path):
                # calculate MSC for no-MR here if dset doesn't contain ds0001242 or whatever
                f, Cxy = coherence(dat[col1], 
                                    no_mr['scr'], 
                                    fs=fs, 
                                    nperseg=nperseg * 10)
                temp = pd.Series(data=Cxy, 
                                index=f, 
                                name=f'{col1}_no_mr')
                msc_eda = pd.concat([msc_eda, temp], 
                                axis=1)
            for col2 in eda:
                if col1 == col2:
                    pass
                else:
                    f, Cxy = coherence(dat[col1], 
                                    dat[col2], 
                                    fs=fs, 
                                    nperseg=nperseg * 10)
                    temp = pd.Series(data=Cxy, 
                                    index=f, 
                                    name=f'{col1}_{col2}')
                    msc_eda = pd.concat([msc_eda, temp], 
                                    axis=1)
        tocks.update()

    new_names = [f.filename.split('_desc')[0] for f in files]
    plots = msc_ecg.columns.unique()
    for plot in plots:
        temp = msc_ecg[plot]
        temp.columns = new_names
        fig,ax = plt.subplots(figsize=(5,5))
        sns.lineplot(data=temp[temp.index < 60], lw=0.9, dashes=False)
        ax.get_legend().remove()
        ax.set_xlabel('Hz')
        ax.set_ylabel('MSC')
        ax.set_ylim([-0.1,1.1])
        fig.savefig(f'{dset_name}_{plot}_msc-ecg.png', dpi=400, bbox_inches='tight')

    plots = msc_eda.columns.unique()
    for plot in plots:
        temp = msc_eda[plot]
        temp.columns = new_names
        fig,ax = plt.subplots(figsize=(5,5))
        sns.lineplot(data=temp[temp.index <= 1], lw=0.9, dashes=False)
        ax.get_legend().remove()
        ax.set_xlabel('Hz')
        ax.set_ylabel('MSC')
        ax.set_ylim([-0.1,1.1])
        ax.set_xlim([-0.1,1.1])
        fig.savefig(f'{dset_name}_{plot}_msc-eda.png', dpi=400, bbox_inches='tight')
