# -*- coding: utf-8 -*-
##### ----------------------------- IMPORTS ----------------------------- #####
import os
import sys
import yaml
import click

# add plot path
parent_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
path = os.path.join(parent_path, 'plots')
if path not in sys.path:
    sys.path.append(path)

# create settings file if it does not exists
temp_path_yaml = 'temp_settings.yaml'
load_path_yaml = 'settings.yaml'
if not os.path.isfile(load_path_yaml):
    import shutil
    shutil.copy(temp_path_yaml, load_path_yaml)
##### ------------------------------------------------------------------- #####

def load_yaml(settings_path):
    with open(settings_path, 'r') as file:
        return yaml.load(file, Loader=yaml.FullLoader)

def save_yaml(settings, settings_path):
    with open(settings_path, 'w') as file:
        yaml.dump(settings, file)

def downsample(search_path, sake_index, new_fs, file_name='downsample.pickle'):
    # downsample
    import pandas as pd
    from preprocess import batch_downsample
    downsample_path = os.path.join(search_path, file_name)
    if os.path.isfile(downsample_path):
        downsampled_df = pd.read_pickle(downsample_path)
    else:
        downsampled_df = batch_downsample(search_path, sake_index, new_fs=new_fs)
        downsampled_df.to_pickle(downsample_path)
    return downsampled_df

@click.group()
@click.pass_context
def main(ctx):
    """
    
    \b---------------------------------------------------                       
    \b---------------- SAKE CONNECTIVITY ----------------                       
    \b---------------------------------------------------                     
    
    """
    
    # load yaml 
    settings = load_yaml(load_path_yaml)
    ctx.obj = settings.copy()
        
### ------------------------------ SET PATH ------------------------------ ### 
@main.command()
@click.argument('path', type=str)
@click.pass_context
def setpath(ctx, path):
    """Set path to index file parent directory"""
    
    # check if index file exists
    index_path = os.path.join(path, ctx.obj['sake_index'])
    if not os.path.isfile(index_path):
        click.secho(f"\n --> File '{ctx.obj['sake_index']}' "  +\
                        f"was not found in '{path}'.\n",
                        fg='yellow', bold=True)
        path=''
            
    # save path
    ctx.obj.update({'search_path': path})
    save_yaml(ctx.obj, load_path_yaml)
    click.secho(f"\n -> Path was set to:'{path}'.\n", fg='green', bold=True)
  
def check_path(ctx):
    if not ctx.obj['search_path']:
        print_str = "\n --> Path was not found, please use -setpath- command."
        click.secho(print_str, fg='yellow', bold=True)
        raise(Exception(print_str))
      
@main.command()
@click.option('--ws', type=str, help='Enter window size (s), e.g. 30')
@click.option('--function', type=str, help='Enter method type: E.g. tort')
@click.pass_context
def coupling(ctx, ws, function='tort'):
    """
    Calculate phase amplitude coupling
    """
    
    # check path
    check_path(ctx)

    if not ws:
        click.secho("\n -> Please enter window size' e.g. --ws 30.\n", fg='yellow', bold=True)
        return
    
    # downsample
    downsampled_df = downsample(ctx.obj['search_path'], ctx.obj['sake_index'],
                                ctx.obj['new_fs'], file_name='downsample.pickle')
    click.secho(f"\n -> Data successfully downsampled to {ctx.obj['new_fs']} Hz'.\n", fg='green', bold=True)
    
    # get coupling index
    from phase_amp import phaseamp_batch
    data = phaseamp_batch(downsampled_df, ctx.obj['iter_freqs'], ctx.obj['new_fs'], int(ws))

    # store data
    data.to_pickle(os.path.join(ctx.obj['search_path'], 'phase_amp_'+ function +'.pickle'))
    click.secho(f"\n -> Coupling completed and data were stored to {ctx.obj['search_path']}'.\n",
                fg='green', bold=True)

    
@main.command()
@click.option('--ws', type=str, help='Enter window size (s), e.g. 5')
@click.option('--function', type=str, help='Analysis type (s), e.g. coh plv')
@click.pass_context
def coherence(ctx, ws, function='coh'):
    """
    Calculate coherence
    """
    # check path
    check_path(ctx)

    if not ws:
        click.secho("\n -> Please enter window size' e.g. --ws 5.\n", fg='yellow', bold=True)
        return
    
    methods = ['coh', 'plv', 'pli']
    method = function.split(' ')
    if not set(method) <= set(methods):
        click.secho(f"\n -> Got '{method}' instead of  '{methods}", fg='yellow', bold=True)
        return
    
    # downsample
    downsampled_df = downsample(ctx.obj['search_path'], ctx.obj['sake_index'],
                                ctx.obj['new_fs'], file_name='downsample.pickle')
    click.secho(f"\n -> Data successfully downsampled to {ctx.obj['new_fs']} Hz'.\n", fg='green', bold=True)
    
    # calculate coherence
    from coherence import coherence_batch
    data = coherence_batch(downsampled_df, ctx.obj['iter_freqs'], ctx.obj['new_fs'], int(ws), method=method)
    data.to_pickle(os.path.join(ctx.obj['search_path'], 'coherence_' + function + '.pickle'))
    click.secho(f"\n -> Coherence completed and data were stored to {ctx.obj['search_path']}'.\n",
                fg='green', bold=True)

@main.command()
@click.option('--method', type=str, help='Analysis type (s), e.g. coherence')
@click.option('--plottype', type=str, help='Analysis type (s), e.g. box')
@click.option('--norm', type=str, help='Enter col-value pair (s), e.g. treatment-baseline1')
@click.option('--function', type=str, help='Function type (s), e.g. coh')
@click.pass_context
def plot(ctx, method, plottype, norm, function):
    """
    Interactive summary plot.

    """
    # check path
    check_path(ctx)
    
    # import modules
    import pandas as pd
    from facet_plot_gui import GridGraph
    
    if method == 'pac':
        file = 'phase_amp_'+ function + '.pickle'
    elif method == 'coherence':
        file = 'coherence_' + function + '.pickle'
     
    # get data
    data = pd.read_pickle(os.path.join(ctx.obj['search_path'], file))
    
    # attempt to initiate normalization
    norm = norm.split('^')
    if len(norm) == 2:
        from normalize import normalize
        y = data.columns[-1]
        base_condition = {'col': norm[0], 'val': norm[1]}
        
    # convert data to appropriate plotting format
    if plottype == 'time':
        plotdf = data
        if len(norm) == 2:
            categories = list(set(plotdf.columns[:-1]) - set([base_condition['col']]))
            plotdf = normalize(plotdf, y, base_condition, categories)
        plotdf = plotdf.drop(columns='method', axis=1).set_index('animal')
        graph = GridGraph(ctx.obj['search_path'], method + function +'.csv', plotdf, x=plottype)
        graph.draw_psd()
    else:
        group_cols = list(data.columns[data.columns.get_loc('time') +1 :-1]) + ['animal']
        plotdf = data.groupby(group_cols).mean().reset_index().drop(columns='time', axis=1)
        if len(norm) == 2:
            categories = list(set(plotdf.columns[:-1]) - set([base_condition['col']]))
            plotdf = normalize(plotdf, y, base_condition, categories)
        graph = GridGraph(ctx.obj['search_path'], method + function +'.csv', plotdf.set_index('animal'), x='band')
        graph.draw_graph(plottype)

    
# Execute if module runs as main program
if __name__ == '__main__':
    
    # start
    main(obj={})

