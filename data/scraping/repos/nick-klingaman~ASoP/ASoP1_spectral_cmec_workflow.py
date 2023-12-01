# Copyright 2021 Lawrence Livermore National Security, LLC
"""
This script is used by cmec-driver to run the ASoP-Spectral metrics.
It is based on the workflow in ASoP1_spectral_main.py and
can be called with the aruments listed below. Keys that can be set
in the config or settings dictionary are: region, timescale-all, mask,
dates-all, and season-all.

Arguments:
    * model_dir:
        directory containing model data
    * obs_dir:
        directory containing obs data
    * wk_dir:
        output directory
    * config_path:
        JSON config file (optional)
    * settings:
        dictionary of settings (optional)

Author: Ana Ordonez
"""
import argparse
from datetime import datetime, timezone
import glob
import itertools
import json
import os
from platform import python_version

import iris

import make_hist_maps
import plot_hist_maps
import plot_hist1d
from ASoP_Spectral_metric import plot_metric
from set_descriptive_text import set_descriptive_text

# set date once for provenance
current_date = datetime.now(timezone.utc).strftime("%b %d %Y %H:%M:%S")+" UTC"
# setting output directory names
figure_dir_name = "asop_figures"
metrics_dir_name = "asop_metrics"

def main(model_dir, obs_dir, wk_dir, config_path=None, settings=None):
    """
    Read in data and create histogram cubes, save these to netcdf files.
    Then plot histogram maps and some regional 1d histograms

    Arguments:
    * model_dir
        Directory containing model precipitation time series and/or
        pre-calculated histogram cubes
    * obs_dir
        Directory containing observational precipitation time series
        and/or pre-calculated histogram cubes.
    * wk_dir
        Path to output directory
    * config_path (optional)
        Path to configuration JSON (for CMEC driver)
    * settings (optional)
        Dictionary containing choices for region and timescale
    """
    # Load CMEC config
    if config_path is not None:
        print("Loading configuration file")
        with open (config_path,"r") as fname:
            settings=json.load(fname)["ASoP/Spectral"]
        print("Settings from configuration file:\n",json.dumps(settings, indent=4))
    elif settings is None:
            settings={
                "regions": {"default":[-10.0, 10.0, 60.0, 160.0]},
                "figure_type": "png",
                "timescale-all": "",
                "mask": None,
                "dates-all": "",
                "season-all": ""}
            print("Using default settings")

    # Re-order the regions from Coherence to Spectral format
    for r in settings["regions"]:
        settings["regions"][r][:]=[settings["regions"][r][i] for i in [2,0,3,1]]
    # Clean up extension in case there is a leading '.'
    ext = '.'+settings.get('figure_type','png').replace(".","")

    # Set up output files and directories
    json_filename=os.path.join(wk_dir,"output.json")
    initialize_descriptive_json(json_filename,wk_dir,model_dir,obs_dir)
    os.mkdir(os.path.join(wk_dir,figure_dir_name))
    os.mkdir(os.path.join(wk_dir,metrics_dir_name))

    # Get input file lists and separate histogram cubes from timeseries
    hist_input_model,model_filenames=get_filename_lists(model_dir)
    hist_input_obs,obs_filenames=get_filename_lists(obs_dir)

    # Make and save histogram cubes if they don't already exist
    # for the timeseries files
    make_hist_model,new_hist_model=check_histogram_files(model_filenames)
    new_hist_model=[os.path.join(wk_dir,f) for f in new_hist_model]
    make_hist_obs,new_hist_obs=check_histogram_files(obs_filenames)
    new_hist_obs=[os.path.join(wk_dir,f) for f in new_hist_obs]

    for hlist in [make_hist_model,make_hist_obs]:
        if hlist:
            print("Making histograms")
            making_histogram_files(hlist,wk_dir)

    # Combine input and newly made histogram files into one list
    hist_filenames_model=sorted(hist_input_model+new_hist_model)
    hist_filenames_obs=(hist_input_obs+new_hist_obs)
    if len(hist_filenames_obs) > 1:
        raise RuntimeError("More than one benchmark dataset found.")
    elif len(hist_filenames_obs) == 0:
        raise RuntimeError("No control datasets provided")

    # Want obs to go first in list for diffs
    hist_filenames=hist_filenames_obs+hist_filenames_model
    runtitles_long=make_runtitle(hist_filenames,settings)
    runtitles_short=make_runtitle(hist_filenames,settings,model_only=True)

    region_dict=settings.get("regions",{"default":[60.0,-10.0,160.0,10.0]})

    for region in region_dict:
        # Plot histogram maps
        print("Plotting histogram maps")
        myregion=region_dict[region]
        for item in hist_filenames_model:
            title1=runtitles_long[item]
            title2=runtitles_long[hist_filenames_obs[0]]
            plotname_root=figure_dir_name+'/compare_{0}_{1}_{2}'.format(title1,title2,region)
            filenames=[item,hist_filenames_obs[0]]
            plot_histogram_maps(filenames,plotname_root,wk_dir,myregion,ext,settings)

        # 1d histogram plots
        print("Plotting 1d histograms")
        timescale=settings.get("timescale-all",None)
        plottitle='All datasets'
        plotname_root=figure_dir_name+'/compare_as_1dhistograms_{0}'.format(region)
        # Plot 1d histograms of model data with obs overplotted
        runtitles_model=[runtitles_short[f] for f in hist_filenames_model]
        runtitles_obs=[runtitles_short[hist_filenames_obs[0]]]
        plot_1d_histograms(
            hist_filenames_model,runtitles_model, \
            hist_filenames_obs,runtitles_obs, \
            timescale,myregion,plottitle,plotname_root,wk_dir,ext)

        # 1d histogram DIFFERENCE plots
        print("Plotting 1d histogram differences")
        title_long=runtitles_long[hist_filenames_obs[0]]
        title_short=runtitles_short[hist_filenames_obs[0]]
        titles=[[title_short,runtitles_short[f]] for f in hist_filenames_model]
        filenames = [[hist_filenames_obs[0],f] for f in hist_filenames_model]
        plottitle='Differences between datasets'
        plotname_root=figure_dir_name+'/compare_as_1dhist_differences_{0}_{1}_{2}'.format(title_long,"all_models",region)
        # Plot differences between 1d histograms from 1 model datasets
        plot_1d_histogram_diffs(
            filenames,titles,timescale, \
            myregion,plottitle,plotname_root,wk_dir,ext)

    # plot histogram metric
    mask=settings.get("mask",None)
    print("Mask: " + str(mask))
    dates=settings.get("dates-all","")
    season=settings.get("season-all","")
    # Mask file must be present for this metric
    if (mask is not None) and (timescale is not None):
        if os.path.exists(mask):
            print("Making histogram metrics")
            json_filename=wk_dir+"/"+metrics_dir_name+"/histogram_metric.json"
            model_combo=[[f,hist_filenames_obs[0]] for f in hist_filenames_model]
            initialize_metrics_json(json_filename,hist_filenames_obs[0],hist_filenames_model,settings)
            make_histogram_metrics(model_combo,season,timescale,dates,mask, \
                wk_dir,json_filename,settings,ext)
        else:
            raise RuntimeError("Mask file not found.")
    else:
        for keyword,val in zip([mask,timescale],["mask","timescale-all"]):
            if val is None:
                raise RuntimeError("Keyword not found: {0}",keyword)

    # output html page
    write_index_html(wk_dir,region_dict,ext)
    print('Processing completed OK!')
    return

def check_histogram_files(filename_list):
    """
    For the timeseries files in model_filenames, check if an
    equivalent histogram file already exists.
    Arguments:
    * filename_list
        List of precipitation timeseries files
    """
    make_hist=[]
    new_hist=[]
    check_for_hist=[".".join(f.split(".")[:-1])+"_hist.nc" for f in filename_list]

    for data,hist in zip(filename_list,check_for_hist):
        if not os.path.exists(hist):
            make_hist.append(data)
            new_hist.append(os.path.basename(hist))
    return make_hist, new_hist

def making_histogram_files(filename_list,wk_dir):
    """
    Read in data and create histogram cubes, save these to netcdf files.

    Arguments:
    * filename_list
        List of precipitation timeseries files

    * wk_dir
        Path to output directory
    """
    desc = {}
    for fname in filename_list:
        print("Loading cube for",fname)
        fname_tmp = os.path.basename(fname)
        hname = os.path.join(wk_dir,".".join(fname_tmp.split(".")[:-1])+"_hist.nc")
        ppndata1=make_hist_maps.read_data_cube(fname)
        ppn_hist_cube=make_hist_maps.make_hist_ppn(ppndata1)
        iris.save(ppn_hist_cube, hname)

        desc.update({os.path.relpath(hname,start=wk_dir): {
            "long_name": "iris histogram cubes",
            "description": "histograms saved individually for model and obs data"}})
    update_json("data",desc,wk_dir+"/output.json")
    return

def plot_histogram_maps(hist_filenames,plotname_root,wk_dir,region,ext,settings):
    """
    Plot histogram maps
    """
    hist_filename1=hist_filenames[0]
    hist_filename2=hist_filenames[1]

    ppn_hist_cube1=make_hist_maps.read_data_cube(hist_filename1)
    ppn_hist_cube2=make_hist_maps.read_data_cube(hist_filename2)

    avg_rain_bins_a,avg_rain_bins_frac_a=make_hist_maps.calc_rain_contr(ppn_hist_cube1)
    avg_rain_bins_b,avg_rain_bins_frac_b=make_hist_maps.calc_rain_contr(ppn_hist_cube2)

    ppn_names=make_runtitle([hist_filename1,hist_filename2],settings)
    ppn1_name=ppn_names[hist_filename1].replace("_"," ")
    ppn2_name=ppn_names[hist_filename2].replace("_"," ")
    names=make_runtitle([hist_filename1,hist_filename2],settings,model_only=True)
    runtitle="{0} vs {1}".format(names[hist_filename1].replace("_"," "),names[hist_filename2].replace("_"," "))

    # (optional) Define how you want to lump the bins together (below is the default)
    all_ppn_bounds = [(0.005, 10.), (10., 50.), (50., 100.), (100., 3000.)]

    # Plot as actual contributions for specific region, e.g. 60 to 160E,10S to 10N
    desc={}
    plotname='{0}_actual_contributions{1}'.format(plotname_root,ext)
    plotname=os.path.join(wk_dir,plotname)
    plot_hist_maps.plot_rain_contr(avg_rain_bins_a,avg_rain_bins_b,plotname,
                  runtitle,ppn1_name,ppn2_name,all_ppn_bounds,region=region)
    desc.update({os.path.relpath(plotname,start=wk_dir): {
        "description": "Actual contribution of each timescale for region {0}".format(region)}})
    # Plot as fractional contributions
    plotname='{0}_fractional_contributions{1}'.format(plotname_root,ext)
    plotname=os.path.join(wk_dir,plotname)
    plot_hist_maps.plot_rain_contr(avg_rain_bins_frac_a,avg_rain_bins_frac_b,plotname,
                  runtitle,ppn1_name,ppn2_name,all_ppn_bounds,region=region,frac=1)
    desc.update({os.path.relpath(plotname,start=wk_dir): {
        "description": "Fractional contribution of each timescale for region {0}".format(region)}})

    update_json("plots",desc, wk_dir+"/output.json")
    return

def plot_1d_histograms(filenames,runtitles,filenames_obs,runtitles_obs,timescale,
                                         myregion,plottitle,plotname_root,wk_dir,ext):
    """
    Plot 1d histograms for a small region.

    This example uses histogram cubes pre-calculated from two different model datasets
    on the same timescale, and compares with those from two observational datasets.

    NOTE that the region and the timescale will appear automatically in the plot title
    """
    desc={}
    plotname='{0}_actual{1}'.format(plotname_root,ext)
    plotname=os.path.join(wk_dir,plotname)
    plot_hist1d.plot_1dhist(plotname,myregion,filenames,runtitles,plottitle,timescale=timescale,
                         filenames_obs=filenames_obs,runtitles_obs=runtitles_obs,log=1)
    desc.update({os.path.relpath(plotname,start=wk_dir): {
        "description": "Actual histogram"}})

    plotname='{0}_fractional{1}'.format(plotname_root,ext)
    plotname=os.path.join(wk_dir,plotname)
    plot_hist1d.plot_1dhist(plotname,myregion,filenames,runtitles,plottitle,timescale=timescale,
                         filenames_obs=filenames_obs,runtitles_obs=runtitles_obs,frac=1,log=1)
    desc.update({os.path.relpath(plotname,start=wk_dir): {
        "description": "Fractional histogram"}})
    update_json("plots",desc,wk_dir+"/output.json")
    return


def plot_1d_histogram_diffs(filenames,runtitles,timescale,
                                         myregion,plottitle,plotname_root,wk_dir,ext):
    """
    Plot 1d histograms for a small region.

    This example uses histogram cubes pre-calculated from two different model datasets
    on the same timescale, and compares with those from two observational datasets.

    NOTE that the region and the timescale will appear automatically in the plot title
    """
    desc={}
    plotname='{0}_actual{1}'.format(plotname_root,ext)
    plotname=os.path.join(wk_dir,plotname)
    plot_hist1d.plot_1dhist(plotname,myregion,filenames,runtitles,plottitle,timescale,log=1)
    desc.update({os.path.relpath(plotname,start=wk_dir): {
        "description": "Actual 1d histogram for region "+str(myregion)}})

    plotname='{0}_fractional{1}'.format(plotname_root,ext)
    plotname=os.path.join(wk_dir,plotname)
    plot_hist1d.plot_1dhist(plotname,myregion,filenames,runtitles,plottitle,timescale,frac=1,log=1)
    desc.update({os.path.relpath(plotname,start=wk_dir): {
        "description": "Fractional 1d histogram for "+str(myregion)}})
    update_json("plots",desc,wk_dir+"/output.json")
    return

def make_histogram_metrics(hist_combo,season,timescale,dates,mask,wk_dir,json_filename,settings,ext):
    """Set up and run the histogram metrics and difference plot."""
    for ppn1,ppn2 in hist_combo:
        titles=make_runtitle([ppn1,ppn2],settings,model_only=True)
        name1=titles[ppn1]
        name2=titles[ppn2]
        tmp_list = [x for x in [timescale,season,dates] if x != ""]
        plotname=wk_dir+"_".join(["/"+figure_dir_name+"/histogram_metric",name1,name2]+tmp_list)+ext
        index_list=plot_metric(ppn1,ppn2,name1,name2,season,timescale,dates,mask,plotname)
        result_list=[index_list[x].data.item() for x in range(6)]
        # Add metrics to file. Use full name as key.
        json_title=make_runtitle([ppn1,ppn2],settings)[ppn1]
        results={json_title: {
            "histogram overlap": {
                "global": result_list[0],
                "land": result_list[1],
                "sea": result_list[2],
                "tropics": result_list[3],
                "NH mid-lat": result_list[4],
                "SH mid-lat": result_list[5]
                }
            }
        }
        update_json("RESULTS",results,json_filename)

        # Write figure metadata
        desc={os.path.relpath(plotname,start=wk_dir): {
        "description": "histogram metric global plot"}}
        update_json("plots",desc,wk_dir+"/output.json")

    # Write metrics file metadata
    desc={os.path.relpath(json_filename,start=wk_dir): {
        "description": "Histogram overlap metrics"}}
    update_json("metrics",desc,wk_dir+"/output.json")
    return

def get_filename_lists(directory):
    """Return lists of files in the directory, separating histogram cubes
    end with '_hist.nc' from timeseries files."""
    hist_list=[]
    tseries_list=[]
    if (directory is not None) and (directory != 'None'):
        file_list=sorted(glob.glob(directory+"/*"))
        hist_list = [f for f in file_list if f.endswith("_hist.nc")]
        tseries_list = [f for f in file_list if f not in set(hist_list)]
    return hist_list, tseries_list

def get_cube_name(data_cube,default_name="no name"):
    # Return data set name obtained by checking common name variables
    cube_name=default_name
    for key in ["source_id","short_name","name","source","model"]:
        if key in data_cube.attributes:
            cube_name=data_cube.attributes[key]
            break
    if "variant_label" in data_cube.attributes:
        cube_name+=("_"+data_cube.attributes["variant_label"])
    return cube_name

def make_runtitle(data_cube_names,settings,model_only=False,return_timescale=False):
    """
    Return a list of names for each data cube for use in figure titles. Option to
    return timescale dictionary for histogram map headings.
    """
    cube_name={}
    extra_params = ["timescale","dates","season"]
    timescale = dict.fromkeys(data_cube_names,"")
    for fname in data_cube_names:
        fbasename=os.path.basename(fname)
        tmp_fname="_".join(fbasename.split("_")[:-1])+".nc"
        if "name" in settings.get(fbasename,{}):
            cube_name[fname]=settings[fbasename]["name"].replace(" ","_")
        elif "name" in settings.get(tmp_fname,{}):
            cube_name[fname]=settings[tmp_fname]["name"].replace(" ","_")
        else:
            data_cube=iris.load_cube(fname)
            cube_name[fname]=get_cube_name(data_cube).replace(" ","_")

        # Get season, dates, timescale if available in settings
        for item in extra_params:
            tmp="unknown"
            # First see if 'all' setting exists
            if settings.get(item+"-all",False):
                tmp=settings[item+"-all"]
            # Check for setting under histogram or regular filename
            elif item in settings.get(fbasename,{}):
                tmp=settings[fbasename][item]
            elif item in settings.get(tmp_fname,{}):
                tmp=settings[tmp_fname][item]
            if tmp!="unknown" and not model_only:
                cube_name[fname]=cube_name[fname]+"_"+tmp
            if return_timescale and item=="timescale":
                timescale[fname]=tmp
    if return_timescale:
        return cube_name,timescale
    return cube_name

def initialize_descriptive_json(json_filename,wk_dir,model_dir,obs_dir):
    """
    Create metadata JSON file that describes package outputs.
    """
    from platform import python_version
    output = {"provenance":{},"index": "index.html","data":{},"metrics":{},"plots":{},"html":"index.html"}
    log_path = wk_dir + "/asop_spectral.log.txt"
    output["provenance"] = {
            "environment": {'iris':iris.__version__,'python':python_version()},
            "modeldata": model_dir,
            "obsdata": obs_dir,
            "log": log_path,
            "date": current_date}
    with open(json_filename,"w") as output_json:
        json.dump(output,output_json, indent=2)
    return

def initialize_metrics_json(json_filename,control,test,settings):
    """
    Initalize histogram metrics json for writing metrics
    from ASoP_Spectral_metric.py
    """
    schema = {"name": "CMEC", "version": "v1", "package": "ASoP"}
    dims = {
        "json_structure": ["test dataset","metric","region"],
        "dimensions": {
            "test dataset": {},
            "metric": {
                "histogram overlap": "area under the fractional histogram that is covered by overlap between two individual histograms"},
            "region": {
                "global": "global region",
                "land": "masked land area from -30 to 30 degrees latitude",
                "sea": "masked ocean area from -30 to 30 degrees latitude",
                "tropics": "-15 to 15 degrees latitude",
                "NH mid-lat": "30 to 60 degrees north",
                "SH mid-lat": "30 to 60 degrees south"}}}
    titles = make_runtitle(test,settings)
    for item in titles:
        dims["dimensions"]["test dataset"].update({titles[item]: {}})
    con_name = make_runtitle([control],settings)[control]
    prov = {"environment":{'iris':iris.__version__,'python':python_version()},
            "date":current_date}
    data={"SCHEMA": schema, "DIMENSIONS": dims, "CONTROL": con_name, "RESULTS": {}, "PROVENANCE": prov}
    with open(json_filename,"w") as output_json:
        json.dump(data,output_json,indent=2)
    return

def update_json(json_key, data_description, json_filename):
    """
    Add the dictionary 'data_description' under the key 'json_key' in
    the descriptive output json if it exists
    """
    if os.path.exists(json_filename):
        with open(json_filename,"r") as output_json:
            output=json.load(output_json)
        output[json_key].update(data_description)
        with open(json_filename,"w") as output_json:
            json.dump(output,output_json,indent=2)
    return

def write_index_html(wk_dir,region_dict,ext):
    """Create an html page that links users to the metrics json and
    plots created by ASoP-Spectral. Results must be located in the
    output directory "wk_dir".
    Arguments:
        * wk_dir: output directory
        * region_dict: dictionary of region names and coordinates
        * ext: figure file extension
    """
    metric_file=metrics_dir_name+'/histogram_metric.json'
    fig_list=[figure_dir_name+'/'+f for f in os.listdir(wk_dir+'/'+figure_dir_name) if f.endswith(ext)]
    hist_metric_exists=os.path.exists(os.path.join(wk_dir,metric_file))
    # Extensive descriptions are set in another function.
    intr_txt,mtrc_txt,hst_mp_txt,hst_txt,hst_df_txt=set_descriptive_text()

    # list unique keyword to identify plots for each category
    fig_keys=["contributions","1dhistograms","differences"]
    subtitle_list=["Histogram Maps","All Histograms","Histogram Difference"]
    subheading_list=["actual","fractional"]
    text_list=[hst_mp_txt,hst_txt,hst_df_txt]

    # Initialize html text
    html_text=[
        '<html>\n','<body>','<head><title>ASoP-Spectral</title></head>\n',
        '<br><h1>ASoP-Spectral results</h1>\n',intr_txt]
    contents = [
        '<h2>Contents</h2>\n',
        '<dl>\n','<dt><a href="#Figures">Figures</a></dt>\n',
        '<dd><a href="#Histogram-Maps">Histogram Maps</a></dd>\n',
        '<dd><a href="#All-Histograms">All Histograms</a></dd>\n',
        '<dd><a href="#Histogram-Difference">Histogram Difference</a></dd>\n',
        '</dl>\n']
    if hist_metric_exists:
        contents.insert(2,'<dt><a href="#Metrics">Metrics</a></dt>\n')
        contents.insert(4,'<dd><a href="#Histogram-Metric-Maps">Histogram Metric Maps</a></dd>\n')
    html_text.extend(contents)

    # Check for optional histogram metric files
    if hist_metric_exists:
        html_text.extend([
            '<section id="Metrics">\n',
            '<h2>Metrics</h2>\n',
            mtrc_txt,
            '<br><a href="{0}" target="_blank" >{0}</a>\n'.format(metric_file),
            '</section>\n',
            '<section id="Figures">\n',
            '<h2>Figures</h2>\n',
            '<section id="Histogram-Metric-Maps">\n',
            '<h3>Histogram Metric Maps</h3>'])
        sub_list=[f for f in fig_list if ('histogram_metric' in f)]
        for fig in sub_list:
            html_text.append(
                '<p><a href="{0}" target="_blank" alt={0}><img src="{0}" '.format(fig)
                +'width="647" alt="{0}"></a></p>\n'.format(fig))
    else:
        html_text.append('<section id="Figures">\n')
        html_text.append('<h2>Figures</h2>\n')

    # Build the rest of the titles, subtitles, text, and figures.
    for title,kword,desc in zip(subtitle_list,fig_keys,text_list):
        html_text.extend([
            '<section id="'+title.replace(' ','-')+'">\n',
            '<h3>{0}</h3>\n'.format(title),
            '<p>{0}</p>'.format(desc)])
        plot_list=[f for f in fig_list if (kword in f)]
        for region in region_dict:
            html_text.append('<h4>{0}</h4>\n'.format(region.replace('_',' ')))
            for heading in subheading_list:
                html_text.append('<h5>{0} contribution</h5>\n'.format(heading.capitalize()))
                sub_list=[f for f in plot_list if ((heading in f) and (region in f))]
                for fig in sub_list:
                    html_text.append('<p><a href="{0}" target="_blank" alt={0}><img src="{0}" width="647" alt="{0}"></a></p>\n'.format(fig))
        html_text.append('</section>\n')
    html_text.append('</section>\n')

    html_text.append('</body>\n</html>\n')
    filename=wk_dir+"/index.html"
    with open(filename,"w") as html_page:
        html_page.writelines(html_text)
    return


if __name__ == '__main__':

    parser=argparse.ArgumentParser(description='Process two model '
        'precipitation datasets and compare them with each other and with two '
        'observational datasets.')
    parser.add_argument('model_dir', help='model directory')
    parser.add_argument('wk_dir', help='output directory')
    parser.add_argument('--obs_dir', help='observations directory', default=None, required=False)
    parser.add_argument('--config', help='configuration file', default=None, required=False)
    args=parser.parse_args()

    model_dir=args.model_dir
    obs_dir=args.obs_dir
    wk_dir=args.wk_dir
    config=args.config

    if obs_dir=="None": obs_dir=None

    main(model_dir, obs_dir, wk_dir, config_path=config, settings=None)
