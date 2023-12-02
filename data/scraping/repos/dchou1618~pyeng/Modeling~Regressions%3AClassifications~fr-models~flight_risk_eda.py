import codecs
import json
from typing import Dict, Any
import os

import matplotlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
from optbinning import OptimalBinning
from plotly.subplots import make_subplots

from preprocess_flight_risk_data import extract_root_fpath

cmap = matplotlib.colormaps.get_cmap('Greens')

# Add LAZ colors
import Lazard_looknfeel_v2 as laz

laz_colours = []
for colour in ['cephalopod ink', 'moonstone', 'moonstone_40', 'gold', 'gold_40']:
    laz_colours += [laz.primary_palette[colour]]
for colour in ['hunter', 'moss', 'olive', 'ocean', 'bark', 'tangerine', 'daffodil', 'lilac', 'chestnut']:
    laz_colours += [laz.extended_palette[colour]]


####################################
# Helper functions for computation #
####################################

def surround_text(txt):
    return f"<span style=\"font-size: 14px;\">{txt}</span>"


def divide_review(review_score, splits):
    if pd.isna(review_score):
        return review_score
    for i in range(len(splits)):
        if review_score < splits[i]:
            return str(i + 1)
    assert review_score > splits[-1], \
        "Review score not greater than lower bound of last bin."
    return str(len(splits) + 1)


def bin_cols(df, col):
    _df = df.copy()
    _df[col] = _df[col].apply(lambda x: 2 * x + 3)
    optb = OptimalBinning(name=col, dtype="numerical", solver="cp",
                          min_n_bins=5,
                          max_n_bins=5)
    optb.fit(_df[col].values, _df["Attrition"])

    return _df[col].apply(lambda x: divide_review(x, optb.splits))


def get_bottom_top_heavy(row, ratio):
    individual = row["Latest " + ratio]
    firmwide = row["Firmwide " + ratio]
    assert individual != firmwide
    return "Bottom Heavy" if individual > firmwide else "Top Heavy"


def add_organization_structures(df):
    ratio_lst = ['Analysts/MDs',
                 'Associates/MDs',
                 'Analysts/MDs+Dirs',
                 'Associates/MDs+Dirs',
                 'Analysts+Associates/MDs+Dirs',
                 "Analysts/Associates",
                 "Analysts/Associates+VPs"]

    for ratio in ratio_lst:
        df[f"{ratio} Organization Structure"] = df.apply(lambda row: get_bottom_top_heavy(row, ratio), axis=1)
    return df


def f(x, feature_of_interest):
    return pd.Series(
        {"Proportion Missing": len(x[pd.isna(x[feature_of_interest])]) / len(x), "Number of Employees": len(x)})


def viz_na(df, var, feature_of_interest):
    missingness = df.groupby(var, as_index=False)[[feature_of_interest]] \
        .apply(lambda x: f(x, feature_of_interest))
    return missingness


def get_avg_from_expected(prop_cuts):
    optimal = 1 / len(prop_cuts)
    dists = [abs(optimal - cut) for cut in prop_cuts]
    return sum(dists) / len(dists)


def break_into_percentiles(numerical_vals):
    '''
    break_into_percentiles -
    '''
    breaks = []
    for i in range(5):
        _break = np.quantile(numerical_vals.dropna(), i / 4)
        if i == 0:
            _break = numerical_vals.dropna().min()
        elif i == 4:
            _break = numerical_vals.dropna().max()
        breaks.append(_break)
    return breaks


def get_category(x, breaks, break_vals=None):
    if x == "REMOTE":
        return f"{round(breaks[-2], 2)} <-> {round(breaks[-1], 2)}"
    if pd.isna(x):
        return x
    assert len(breaks) == 5
    x = float(x)
    for i in range(len(breaks) - 1):
        if x <= breaks[i + 1] and x >= breaks[i]:
            if break_vals is None:
                return f"{round(breaks[i], 2)} <-> {round(breaks[i + 1], 2)}"
            else:
                return break_vals[i]
    assert False, "Not in a bin"


def add_review_comparison(df):
    for var_review_comparison in ['self_exceeds_reviewer_median T-1',
                                  'self_exceeds_reviewer_mean T-1',
                                  "self_exceeds_360_manager_median T-1",
                                  "self_exceeds_360_manager_mean T-1",
                                  "self_exceeds_manager_median T-1",
                                  "self_exceeds_manager_mean T-1"]:
        df[var_review_comparison] = df[var_review_comparison].apply(
            lambda x: "1" if x > 0 else x if pd.isna(x) else "0")
    for var in ["self_exceeds_reviewer_median T-1",
                "self_exceeds_reviewer_mean T-1"]:
        df[var] = df[var].replace({"0": "360 Reviews Exceed Self Review",
                                   "1": "Self Review Exceeds 360 Reviews"})
    return df


def sort_integer_of_str(lst):
    '''
    sort_integer_of_str assumes all entries in the
    input list "lst" are digit strings.
    '''
    assert len(lst) > 0, "List has no elements"
    if type(lst[0]) == str and all(val.isdigit() for val in lst):
        return [str(val) for val in sorted([int(val) for val in lst])]
    else:
        return lst


def bucket_values(x, breaks):
    if pd.isna(x):
        return x
    if x < breaks[0]:
        return "0"

    for i in range(len(breaks) - 1):
        if breaks[i + 1] > x:
            if x >= breaks[i]:
                return str(i + 1)
    if x >= breaks[len(breaks) - 1]:
        return str(len(breaks))
    assert False, "Should not end up here."


def get_sizes_in_order_of_labels(labels, sizes, col):
    lst = []
    for label in labels:
        label_size_df = sizes[sizes[col] == label]
        if len(label_size_df) == 0:
            lst.append(0)
        else:
            lst.append(label_size_df["size"].iloc[0])
    return np.array(lst)


def add_hc_ratio_structures(df):
    for var in ['Latest Analysts/MDs', 'Latest Associates/MDs',
                'Latest Analysts/MDs+Dirs', 'Latest Associates/MDs+Dirs',
                'Latest Analysts+Associates/MDs+Dirs',
                "Latest Analysts/Associates",
                "Latest Analysts/Associates+VPs"]:
        df[var] = df[var].apply(lambda x: "Bottom Heavy" if x > 1 else "Top Heavy")
    return df


def divide_comp_tier_changes(comp_tier_change, splits):
    if pd.isna(comp_tier_change):
        return comp_tier_change
    for i in range(len(splits)):
        if comp_tier_change < splits[i]:
            if i == 0:
                return f"<{splits[i]}"
            else:
                return f"[{splits[i - 1]}, {splits[i]})"
    assert comp_tier_change > splits[-1], \
        "Review score not greater than lower bound of last bin."
    return f">{splits[len(splits) - 1]}"


def bin_comp_tier_changes(df, col):
    _df = df.copy()
    optb = OptimalBinning(name=col, dtype="numerical", solver="cp",
                          min_n_bins=2,
                          max_n_bins=10)
    optb.fit(_df[col].values, _df["Attrition"])
    return _df[col].apply(lambda x: divide_comp_tier_changes(x, optb.splits))


def add_compensation_features(df):
    try:
        df["Comp Tier Change T-1"] = (df["Comp Tier T-1"] - df["Comp Tier T-2"])
        tier_mapping = {0: '1-', 1: "1", 2: "1+", 3: '2-', 4: "2", 5: "2+",
                        6: '3-', 7: "3", 8: '3+', 9: '4-', 10: "4", 11: '4+',
                        12: '5-', 13: "5", 14: '5+'}

        df["Comp Tier T-1"] = df["Comp Tier T-1"].map(tier_mapping)
        df["Comp Tier T-1"] = df["Comp Tier T-1"].apply(lambda x: str(x) if not pd.isna(x) else x)
        df["Comp Tier T-1"] = df["Comp Tier T-1"].replace({val: val[:-1] for val in ["2+", "3-", "3+", "4-", "4+",
                                                                                     "5-", "5+"]})

        if type(df["Comp Tier Change T-1"].dropna().iloc[0]) != str:
            df["Comp Tier Change T-1"] = bin_comp_tier_changes(df, "Comp Tier Change T-1")
    except Exception as e:
        print("Error in add_compensation_features:", e)

    df["YOY TC % Chg (bin) T-1"] = df["YOY TC % Chg (bin) T-1"].replace({"-75.0% - -50.0%": "<0.0%",
                                                                         "-50.0% - -25.0%": "<0.0%",
                                                                         "-25.0% - 0.0%": "<0.0%",
                                                                         "-100.0% - -75.0%": "<0.0%"})
    return df


def add_attrition(original_df, df):
    df["Left the Company?"] = df["Attrition"].apply(lambda x: "Attritioned" if x == 1 else "Not Attritioned")
    original_df["Left the Company?"] = original_df["Attrition"].apply(lambda x: "Attritioned"
    if x == 1 else "Not Attritioned")
    return original_df, df


def latest_perc_women_features(df):
    colnames_df = set(df.columns)
    historical_female_group_cols = [col for col in df.columns if "Women" in col]
    latest_female_group_cols = ["Latest " + " ".join(phrase.split()[1:]) \
                                for phrase in historical_female_group_cols]
    for i, colname in enumerate(latest_female_group_cols):
        if colname not in colnames_df:
            df[colname] = df[historical_female_group_cols[i]].apply(lambda lst: [float(val) \
                                                                                 for val in lst[1:-1].split(", ")][-1]
            if type(lst) == str else lst)
    return df, latest_female_group_cols


############################
# Heatmap Helper Functions #
############################
def update_heatmap(heatmap, attrition_covariates):
    if len(heatmap) == 0:
        heatmap = attrition_covariates
    else:
        heatmap = np.hstack((heatmap, attrition_covariates))
    return heatmap


def update_heatmap_subplot(subplot_fig, new_name, widths, heights, x_labels, y_labels, row, col,
                           xaxis_title="Year"):
    subplot_fig.update_xaxes(range=[0, 100], title=xaxis_title, row=row, col=col)
    subplot_fig.update_yaxes(range=[0, 100], title=new_name, row=row, col=col)
    subplot_fig.update_xaxes(
        tickvals=np.cumsum(widths) - widths / 2,
        ticktext=["%s<br>%d" % (l, w) + "%" for l, w in zip(x_labels, widths)],
        row=row, col=col
    )

    subplot_fig.update_yaxes(
        tickvals=np.cumsum(heights) - heights / 2,
        ticktext=["%s: %d" % (l, w) + "%" for l, w in zip(y_labels, heights)],
        row=row, col=col
    )
    return subplot_fig


def construct_2d_heatmap(df, xcol, ycol, xlabels=None, ylabels=None):
    x_labels = sort_integer_of_str(df[xcol].dropna().unique()) if xlabels is None \
        else xlabels
    y_labels = sort_integer_of_str(df[ycol].dropna().unique()) if ylabels is None \
        else ylabels
    x_sizes = df.groupby([xcol], as_index=False).size()
    x_sizes = get_sizes_in_order_of_labels(x_labels, x_sizes, xcol)

    y_sizes = df.groupby([ycol], as_index=False).size()
    y_sizes = get_sizes_in_order_of_labels(y_labels, y_sizes, ycol)

    widths = 100 * (x_sizes / len(df.dropna(subset=[xcol])))
    heights = 100 * (y_sizes / len(df.dropna(subset=[ycol])))
    widths = np.round(widths, 2)
    heights = np.round(heights, 2)

    data = {label: [] for label in y_labels}
    counts_data = {label: [] for label in y_labels}

    data_min, data_max = float("inf"), float("-inf")

    for label in y_labels:
        existing_review_positivity = df[df[ycol] == label]
        d = {"Stayers": np.array([]),
             "Leavers": np.array([])}
        for x_label in x_labels:
            counts = existing_review_positivity[ \
                existing_review_positivity[xcol] == x_label] \
                ["Attrition"].value_counts()
            counts = dict(counts)
            for attrition_category in [0, 1]:
                if attrition_category not in counts:
                    counts[attrition_category] = 0
            d["Stayers"] = np.append(d["Stayers"], counts[0])
            d["Leavers"] = np.append(d["Leavers"], counts[1])
        totals = d["Stayers"] + d["Leavers"]
        counts_data[label] = totals
        data[label] = np.nan_to_num(np.round(100 * (np.divide(d["Leavers"], \
                                                              totals)), 2))
        curr_min = np.min(data[label])
        curr_max = np.max(data[label])
        if curr_min < data_min:
            data_min = curr_min
        if curr_max > data_max:
            data_max = curr_max
    norm = matplotlib.colors.Normalize(vmin=data_min, vmax=data_max)
    fig = go.Figure()
    for i, key in enumerate(data):
        colors = ["rgba(" + ",".join([str(val) for val in cmap(norm(datum))]) + ")" for datum in data[key]]
        fig.add_trace(go.Bar(
            name=key,
            y=[heights[i]] * len(data[key]),
            x=np.cumsum(widths) - widths,
            width=widths,
            offset=0,
            customdata=np.transpose([np.nan_to_num(data[key]), x_labels,
                                     np.nan_to_num(counts_data[key].astype(int))]),
            textposition="inside",
            textfont_color="black",
            hovertemplate="<br>".join([
                f"Attrition Rate: " + "%{customdata[0]}" + "%",
                f"# of Employees <br> ({key}," +
                " %{customdata[1]}): %{customdata[2]}"
            ]),
            marker=dict(color=colors)
        ))

    fig.update_xaxes(
        tickvals=np.cumsum(widths) - widths / 2,
        ticktext=["%s<br>%d" % (l, w) + "%" for l, w in zip(x_labels, widths)]
    )

    fig.update_yaxes(
        tickvals=np.cumsum(heights) - heights / 2,
        ticktext=["%s: %d" % (l, w) + "%" for l, w in zip(y_labels, heights)]
    )

    fig.update_xaxes(range=[0, 100])
    fig.update_yaxes(range=[0, 100])

    fig.update_layout(
        title_text=f"Attrition Rate of {ycol} vs. {xcol}",
        xaxis_title=xcol,
        yaxis_title=ycol,
        barmode="stack",
        showlegend=False,
        uniformtext=dict(mode="hide", minsize=12),
    )
    return fig, widths, heights, x_labels, y_labels, xcol, ycol


def retrieve_relevant_data(original_df, modified_df, var):
    skip_var = False
    if var not in modified_df.columns:
        data_to_plot = original_df
        if var not in data_to_plot.columns:
            skip_var = True
    else:
        data_to_plot = modified_df
    return data_to_plot, skip_var


def map_data(data_to_plot, var, unique_old_vals, mapping):
    if set(data_to_plot[var].dropna().unique()) == unique_old_vals:
        data_to_plot[var] = data_to_plot[var].map(mapping)
    return data_to_plot


def fixed_preprocessing(data_to_plot, var, category_orderings):
    if "Review" in var and ("MEAN" in var or "MEDIAN" in var):
        if type(data_to_plot[var].dropna().iloc[0]) != str:
            data_to_plot[var] = bin_cols(data_to_plot, var).apply(lambda x: str(x) if not pd.isna(x) else x)
    if "self_exceeds" in var:
        data_to_plot = map_data(data_to_plot, var, {"0", "1"}, {"1": "Self > Reviews",
                                                                "0": "Reviews > Self"})
        data_to_plot = map_data(data_to_plot, var, {"Self Review Exceeds 360 Reviews",
                                                    "360 Reviews Exceed Self Review"},
                                {"Self Review Exceeds 360 Reviews": "Self > 360",
                                 "360 Reviews Exceed Self Review": "360 > Self"})

    if type(data_to_plot[var].dropna().iloc[0]) != str:

        breaks = break_into_percentiles(data_to_plot[var])
        category_order = [str(round(breaks[i], 2)) + " <-> " + \
                          str(round(breaks[i + 1], 2)) for i in range(len(breaks) - 1)]

        data_to_plot[var] = data_to_plot[var].apply(lambda x: get_category(x, breaks, \
                                                                           break_vals=category_order))
        if any(term in var for term in ["Turnover", "%"]):
            data_to_plot[var] = data_to_plot[var].apply(lambda x: get_percentage_bin(x) if not pd.isna(x) \
                else x)
            category_order = [str(np.round(100 * breaks[i])) + "% <-> " + \
                              str(np.round(100 * breaks[i + 1])) + "%" for i in range(len(breaks) - 1)]
            new_category_order = []
            seen = set()
            for val in category_order:
                if val not in seen:
                    seen.add(val)
                    new_category_order.append(val)
            category_order = new_category_order
    else:
        data_uniq_vals = data_to_plot[var].dropna().unique()
        if var in category_orderings:
            category_order = category_orderings[var]
        elif "Much More Top Heavy" in data_uniq_vals:
            category_order = ["Much More Top Heavy",
                              "Slightly More Top Heavy",
                              "Slightly More Bottom Heavy",
                              "Much More Bottom Heavy"]
        elif "Q2" in data_uniq_vals:
            category_order = ["Q1 (Lowest 360 Reviews)", "Q2", "Q3", "Q4 (Highest 360 Reviews)"]
        elif set(data_uniq_vals) == {"0", "1", "2", "3"} or set(data_uniq_vals) == {"0", "1", "2", "3", "4"}:
            category_order = sorted(data_uniq_vals)
            old_min, old_max = category_order[0], category_order[-1]
            category_order[0] += "(Lowest)"
            category_order[-1] += "(Highest)"
            data_to_plot[var] = data_to_plot[var].replace(old_min, category_order[0]) \
                .replace(old_max, category_order[-1])
        else:
            category_order = sorted(data_to_plot[var].dropna().unique(), key=lambda s: float(s.split(" <-> ")[0]) \
                if len(s.split(" <-> ")) > 1 else s)
    return data_to_plot, category_order


def get_percentage_bin(b):
    start, end = b.split("<->")
    start_perc, end_perc = f"{np.round(100 * float(start))}%", f"{np.round(100 * float(end))}%"
    return f"{start_perc} <-> {end_perc}"


def reorder_percentile_bins(bins, var):
    if all("<->" in b for b in bins):
        if all("%" in b for b in bins):
            return bins
        percentile_bin_mapping = {float(b.split("<->")[-1]): b for b in bins}
        reordered_mapping = sorted(percentile_bin_mapping.items(), key=lambda x: x[0])
        reordered_bins = [val for _, val in reordered_mapping]
        if any(term in var for term in ["Turnover", "%"]):
            reordered_bins = [get_percentage_bin(b) for b in reordered_bins]
        return reordered_bins
    return bins


def display_attrition_bars_and_annual_breakdown(data_to_plot, var, row_width, row_height, category_order,
                                                color_seq, subplot_fig, new_name, category_orderings):
    '''
    display_attrition_bars_and_annual_breakdown - displays the first two visuals on a row.
    '''

    category_order = reorder_percentile_bins(category_order, var)
    if var in category_orderings:
        category_order = category_orderings[var]
    data_to_plot = data_to_plot[data_to_plot[var].isin(set(category_order))]
    grouped_data = data_to_plot.groupby(["Left the Company?", var],
                                        as_index=False).count()[["Left the Company?",
                                                                 var, "Year"]] \
        .rename(columns={"Year": "count"})
    fig_heatmap, widths, heights, x_labels, y_labels, xcol, ycol = \
        construct_2d_heatmap(data_to_plot, "Year", var, ylabels=category_order)
    fig_px = px.histogram(grouped_data, x="Left the Company?",
                          y="count", color=var,
                          barnorm='percent',
                          color_discrete_sequence=color_seq,
                          category_orders={var: category_order})
    for trace in range(len(fig_px["data"])):
        subplot_fig.add_trace(fig_px["data"][trace], row=1, col=1)

    subplot_fig.for_each_trace(
        lambda trace: trace.update(text=trace.name)
    )

    subplot_fig.update_traces(texttemplate="%{text}")

    for trace in range(len(fig_heatmap["data"])):
        subplot_fig.add_trace(fig_heatmap["data"][trace], row=1, col=2)

    subplot_fig = update_heatmap_subplot(subplot_fig, new_name, widths, heights, x_labels, y_labels, 1, 2)

    subplot_fig.update_xaxes(title="Attrition", categoryorder="array",
                             categoryarray=["Attritioned", "Not Attritioned"], row=1, col=1)
    subplot_fig.update_yaxes(title=f"% Breakdown of {new_name}",
                             row=1, col=1)
    subplot_fig.update_xaxes(row=1, col=1)

    subplot_fig.update_layout(barmode="stack",
                              barnorm="percent",
                              legend_traceorder='reversed',
                              width=row_width,
                              height=row_height,
                              showlegend=False,
                              font=dict(size=12 if var == "self_exceeds_360_manager_median T-1" \
                                  else 10),
                              margin=dict(l=0, b=0))
    return subplot_fig


def get_fixed_subplot_titles(var, group_level_cols):
    self_exceeds_reviewers_titles = (surround_text(f"vs. Attrition"), \
                                     surround_text(f"vs. Year"),
                                     surround_text("vs. Latest Median 360 Performance Review"))
    if var in group_level_cols:
        if "Department" in var:
            group_level_name = "vs. Turnover Rate by Department, Region"
        else:
            group_level_name = "vs. Turnover Rate by Supervisor"
        group_titles = (surround_text("vs. Attrition"), \
                        surround_text("vs. Year"),
                        surround_text(group_level_name))
    other_titles = (surround_text("vs. Attrition"), \
                    surround_text("vs. Year"))
    years_in_title_titles = (surround_text("vs. Attrition"), \
                             surround_text("vs. Year"),
                             surround_text("vs. Self > Median (360, Manager)"))
    age_titles = (surround_text("vs. Attrition"), \
                  surround_text("vs. Year"),
                  surround_text("vs. Gender (All Associates, VPs, Dirs)"))

    comp_titles = (surround_text("vs. Attrition"), \
                   surround_text("vs. Year"),
                   surround_text("vs. Latest Median 360 Performance Review"),
                   surround_text("vs. Self > Median (360, Manager)"))

    if var == "self_exceeds_360_manager_median T-1":
        return self_exceeds_reviewers_titles
    elif var == "Years In Title":
        return years_in_title_titles
    elif var == "Age":
        return age_titles
    elif var in {"Comp Tier T-1", "YOY TC % Chg (bin) T-1", "Comp Tier Change T-1"}:
        return comp_titles
    elif var in group_level_cols:
        return group_titles
    else:
        return other_titles


def get_fixed_horizontal_spacing(var):
    if var == "self_exceeds_360_manager_median T-1":
        return 0.175
    elif var in {"Comp Tier T-1"}:
        return 0.075
    elif var in {"TC Mkt Data Quartile Change T-1"}:
        return 0.1225
    elif var in {"YOY TC % Chg (bin) T-1", "Comp Tier Change T-1",
                 "TC Mkt Data Quartile T-1"}:
        return 0.1225
    elif var in {"Age"}:
        return 0.125
    elif var in {"Latest % Change in Women by Supervisor", "Latest % Change in Women by Department"}:
        return 0.175
    else:
        return 0.175


def update_covariate_subplot_fig(data_to_plot, xcol, ycol, xlabels, ylabels,
                                 xaxis_title, subplot_fig, row_height, new_name, row=1, col=3):
    subplot, widths, heights, x_labels, y_labels, xcol, ycol = construct_2d_heatmap(data_to_plot,
                                                                                    xcol, ycol,
                                                                                    ylabels=ylabels,
                                                                                    xlabels=xlabels)

    for trace in range(len(subplot["data"])):
        subplot_fig.add_trace(subplot["data"][trace], row=row, col=col)

    subplot_fig = update_heatmap_subplot(subplot_fig, new_name,
                                         widths, heights, x_labels,
                                         y_labels, row, col, xaxis_title=xaxis_title)
    return subplot_fig


def get_base_subplots(data_to_plot, var, category_orderings, row_height, subplot_fig, new_name):
    if var == "self_exceeds_360_manager_median T-1":
        subplot_fig = update_covariate_subplot_fig(data_to_plot, "360 Reviewer - MEDIAN T-1",
                                                   "self_exceeds_360_manager_median T-1",
                                                   [str(num) for num in range(1, 6)],
                                                   ["Self > Reviews", "Reviews > Self"],
                                                   "Latest Median 360 Performance Review",
                                                   subplot_fig, row_height, new_name)
    elif var == "Years In Title":
        subplot_fig = update_covariate_subplot_fig(data_to_plot,
                                                   "self_exceeds_360_manager_median T-1",
                                                   "Years In Title",
                                                   ["Self > Reviews",
                                                    "Reviews > Self"],
                                                   category_orderings["Years In Title"],
                                                   # ["0.17 <-> 2.0",
                                                   #  "2.0 <-> 2.91",
                                                   #  "2.91 <-> 3.53",
                                                   #  "3.53 <-> 20.45"],
                                                   "Self > Median Review (360, Manager)",
                                                   subplot_fig, row_height, new_name)
    elif var == "Age":
        subplot_fig = update_covariate_subplot_fig(data_to_plot,
                                                   "Sex",
                                                   "Age",
                                                   ["Female", "Male"],
                                                   category_orderings["Age"],
                                                   # ['23.76 <-> 29.21', '29.21 <-> 32.07',
                                                   #  '32.07 <-> 35.23', '35.23 <-> 54.61'],
                                                   "Gender",
                                                   subplot_fig, row_height, new_name)

    # elif var in {"Comp Tier T-1", "YOY TC % Chg (bin) T-1", "Comp Tier Change T-1"}:
    elif var in {"TC Mkt Data Quartile T-1", "YOY TC % Chg (bin) T-1", "TC Mkt Data Quartile Change T-1"}:
        subplot_fig = update_covariate_subplot_fig(data_to_plot,
                                                   "360 Reviewer - MEDIAN T-1",
                                                   var,
                                                   [str(num) for num in range(1, 6)],
                                                   category_orderings[var],
                                                   "Latest Median 360 Performance Review",
                                                   subplot_fig, row_height, new_name)
        subplot_fig = update_covariate_subplot_fig(data_to_plot,
                                                   "self_exceeds_360_manager_median T-1",
                                                   var,
                                                   ["Self > Reviews", "Reviews > Self"],
                                                   category_orderings[var],
                                                   "Self > Median Review (360, Manager)",
                                                   subplot_fig, row_height, new_name, row=1, col=4)
    elif var in {'Latest % Change in Women by Supervisor',
                 'Latest % Change in Women by Department',
                 "Turnover in Group Among Juniors",
                 "Turnover in Group Among Seniors",
                 "Age Relative to Title"}:
        if "Department" in var:
            subplot_fig = update_covariate_subplot_fig(data_to_plot,
                                                       "Turnover Rate by Department, Region",
                                                       var,
                                                       category_orderings["Turnover Rate by Department, Region"],
                                                       category_orderings[var],
                                                       "Turnover Rate by Department, Region",
                                                       subplot_fig, row_height, new_name)
        else:
            subplot_fig = update_covariate_subplot_fig(data_to_plot,
                                                       "Turnover Rate by Supervisor",
                                                       var,
                                                       category_orderings["Turnover Rate by Supervisor"],
                                                       category_orderings[var],
                                                       "Turnover Rate by Supervisor",
                                                       subplot_fig, row_height, new_name)
    return subplot_fig


def get_group_level_addons(data_to_plot, var, row_width, row_height, subplot_fig,
                           subplot_addons, new_name, category_orderings):
    subplot_fig = update_covariate_subplot_fig(data_to_plot,
                                               "self_exceeds_360_manager_median T-1",
                                               var,
                                               ["Self > Reviews", "Reviews > Self"],
                                               category_orderings[var],
                                               "Self > Median Review (360, Manager)",
                                               subplot_fig, row_height, new_name, row=1, col=1)
    # subplot_fig = update_covariate_subplot_fig(data_to_plot,
    #                                            "Comp Tier T-1",
    #                                            var,
    #                                            category_orderings["Comp Tier T-1"],
    #                                            category_orderings[var],
    #                                            "Latest Comp Tier",
    #                                            subplot_fig, row_height, new_name, row=1, col=2)
    subplot_fig = update_covariate_subplot_fig(data_to_plot,
                                               "TC Mkt Data Quartile T-1",
                                               var,
                                               category_orderings["TC Mkt Data Quartile T-1"],
                                               category_orderings[var],
                                               "Latest Compensation Market Quartile",
                                               subplot_fig, row_height, new_name, row=1, col=2)
    instrument_var = " ".join(var.replace("Change in ", "").split())
    if type(data_to_plot[instrument_var].dropna().iloc[0]) != str:
        data_to_plot, category_order = fixed_preprocessing(data_to_plot, instrument_var, category_orderings)
        subplot_fig = update_covariate_subplot_fig(data_to_plot,
                                                   instrument_var,
                                                   var,
                                                   category_order,
                                                   category_orderings[var],
                                                   instrument_var,
                                                   subplot_fig, row_height,
                                                   new_name, row=1, col=3)
    else:
        subplot_fig = update_covariate_subplot_fig(data_to_plot,
                                                   instrument_var,
                                                   var,
                                                   category_orderings[instrument_var],
                                                   category_orderings[var],
                                                   instrument_var,
                                                   subplot_fig, row_height,
                                                   new_name, row=1, col=3)
    subplot_fig.update_layout(barmode="stack",
                              barnorm="percent",
                              legend_traceorder='reversed',
                              width=row_width,
                              height=row_height,
                              showlegend=False,
                              font=dict(size=10),
                              margin=dict(l=0, b=0))
    subplot_addons.append(subplot_fig)
    return subplot_addons


def get_age_subplot_addons(data_to_plot, row_width, row_height, subplot_fig, subplot_addons, new_name,
                           category_orderings):
    associates = data_to_plot[data_to_plot["Corporate Title"] == "Associate"]
    vps = data_to_plot[data_to_plot["Corporate Title"] == "Vice President"]
    directors = data_to_plot[data_to_plot["Corporate Title"] == "Director"]
    subplot_fig = update_covariate_subplot_fig(associates,
                                               "Sex",
                                               "Age",
                                               ["Female", "Male"],
                                               category_orderings["Age"],
                                               # ['23.76 <-> 29.21', '29.21 <-> 32.07',
                                               #  '32.07 <-> 35.23', '35.23 <-> 54.61'],
                                               "Gender",
                                               subplot_fig, row_height, new_name, row=1, col=1)
    subplot_fig = update_covariate_subplot_fig(vps,
                                               "Sex",
                                               "Age",
                                               ["Female", "Male"],
                                               category_orderings["Age"],
                                               # ['23.76 <-> 29.21', '29.21 <-> 32.07',
                                               #  '32.07 <-> 35.23', '35.23 <-> 54.61'],
                                               "Gender",
                                               subplot_fig, row_height, new_name, row=1, col=2)
    subplot_fig = update_covariate_subplot_fig(directors,
                                               "Sex",
                                               "Age",
                                               ["Female", "Male"],
                                               category_orderings["Age"],
                                               # ['29.21 <-> 32.07', '32.07 <-> 35.23', '35.23 <-> 54.61'],
                                               "Gender",
                                               subplot_fig, row_height, new_name, row=1, col=3)

    subplot_fig.update_layout(barmode="stack",
                              barnorm="percent",
                              legend_traceorder='reversed',
                              width=row_width,
                              height=row_height,
                              showlegend=False,
                              font=dict(size=10),
                              margin=dict(l=0, b=0))
    subplot_addons.append(subplot_fig)
    return subplot_addons


def get_gender_subplot_addons(data_to_plot, row_width, row_height,
                              subplot_fig, subplot_addons, category_orderings, new_name):
    subplot_fig = update_covariate_subplot_fig(data_to_plot,
                                               "Years In Title",
                                               "Sex",
                                               category_orderings["Years In Title"],
                                               ["Female", "Male"],
                                               "Years In Title",
                                               subplot_fig, row_height, new_name, row=1, col=1)
    # subplot_fig = update_covariate_subplot_fig(data_to_plot,
    #                                            "Comp Tier T-1",
    #                                            "Sex",
    #                                            category_orderings["Comp Tier T-1"],
    #                                            ["Female", "Male"],
    #                                            "Latest Comp Tier",
    #                                            subplot_fig, row_height, new_name, row=1, col=2)

    subplot_fig = update_covariate_subplot_fig(data_to_plot,
                                               "TC Mkt Data Quartile T-1",
                                               "Sex",
                                               category_orderings["TC Mkt Data Quartile T-1"],
                                               ["Female", "Male"],
                                               "Latest Compensation Market Quartile",
                                               subplot_fig, row_height, new_name, row=1, col=2)

    subplot_fig.update_layout(barmode="stack",
                              barnorm="percent",
                              legend_traceorder='reversed',
                              width=row_width,
                              height=row_height,
                              showlegend=False,
                              font=dict(size=10),
                              margin=dict(l=0, b=0))
    subplot_addons.append(subplot_fig)
    return subplot_addons


def get_subplot_addons(data_to_plot, var, row_width, row_height, subplot_addons, new_name, category_orderings):
    if var == "Age":
        subplot_age_2 = make_subplots(rows=1, cols=3,
                                      subplot_titles=(surround_text("vs. Gender (Associates only)"),
                                                      surround_text("vs. Gender (VPs only)"),
                                                      surround_text("vs. Gender (Dirs only)")),
                                      horizontal_spacing=0.125)
        subplot_addons = get_age_subplot_addons(data_to_plot,
                                                row_width,
                                                row_height,
                                                subplot_age_2,
                                                subplot_addons, new_name,
                                                category_orderings)
    elif var == "Sex":

        # subplot_sex_2 = make_subplots(rows=1, cols=2,
        #                               subplot_titles=(surround_text("vs. Years In Title"),
        #                                               surround_text("vs. Latest Comp Tier")),
        #                               horizontal_spacing=0.125)
        subplot_sex_2 = make_subplots(rows=1, cols=2,
                                      subplot_titles=(surround_text("vs. Years In Title"),
                                                      surround_text("vs. Latest Compensation Market Quartile")),
                                      horizontal_spacing=0.125)
        subplot_addons = get_gender_subplot_addons(data_to_plot, row_width, row_height,
                                                   subplot_sex_2, subplot_addons, category_orderings, new_name)
    elif var in {"Latest % Change in Women by Supervisor", "Latest % Change in Women by Department"}:
        instrument_var = " ".join(var.replace("Change in ", "").split())
        # subplot_group_2 = make_subplots(rows=1, cols=3,
        #                                 subplot_titles=(surround_text("vs. Self > Median Review (360, Manager)"),
        #                                                 surround_text("vs. Latest Comp Tier"),
        #                                                 surround_text(f"vs. {instrument_var}")),
        #                                 horizontal_spacing=0.175)
        subplot_group_2 = make_subplots(rows=1, cols=3,
                                        subplot_titles=(surround_text("vs. Self > Median Review (360, Manager)"),
                                                        surround_text("vs. Latest Compensation Market Quartile"),
                                                        surround_text(f"vs. {instrument_var}")),
                                        horizontal_spacing=0.175)
        subplot_addons = get_group_level_addons(data_to_plot, var, row_width, row_height,
                                                subplot_group_2, subplot_addons,
                                                new_name, category_orderings)
    return subplot_addons


def display_visuals(original_df, df, important, unimportant, variables_renamed, category_orderings,
                    row_height, row_width, laz_colours):
    fig_dict = dict()
    fig_vars = []
    additional_cols_dict = {"self_exceeds_360_manager_median T-1": 1,
                            "Years In Title": 1,
                            "Age": 1,
                            # "Comp Tier T-1": 2,
                            "YOY TC % Chg (bin) T-1": 2,
                            "TC Mkt Data Quartile T-1": 2,
                            "TC Mkt Data Quartile Change T-1": 2}
    # "Comp Tier Change T-1": 2}
    group_level_cols = {"Latest % Change in Women by Supervisor", "Latest % Change in Women by Department",
                        "Turnover in Group Among Juniors", "Turnover in Group Among Seniors",
                        "Age Relative to Title"}
    for col in group_level_cols:
        additional_cols_dict[col] = 1
    for var in important + list(unimportant):
        data_to_plot, skip_var = retrieve_relevant_data(original_df, df, var)
        if skip_var: continue
        data_to_plot["Year"] = data_to_plot["Year"].apply(str)
        _, category_order = fixed_preprocessing(data_to_plot, var, category_orderings)
        if var not in category_orderings:
            category_orderings[var] = category_order
    for var in important + list(unimportant):
        data_to_plot, skip_var = retrieve_relevant_data(original_df, df, var)
        if skip_var: continue
        new_name = variables_renamed[var]
        data_to_plot["Year"] = data_to_plot["Year"].apply(str)
        data_to_plot, category_order = fixed_preprocessing(data_to_plot, var, category_orderings)
        color_palettes = laz_colours[:len(data_to_plot[var].dropna().unique())]
        color_seq = [f"rgb({','.join([str(int(255 * rgb_val)) for rgb_val in palette])})" \
                     for palette in color_palettes]
        subplot_num_cols = 2 + (0 if var not in additional_cols_dict \
                                    else additional_cols_dict[var])

        subplot_fig = make_subplots(rows=1, cols=subplot_num_cols,
                                    subplot_titles=get_fixed_subplot_titles(var, group_level_cols),
                                    horizontal_spacing=get_fixed_horizontal_spacing(var))
        print(var, category_order, data_to_plot[var].iloc[0], type(data_to_plot[var].iloc[0]))
        subplot_fig = display_attrition_bars_and_annual_breakdown(data_to_plot, var,
                                                                  row_width, row_height, category_order,
                                                                  color_seq, subplot_fig, new_name,
                                                                  category_orderings)

        subplot_fig = get_base_subplots(data_to_plot, var, category_orderings, row_height, subplot_fig,
                                        new_name)
        fig_vars.append((var, new_name))
        subplot_addons = [subplot_fig]
        subplot_addons = get_subplot_addons(data_to_plot, var, row_width, row_height,
                                            subplot_addons, new_name, category_orderings)
        fig_dict[new_name] = subplot_addons
    return fig_dict, fig_vars


##################
# Data Ingestion #
##################

def ingest_preprocessed_data(root_path: str, all_data_relpath: str, variable_renamed_relpath: str):
    """
    ingest_processed_data
    :param root_path: str
    :param all_data_relpath: str
    :param variable_renamed_relpath: str
    """
    original_df = pd.read_csv(f"{root_path}/{all_data_relpath}")
    df = original_df.drop_duplicates(subset=["ID", "Year"])
    df = df.drop(columns=["FTE",
                          "Category", "Empl Class",
                          "Years of Service (bin)",
                          "Bus Unit",
                          "Term Date",
                          "Hire Date",
                          "ID",
                          "Department",
                          "Functional Title",
                          "Company"])
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]
    df = add_organization_structures(df)
    df = df.drop(columns=['Firmwide Analysts/MDs', 'Firmwide Associates/MDs',
                          'Firmwide Analysts/MDs+Dirs', 'Firmwide Associates/MDs+Dirs',
                          'Firmwide Analysts+Associates/MDs+Dirs',
                          "Firmwide Analysts/Associates+VPs",
                          "Firmwide Analysts/Associates"])
    df = add_review_comparison(df)
    breaks = break_into_percentiles(df["Years In Title"])
    df["Years In Title"] = df["Years In Title"].apply(lambda x: get_category(x, breaks))
    df = add_hc_ratio_structures(df)
    df = add_compensation_features(df)
    original_df, df = add_attrition(original_df, df)

    variables_renamed = json.load(open(f"{root_path}/{variable_renamed_relpath}", "r"))
    df, latest_female_group_cols = latest_perc_women_features(df)
    return original_df, df, variables_renamed, latest_female_group_cols


def create_flight_risk_heatmaps(original_df, df, laz_colours, variables_renamed, latest_female_group_cols):
    important = set(['Turnover in Group Among Juniors',
                     'Turnover in Group Among Seniors',
                     "Age Relative to Title",
                     "Years In Title", "self_exceeds_360_manager_median T-1",
                     "Age", "Sex",
                     "TC Mkt Data Quartile T-1",
                     "TC Mkt Data Quartile Change T-1",
                     # "Comp Tier T-1",
                     "YOY TC % Chg (bin) T-1",
                     # "Comp Tier Change T-1"
                     ] + latest_female_group_cols)
    unimportant = set(variables_renamed.keys()).difference(important)
    unimportant = unimportant.difference({"Comp Tier T-1", "Comp Tier Change T-1"})
    important = ["Latest % Change in Women by Supervisor",
                 "Latest % Change in Women by Department",
                 'Turnover in Group Among Juniors',
                 'Turnover in Group Among Seniors',
                 "Age Relative to Title", "self_exceeds_360_manager_median T-1", "Years In Title", "Age",
                 "Sex",
                 "TC Mkt Data Quartile T-1",
                 "TC Mkt Data Quartile Change T-1",
                 # "Comp Tier T-1",
                 "YOY TC % Chg (bin) T-1",
                 # "Comp Tier Change T-1"
                 ]

    category_orderings = {
        # "Comp Tier T-1": ["1", "2", "3", "4", "5"],
        "Years of Service (bin)": ["0-1 Years",
                                   "1-2 Years",
                                   "2-5 Years",
                                   "5-10 Years",
                                   "10+ Years"],
        "YOY TC % Chg (bin) T-1": ["<0.0%",
                                   "0.0% - 25.0%",
                                   "25.0% - 50.0%",
                                   "50.0% - 75.0%",
                                   "75.0% - 100.0%",
                                   "100%+"],
        "TC Mkt Data Quartile T-1": ["1", "2", "3", "4"],
        "Comp Tier Change T-1": ['<-0.5', '[-0.5, 2.5)', '>2.5']}
    # "Years In Title": ["0.17 <-> 2.0", "2.0 <-> 2.91",
    #                    "2.91 <-> 3.53", "3.53 <-> 20.45"],
    # "Age Relative to Title": ['-2.79 <-> -0.89', '-0.89 <-> -0.2',
    #                           '-0.2 <-> 0.68', '0.68 <-> 3.85']}

    # add to category_orderings

    df["TC Mkt Data Quartile T-1"] = df["TC Mkt Data Quartile T-1"] \
        .apply(lambda x: x if pd.isna(x) else str(int(x)))

    for col in ["Years In Title", "Age Relative to Title",
                "TC Mkt Data Quartile Change T-1"]:
        breaks = break_into_percentiles(original_df[col])
        category_order = [str(round(breaks[i], 2)) + " <-> " + \
                          str(round(breaks[i + 1], 2)) for i in range(len(breaks) - 1)]
        breaks = original_df[col].apply(lambda x: get_category(x, breaks, \
                                                               break_vals=category_order))

        binned = bin_comp_tier_changes(original_df, col)
        if len(set(breaks.unique())) != len(breaks.unique()):
            original_df[col] = binned
            category_orderings[col] = bin_comp_tier_changes(original_df, col)
        else:

            original_df[col] = breaks

            ordered_lst = sorted(original_df[col].dropna().unique(),
                                 key=lambda s: float(s.split(" <-> ")[0]) \
                                     if len(s.split(" <-> ")) > 1 else s)

            category_orderings[col] = ordered_lst

    fig_dict, fig_vars = display_visuals(original_df, df, important, unimportant, variables_renamed, category_orderings,
                                         500, 1500, laz_colours)
    return fig_dict, fig_vars, unimportant


def _write_plotly_go_to_html(
        fig_dict, fig_vars, unimportant, root_path, output_filepath="output/Flight Risk Visuals V01.html"
):
    """
    Write list of plotly graph objects to html
    """
    divider_added = False
    with open(f"{root_path}/{output_filepath}", "w") as f:
        for original_name, new_name in fig_vars:
            if original_name in unimportant and not divider_added:
                divider_added = True
                f.write("<hr>")
            f.write(f"<h1>{new_name}</h1>")
            for figure in fig_dict[new_name]:
                f.write(figure.to_html(full_html=False, include_plotlyjs="cdn"))
        html = codecs.open(f"{root_path}/{output_filepath}", "r", "utf-8").read()
    f.close()
    return html


def traverse_path(paths, df_managers, curr_path, curr_manager, manager_col):
    direct_reports = df_managers[df_managers[manager_col] == curr_manager]["ID"].tolist()
    if len(direct_reports) == 0:
        paths.append(curr_path + [curr_manager])
    else:
        for report in direct_reports:
            traverse_path(paths, df_managers, curr_path + [curr_manager], report, manager_col)


def get_all_paths(df_managers, manager_col, root_manager="All"):
    paths = []
    traverse_path(paths, df_managers, [], curr_manager=root_manager, manager_col=manager_col)
    return paths


def manager_level_attrition_rate(x, master_df_year):
    max_idx = x.index.max()
    lowest_level_attrition = x[x.index == max_idx].iloc[0]
    while lowest_level_attrition is None and max_idx >= 0:
        lowest_level_attrition = x[x.index == max_idx].iloc[0]
        max_idx -= 1

    id_matches = master_df_year[master_df_year["ID"] == lowest_level_attrition]
    assert len(id_matches) == 1, f"Not exactly one ID match: {x}: {len(id_matches)}"
    return id_matches["Attrition"].iloc[0]


def generate_manager_treemap(original_df, manager_col, years, root_path, master_df_relpath, name_addon):
    fig_dict: Dict[Any, Any] = dict()
    master_df = pd.read_excel(f"{root_path}/{master_df_relpath}")
    df_managers = original_df[["ID", "Year", "Attrition", manager_col]]

    fig_vars = []
    for year in years:
        df_curr_managers = df_managers[df_managers["Year"] == year]
        df_curr_managers = df_curr_managers.replace("None", "All")
        master_df_year = master_df[master_df["Year"] == year]
        children_id = set(df_curr_managers["ID"])
        # We get supervisors that don't have supervisors above them.
        # Rows where manager is not among IDs

        parents_not_in_children_list = df_curr_managers[~df_curr_managers[manager_col].isin(children_id)].copy()

        parents_not_in_children_list["ID"] = "All"
        parents_not_in_children_list = parents_not_in_children_list.rename(columns={manager_col: "ID",
                                                                                    "ID": manager_col})
        df_curr_managers = pd.concat([parents_not_in_children_list, df_curr_managers])
        df_curr_managers = df_curr_managers.drop_duplicates(subset=["ID", "Year", manager_col])
        df_curr_managers = df_curr_managers[df_curr_managers["ID"] != "All"]
        df_curr_managers = df_curr_managers.reset_index(drop=True)

        paths = get_all_paths(df_curr_managers, manager_col=manager_col)
        max_path_len = max([len(path) for path in paths])
        paths = [tuple(path + [None] * (max_path_len - len(path))) for path in paths]
        paths_df = pd.DataFrame(paths).drop_duplicates()

        paths_df["Attrition % Rate"] = paths_df.apply(lambda row: manager_level_attrition_rate(row, master_df_year),
                                                      axis=1)
        paths_df["Attrition % Rate"] = paths_df["Attrition % Rate"].apply(lambda x: x * 100)
        fig = px.treemap(paths_df, path=list(paths_df.columns)[:-1],
                         hover_data=["Attrition % Rate"],
                         color="Attrition % Rate",
                         color_continuous_scale="amp")
        fig.data[0].hovertemplate = 'Attrition Rate: %{customdata[0]:.2f}%'

        fig.update_traces(root_color="lightgrey")
        fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
        treemap_name = f"Hierarchy of Managers in {year}{name_addon}"
        fig_dict[treemap_name] = [fig]
        fig_vars.append((treemap_name, treemap_name))
    return fig_dict, fig_vars, set()


def get_centroid_point(x, xcol, ycol):
    n = len(x[xcol])
    return [100 * (sum(x[xcol]) / n), 100 * (sum(x[ycol]) / n), n]


def get_quiver_cols(groupby_df, xcol, ycol):
    quiver_df = pd.DataFrame()
    xvals = groupby_df[xcol]
    yvals = groupby_df[ycol]
    pairwise_xdiff = [xvals[i + 1] - xvals[i] for i in range(len(xvals) - 1)]
    pairwise_ydiff = [yvals[i + 1] - yvals[i] for i in range(len(yvals) - 1)]
    quiver_df[xcol] = xvals[:-1]
    quiver_df[ycol] = yvals[:-1]
    quiver_df["u"] = pairwise_xdiff
    quiver_df["v"] = pairwise_ydiff
    return quiver_df


def get_groupby_df(centroid_points, xcol, ycol, curr_key):
    keys = centroid_points.keys()
    vals = centroid_points.values
    groupby_df = pd.DataFrame()
    groupby_df[curr_key] = keys
    groupby_df[xcol] = [val[0] for val in vals]
    groupby_df[ycol] = [val[1] for val in vals]
    groupby_df["Size"] = [val[2] for val in vals]
    return groupby_df


def plot_department_charts(original_df, xcol, ycol, laz_colours, topk=10):
    """
    plot_department_charts - plot the top 10 departments by size.
    # Obtain Latest Statistics Plotted against Department Level values.
    # Example plots #
    # Latest % Change in Women vs Latest % Women
    # Latest % Change in Women vs. Turnover Rate
    # Latest % Women vs. Turnover Rate

    :param original_df: pd.DataFrame
    """
    fig_dict = dict()
    fig_vars = []
    df_local = original_df.copy()
    # top 10 in earliest year
    start_year, end_year = df_local["Year"].min(), df_local["Year"].max()
    df_local = df_local.dropna(subset=[xcol, ycol])
    uniq_departments = set(df_local[df_local["Year"] == start_year]["Department"].value_counts()[:topk].keys())
    df_local = df_local[df_local["Department"].isin(uniq_departments)]
    fig = go.Figure()

    color_palettes = laz_colours[:topk]
    color_seq = [f"rgb({','.join([str(int(255 * rgb_val)) for rgb_val in palette])})" \
                 for palette in color_palettes]

    for year in range(start_year, end_year + 1):
        curr_year = df_local[df_local["Year"] == year]
        centroid_points = curr_year.groupby(["Department"]).apply(lambda x: get_centroid_point(x, xcol, ycol))
        groupby_df = get_groupby_df(centroid_points, xcol, ycol, curr_key="Department")
        fig.add_trace(go.Scatter(x=groupby_df[xcol],
                                 y=groupby_df[ycol],
                                 marker=dict(color=color_seq,
                                             size=[35] * len(color_seq)),
                                 mode="markers",
                                 name=year,
                                 hovertemplate=
                                 f'<b>{ycol}</b>:' + ' %{y:.2f}' + "%" +
                                 f'<br><b>{xcol}</b>:' + ' %{x:.2f}' + '%<br>' +
                                 '<b>%{text}</b>' + f'<br><b>Year: {year}</b>',
                                 text=[f"Size of Group: {txt}<br>Year: {dept}" for dept, txt in \
                                       zip(groupby_df["Department"], groupby_df["Size"])]))
    for i, dept in enumerate(uniq_departments):
        curr_dept = df_local[df_local["Department"] == dept]
        centroid_points = curr_dept.groupby(["Year"]).apply(lambda x: get_centroid_point(x, xcol, ycol))
        groupby_df = get_groupby_df(centroid_points, xcol, ycol, curr_key="Year")
        groupby_df = groupby_df.sort_values("Year")
        quiver_df = get_quiver_cols(groupby_df, xcol, ycol)
        fig.add_trace(ff.create_quiver(quiver_df[xcol], quiver_df[ycol], quiver_df["u"],
                                       quiver_df["v"], scale=1, arrow_scale=0.1, name=dept + " Progression",
                                       legendgroup=dept, line_width=3,
                                       marker=dict(color=color_seq[i])).data[0])
        fig.add_trace(go.Scatter(x=groupby_df[xcol],
                                 y=groupby_df[ycol],
                                 marker=dict(color=color_seq[i],
                                             size=35),
                                 mode="markers",
                                 name=dept,
                                 legendgroup=dept,
                                 hovertemplate=
                                 f'<b>{ycol}</b>:' + ' %{y:.2f}' + "%" +
                                 f'<br><b>{xcol}</b>:' + ' %{x:.2f}' + '%<br>' +
                                 '<b>%{text}</b>' + f'<br><b>Department: {dept}</b>',
                                 text=[f"Size of Group: {txt}<br>Year: {year}" for year, txt in zip(groupby_df["Year"],
                                                                                                    groupby_df[
                                                                                                        "Size"])]))
    plt_title = f"{ycol} vs. {xcol} ({start_year}-{end_year})"
    fig.update_layout(title=plt_title)
    fig.update_xaxes(title=xcol)
    fig.update_yaxes(title=ycol)
    fig.show()
    fig_dict[plt_title] = [fig]
    fig_vars.append((plt_title, plt_title))
    return fig_dict, fig_vars, set()


def prepare_for_department_charts(original_df):
    dept_df = original_df.copy()
    for col in ["Latest % Change in Women by Department", "Latest % Women by Department"]:
        historical_col = "Historical " + col.split("Latest ")[-1]
        dept_df[col] = dept_df[historical_col].apply(lambda x: float(x[1:-1].split(", ")[-1]) if not pd.isna(x) else x)
    return dept_df


#########################################
# Focusing on Exit Interview Sentiment  #
# - may want to generalize treemap code #
#########################################

def ingest_exit_interviews(root_path, rel_path, sheet_names=[]):
    exit_dfs = []
    for name in sheet_names:
        curr_df = pd.read_excel(f"{root_path}/{rel_path}", name)
        curr_df["Exit Dates"] = name
        for col in curr_df:
            if "Reason for Leaving" in col:
                curr_df[col] = curr_df[col].replace(np.nan, 0)
        curr_df["Negative Reasons"] = curr_df.apply(lambda row: row["Reason for Leaving: I don't feel valued here."] + \
                                                                row[
                                                                    "Reason for Leaving: I am not satisfied with executive leadership."] + \
                                                                row[
                                                                    "Reason for Leaving: I am not satisfied with my manager."] + \
                                                                row[
                                                                    "Reason for Leaving: I am not able to successfully balance my work and personal life."] + \
                                                                row[
                                                                    "Reason for Leaving: I am not satisfied with Lazard's benefit and wellness offerings."],
                                                    axis=1)

        curr_df["Other Reasons"] = curr_df.apply(lambda row: \
                                                     row["Reason for Leaving: I received a better opportunity."] + \
                                                     row["Reason for Leaving: I am changing career paths."] + \
                                                     row["Reason for Leaving: I am moving to a different industry."] + \
                                                     row[
                                                         "Reason for Leaving: I am leaving for a reason that is not within Lazard's control."] + \
                                                     row["Reason for Leaving: I am retiring."] + row[
                                                         "Reason for Leaving: Other."], axis=1)
        exit_dfs.append(curr_df[~pd.isna(curr_df["Country"])])

    return pd.concat(exit_dfs).reset_index(drop=True)


def display_exit_interview_sentiment(exit_df, list_of_groupby_cols=[["Department"]], color_cols=["Belonging"],
                                     color_scales=["Blues"]):
    assert len(color_scales) == len(color_cols), "Number of color scales don't match up to color columns."
    fig_dict = dict()
    fig_vars = []
    print(exit_df)
    for groupby_cols in list_of_groupby_cols:
        for i, color_col in enumerate(color_cols):
            fig = px.treemap(exit_df, path=groupby_cols,
                             hover_data=[color_col],
                             color=color_col,
                             color_continuous_scale=color_scales[i])
            # fig.data[0].hovertemplate = 'Average Sense of Belonging: %{customdata[0]:.2f}%'
            fig.update_traces(root_color="lightgrey")
            fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
            treemap_name = f"Exit Interviewees by {', '.join(groupby_cols)} - {color_col}"
            fig_dict[treemap_name] = [fig]
            fig_vars.append((treemap_name, treemap_name))
    return fig_dict, fig_vars, set()


# Displaying shapley plots.


@st.cache_data
def cache_agent(data):
    """
    cache_agent
    :param data pd.DataFrame:
    :param date str:
    :return:
    """
    # Temperature is set to 0 to be absolutely deterministic.
    agent = create_pandas_dataframe_agent(OpenAI(temperature=0), data, verbose=True)
    return agent


def query_to_guide_eda(root_path, rel_path, sheetname, queries):
    outputs = []
    for query in queries:
        data = pd.read_excel(f"{root_path}/{rel_path}", sheetname)
        agent = cache_agent(data)
        outputs.append(agent.run(query))
    return outputs


if __name__ == "__main__":
    root_path = extract_root_fpath(root_name="llama")
    ######
    # Experimenting with langchain
    ######
    # queries = ["List the columns that are associated with Attrition.",
    #            "Identify pairs of columns such that the groups formed by cutting the data " +
    #            "by these columns have maximized differences in the proportion of Attrition==1."]
    # answers = query_to_guide_eda(root_path, rel_path="data/ref/HR Flight Risk Predictors - FA - Pre-2023.xlsx",
    #                              sheetname="Flight Risk Predictors - FA",
    #                              queries=queries)
    # assert len(answers) == len(queries), "Number of answers not equal to the number of questions"
    # for i in range(len(queries)):
    #     print(f"{queries[i]}: {answers[i]}\n")

    #######################
    # Exit Interview Data #
    #######################
    exit_df = ingest_exit_interviews(root_path, rel_path="data/ref/ExitMaster_v2.xlsx",
                                     sheet_names=["8.11.22-3.31.23"])
    fig_dict, fig_vars, unimportant = display_exit_interview_sentiment(exit_df,
                                                                       list_of_groupby_cols=[["Department"],
                                                                                     ["Bus Unit"],
                                                                                     ["Business Area"],
                                                                                     ["Country"],
                                                                                     ["Corporate Title"],
                                                                                     ["Gender"]],
                                                                       color_cols=["Belonging",
                                                                                   "Reason for Leaving: I don't feel valued here.",
                                                                                   "Reason for Leaving: I am not satisfied with executive leadership.",
                                                                                   "Reason for Leaving: I am not satisfied with my manager.",
                                                                                   "Reason for Leaving: I am not able to successfully balance my work and personal life.",
                                                                                   "Reason for Leaving: I am not satisfied with Lazard's benefit and wellness offerings."
                                                                                   ],
                                                                       color_scales=["Blues", "amp",
                                                                                     "amp",
                                                                                     "amp",
                                                                                     "amp",
                                                                                     "amp"])

    #####
    # Department Level Charts and Manager Treemap
    #####
    # original_df, df, variables_renamed, latest_female_group_cols = \
    #     ingest_preprocessed_data(root_path=root_path,
    #                              all_data_relpath="data/raw/df_all_perf_comps-am.csv",
    #                              variable_renamed_relpath="data/ref/variables_renamed.json")
    # dept_df = prepare_for_department_charts(original_df)
    # fig_dict, fig_vars, unimportant = plot_department_charts(dept_df, "Latest % Change in Women by Department",
    #                                                          "Latest % Women by Department", laz_colours)
    # fig_dict, fig_vars, unimportant = generate_manager_treemap(original_df, manager_col="Supv ID",
    #                                                            years=list(range(2018, 2023)),
    #                                                            root_path=root_path,
    #                                                            master_df_relpath="data/ref/all_employees_base.xlsx",
    #                                                            name_addon=" - FA")

    # original_df = original_df.drop(columns=[col for col in original_df.columns if "Comp Tier" in col])

    ######
    # Baseline heatmap visuals
    ######

    # Plots the heatmap visuals.
    # fig_dict, fig_vars, unimportant = \
    #     create_flight_risk_heatmaps(original_df, df, laz_colours, variables_renamed, latest_female_group_cols)

    html = _write_plotly_go_to_html(fig_dict, fig_vars, unimportant, root_path,
                                    output_filepath="output/Exit Interviews Visuals 2023-04-24.html")
