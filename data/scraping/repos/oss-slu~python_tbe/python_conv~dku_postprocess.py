import pandas as pd
from plotnine import ggplot, geom_line, aes, theme, element_text
from openair import timeVariation, summaryPlot

# Set the directory and file name for the input data
indir = ""
flroot = "dku_bluesky_rajshahi_ns4_20220421_20230501_coords_hr_lt"
flin = f"{indir}{flroot}.csv"
flrootout = f"p{flroot}"

# Set the seasons and time range
seasonsct = "Dry"  # NULL, 'Dry', 'Wet'
tb_season = pd.DataFrame({"name": ["Dry", "Wet", "Dry"], "start_mmdd": [1, 401, 1101]})

timeminsct = None
timemaxsct = None

timezone_output = "Asia/Dhaka"

count_min = 3

domatch = 1
doallsites = 0
doallseasons = 1

dowritecsv = 0

fgwidth = 1000
fgheight = 800

# Read the data
tb = pd.read_csv(flin)

# Set time zone
tb["date"] = pd.to_datetime(tb["date"])
tb["date"] = tb["date"].dt.tz_localize(timezone_output)

print(f"Read {len(tb)} rows from {min(tb['date'])} to {max(tb['date'])}")

# Create column with seasons
tb["season0"] = pd.cut(
    100 * tb["date"].dt.month + tb["date"].dt.day,
    bins=[0] + list(tb_season["start_mmdd"]) + [1231],
    labels=tb_season["name"],
)

# Filter data by start/end time
if timeminsct is not None:
    tb = tb[tb["date"] >= timeminsct]
    if len(tb) == 0:
        raise ValueError(f"No data found, check data selection in timeminsct: {timeminsct}")
    else:
        print(f"Filtered tb by timeminsct {timeminsct}: {len(tb)} rows from {min(tb['date'])} to {max(tb['date'])}")

if timemaxsct is not None:
    tb = tb[tb["date"] <= timemaxsct]
    if len(tb) == 0:
        raise ValueError(f"No data found, check data selection in timemaxsct: {timemaxsct}")
    else:
        print(f"Filtered tb by timemaxsct {timemaxsct}: {len(tb)} rows from {min(tb['date'])} to {max(tb['date'])}")

print(f"Using {len(tb)} rows from {min(tb['date'])} to {max(tb['date'])}")

# Filter data by season
if seasonsct is not None:
    tb = tb[tb["season0"] == seasonsct]
    if len(tb) == 0:
        raise ValueError(f"No data found, check data selection in seasonsct: {seasonsct}")
    else:
        print(f"Filtered tb by season {seasonsct}: {len(tb)} rows from {min(tb['date'])} to {max(tb['date'])}")

# Filter data by pm25count
if count_min is not None:
    tb = tb[tb["pm25count"] >= count_min]
    if len(tb) == 0:
        raise ValueError(f"No data left after screening for pm25count >= {count_min} (count_min)")
    else:
        print(
            f"Filtered tb by pm25count >= {count_min} (count_min), {len(tb)} rows from {min(tb['date'])} to {max(tb['date'])}"
        )

# Matching data: do only times which have valid data for all sites
if domatch:
    tb_wide = tb.pivot(index="date", columns="sitename", values="pm25")
    asites = tb["sitename"].unique()
    tb_wide = tb_wide.dropna(subset=asites, how="any")
    nrow_allvalid = len(tb_wide)
    nrow_all = len(tb_wide) + tb["date"].isna().sum()
    print(f"{nrow_allvalid}/{nrow_all} rows with valid data for all sites")
    if nrow_allvalid == 0:
        raise ValueError("No valid data found, stop here")

    tb = tb[tb["date"].isin(tb_wide.index)]

    if dowritecsv:
        flcsv = f"{flrootout}_postprocess_wide.csv"
        tb_wide.to_csv(flcsv)
        print(f"Wrote data with one column per site (tb_wide) to {flcsv} for QAQC")

# Date string for plot titles
date_str = f"{min(tb['date']).date()} to {max(tb['date']).date()}"
date_str_flname = f"{min(tb['date']).strftime('%Y%m%d')}_{max(tb['date']).strftime('%Y%m%d')}"
if seasonsct is not None:
    date_str = f"{date_str_flname} ({seasonsct})"
    date_str_flname = f"{date_str_flname}_{seasonsct.lower()}"

# Summary time series, histogram, and metrics
summaryPlot(tb.rename(columns={"sitename": "site"}), pollutant="pm25")
flout = f"{flrootout}_summary.png"
print(f"Summary plot created: {flout}")

# Plot time series
plts = (
    ggplot(tb)
    + geom_line(aes(x="date", y="pm25", color="sitename"))
    + theme(axis_text_x=element_text(rotation=90, hjust=1))
)
print(plts)

# Plot temporal variations by day, week, month: overlay sites
ptv = timeVariation(
    tb,
    pollutant="pm25",
    group="sitename",
    normalise=False,
    key_columns=2,
    statistic="median",
    conf_int=[0.75],
    main=f"{varsct} ({varunits}) by site\n{date_str}",
)
print(ptv)

# Time Variation plot for each site one by one
if doallsites:
    asites = tb["sitename"].unique()
    for sitesct in asites:
        siteid = saq_sitename2id(sitesct)
        ptv = timeVariation(
            tb[tb["sitename"] == sitesct],
            pollutant="pm25",
            group="season0",
            normalise=False,
            main=f"{varsct} ({varunits}) for site {sitesct}\n{date_str}",
        )
        flout = f"{flrootout}_tvsite_{siteid}.png"
        print(f"Time Variation plot created: {flout}")

# Time Variation plot for each season one by one
if doallseasons:
    aseasons = tb["season0"].unique()
    for seasonsct in aseasons:
        ptv = timeVariation(
            tb[tb["season0"] == seasonsct],
            pollutant="pm25",
            group="sitename",
            normalise=False,
            main=f"{varsct} ({varunits}) for season {seasonsct}\n{date_str}",
        )
        flout = f"{flrootout}_tvseason_{seasonsct.lower()}.png"
        print(f"Time Variation plot created: {flout}")

if dowritecsv:
    flcsv = f"{flrootout}_postprocess_qa.csv"
    tb.to_csv(flcsv)
    print(f"Wrote data tibble (tb) to {flcsv} for QAQC")
