"""Pull data from API and build data frame."""

import numpy as np
import pandas as pd

from openair_cologne.influx import query_influx

LAST_N_DAYS = 250

# %% DATA FROM LANUV STATIONS
lanuv_dict = query_influx("SELECT station, NO2 AS no2 "
                          "FROM lanuv_f2 "
                          f"WHERE time >= now() - "f"{LAST_N_DAYS}d "
                          "AND time <= now() ")

# make clean data frame
df_lanuv = lanuv_dict['lanuv_f2'] \
    .rename_axis('timestamp').reset_index()

df_lanuv = df_lanuv[df_lanuv.timestamp.dt.minute == 0]
df_lanuv = df_lanuv[df_lanuv.timestamp.dt.second == 0]

df_lanuv = df_lanuv.assign(
    timestamp=pd.to_datetime(df_lanuv.timestamp.astype(np.int64) // 10 ** 6,
                             unit='ms',
                             utc=True))

# %% DATA FROM OPENAIR
openair_dict = query_influx("SELECT "
                            "median(hum) AS hum, median(pm10) AS pm10, "
                            "median(pm25) AS pm25, median(r1) AS r1, "
                            "median(r2) AS r2, median(rssi) AS rssi, "
                            "median(temp) AS temp "
                            "FROM all_openair "
                            f"WHERE time >= now() - {LAST_N_DAYS}d "
                            "AND time <= now() "
                            "GROUP BY feed, time(1h) fill(-1)")
# clean dictionary keys
openair_dict_clean = {k[1][0][1]: openair_dict[k]
                      for k in openair_dict.keys()}

# initialize empty data frame
df_openair = pd.DataFrame()
# fill data frame with data from all frames
# OPTIONAL: replace for-loop with map-reduce
for feed in list(openair_dict_clean.keys()):
    df_feed = pd.DataFrame.from_dict(openair_dict_clean[feed]) \
        .assign(feed=feed) \
        .rename_axis('timestamp').reset_index()
    df_openair = df_openair.append(df_feed)

# shift timestamp one hour into the future
df_openair_shifted = df_openair \
    .assign(timestamp=lambda d: d.timestamp + pd.Timedelta('1h'))

# %% WRITE RESULTS
df_lanuv.to_parquet('data/df_lanuv.parquet')
df_openair_shifted.to_parquet('data/df_openair.parquet')
