from typing import Dict
import pandas as pd
import numpy as np
import pytz
from datetime import datetime
import json

import re
from openai_insulin_cleanup import prep_openai_from_key, get_drug_conversions
from utils import save_as_json

run_local = False


def read_sheet(sheet_uri : str) -> Dict[str, pd.DataFrame]:
    sheet_names = ["glucose"]
    return {n: pd.read_excel(sheet_uri, engine='openpyxl') for n in sheet_names}


def cleanup_date(raw_date : np.dtype('O')) -> datetime | float:
    """
    My sheet is gross apparently. We have the following known formats:
    YYYY-DD-MM 00:00:00   (yes month/day reversed)
    MM/DD/YYYY
    MM:DD/YYYY (typo)
    """
    if pd.isna(raw_date):
        return raw_date
    date_s = str(raw_date)

    re1 = r'^(?P<year>\d{4})-(?P<day>\d{2})-(?P<month>\d{2}) 00:00:00$'
    m = re.match(re1, date_s)
    if not m:
        re2 = r'^(?P<month>\d{1,2})[:/](?P<day>\d{2})[:/](?P<year>\d{4})$'
        m = re.match(re2, date_s)
    if not m:
        raise Exception(f"Unable to convert date {raw_date}")

    return datetime(int(m.group('year')), int(m.group('month')), int(m.group('day')))


def clean_glucose_as_pandas(glucose_df : pd.DataFrame) -> pd.DataFrame:
    
    df = glucose_df[['Date', 'Time', 'Type', 'Units', 'Notes']]
    df.loc[:, 'Date'] = df.Date.apply(cleanup_date)
    df.loc[:, 'Date'] = df.Date.interpolate(method='pad')

    def fix_timezones(s):
        if pd.isna(s):
            return s
        s = s.lower()
        if 'time zone' in s:
            return s.replace('time zone', '').strip().upper()
        return np.nan

    df["timezone"] = df.Notes.apply(fix_timezones)
    df.loc[0, 'timezone'] = 'PST' # fix first value

    df['had_timezone'] = pd.isna(df.timezone) == False

    df.timezone = df.timezone.interpolate(method='pad')

    tz_correct = {
        'PST': 'America/Los_Angeles',
        'CST': 'US/Central'
    }

    def build_timestamp(row):
        s = f"{row['Date'].strftime('%Y-%m-%d')}T{row['Time'].isoformat()}"
        tz = pytz.timezone(tz_correct.get(row['timezone'], row['timezone']))
        d = datetime.strptime(s, '%Y-%m-%dT%H:%M:00')
        return tz.localize(d)

    df['timestamp'] = df.apply(build_timestamp, axis=1)
    return df


def convert_to_dicts(df : pd.DataFrame):
    cols = df.columns
    for row in df.to_records(index=False):
        yield {k: v for k, v in zip(cols, row)}


def clean_type(rows, raw_insulin_types):
    """Clean insulin type"""
    insulin_conversions, conversion_version = get_drug_conversions(raw_insulin_types)
    for row in rows:
        try:
            row["CleanType"] = insulin_conversions[row["Type"]]
        except KeyError:
            if run_local:
                import ipdb; ipdb.set_trace()
            else:
                raise

        row["CleanTypeVersion"] = conversion_version
    return rows



if __name__ == "__main__":
    run_local = True
    creds = json.loads(open("../../creds.json", "r").read())
    prep_openai_from_key(creds["openai"])

    sheets = read_sheet("/tmp/local_copy.xlsx")
    glucose = clean_glucose_as_pandas(sheets['glucose'])
    raw_insulin_types = glucose.Type.unique()
    raw_rows = list(convert_to_dicts(glucose))
    rows = clean_type(raw_rows, raw_insulin_types)
    save_as_json(rows, "/tmp/clean1.json")
