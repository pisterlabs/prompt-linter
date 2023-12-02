import json
import pandas as pd
from openai_insulin_cleanup import prep_openai_from_key, convert_notes
from utils import save_as_json


def extract_data(path):
    with open(path, 'r') as fh:
        return [json.loads(l) for l in fh.readlines()]


def enrich(rows, ml_version):
    """Enrich any lines that need it."""
    lines_missing_enrichment = [r for r in rows if r.get("ParsedNotesVersion") != ml_version and not pd.isna(r.get('Notes'))]
    lines_with_enrichment = [r for r in rows if r.get("ParsedNotesVersion") == ml_version or pd.isna(r.get('Notes'))]

    lines_missing_enrichment = list(convert_notes(lines_missing_enrichment))
    all_lines = lines_missing_enrichment + lines_with_enrichment

    # sort!
    all_lines.sort(key=lambda x: x['timestamp'])
    return all_lines



if __name__ == "__main__":
    creds = json.loads(open("../../creds.json", "r").read())
    prep_openai_from_key(creds["openai"])

    rows = extract_data("/tmp/clean1.json")
    current_version = "1.0.0_tdv003"

    rows = enrich(rows, current_version)

    save_as_json(rows, "/tmp/clean2.json")

    # laptop run: {'completion_tokens': 9161, 'prompt_tokens': 10138, 'total_tokens': 19299}