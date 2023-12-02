"""Process  census metadata with OpenAI
"""

__author__ = "eric@civicknowledge.com"


def rewrite_path(d):
    """Call the OpenAI completions interface to re-write the extra path for a census variable
    into an English statement that can be used to describe the variable."""

    import os

    import openai

    openai.api_key = os.getenv("OPENAI_API_KEY")

    var_disc = "\n".join(f"{unique_id}: '{path}'" for unique_id, path in d)

    prompt = f"""
This text describes a variable in a dataset, with a unique id and a path:

  B06007: '/current residence; south/movers to different state'

These strings are to be converted to YAML blocks in a list that describes the variable, along with a shorter
identifying name for the variable, the variable name should be lowercase, using underscores instead of spaces,
and the output is this YAML  format for the reponse:

'''

- unique_id: <unique id>
  path: "<path>"
  name: "<variable_name>"
  description: "People who <description>"
'''

Rewrite the following variable descriptions as YAML blocks:

{var_disc}
          """

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    return response


def get_census_metadata(year=2020, release=5):
    from publicdata.census.files.metafiles import TableLookup

    tl = TableLookup(year, release)

    tdf = tl.tables_df
    cdf = tl.columns_df

    return tdf, cdf


def census_path_batch_tasks(cdf, db):
    """Create a set of chunked tasks, excluding those that are already in the database"""

    d = []
    from slugify import slugify

    for gname, g in cdf[cdf.filtered_path != ""].groupby("table_id"):
        for fp in g.filtered_path.unique():
            k = slugify(f"{gname}-{fp}")
            if fp and k not in db:
                d.append((gname, fp))

    return d


def write_responses(response, db):
    """
    :param response: Response from OpenAI
    :type response:
    :param db: Database for storing results
    :type db: dict-like, but probably a shelve
    :return:
    :rtype:

    Read the response from rewrite_path, extract each records, and save the records
    to a shelve database, using the unique_id as the key"""

    import yaml
    from slugify import slugify

    response_objs = yaml.safe_load(response["choices"][0]["text"])

    for i, o in enumerate(response_objs):
        k = slugify(f"{o['unique_id']}-{o['path']}")
        db[k] = o
