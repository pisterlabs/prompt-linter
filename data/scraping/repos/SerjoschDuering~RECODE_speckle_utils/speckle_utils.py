#speckle utils
import json 
import pandas as pd
import numpy as np
import specklepy
from specklepy.api.client import SpeckleClient
from specklepy.api.credentials import get_default_account, get_local_accounts
from specklepy.transports.server import ServerTransport
from specklepy.api import operations
from specklepy.objects.geometry import Polyline, Point, Mesh

from specklepy.api.wrapper import StreamWrapper
try:
    import openai
except:
    pass

import requests
from datetime import datetime
import copy


# HELP FUNCTION ===============================================================
def helper():
    """
    Prints out the help message for this module.
    """
    print("This module contains a set of utility functions for speckle streams.")
    print("______________________________________________________________________")
    print("It requires the specklepy package to be installed -> !pip install specklepy")
    print("the following functions are available:")
    print("getSpeckleStream(stream_id, branch_name, client)")
    print("getSpeckleGlobals(stream_id, client)")
    print("get_dataframe(objects_raw, return_original_df)")
    print("updateStreamAnalysis(stream_id, new_data, branch_name, geometryGroupPath, match_by_id, openai_key, return_original)")
    print("there are some more function available not documented fully yet, including updating a notion database")
    print("______________________________________________________________________")
    print("for detailed help call >>> help(speckle_utils.function_name) <<< ")
    print("______________________________________________________________________")
    print("standard usage:")
    print("______________________________________________________________________")
    print("retreiving data")
    print("1. import speckle_utils & speckle related libaries from specklepy")
    print("2. create a speckle client -> client = SpeckleClient(host='https://speckle.xyz/')" )
    print("                              client.authenticate_with_token(token='your_token_here')")
    print("3. get a speckle stream -> stream = speckle_utils.getSpeckleStream(stream_id, branch_name, client)")
    print("4. get the stream data -> data = stream['pth']['to']['data']")
    print("5. transform data to dataframe -> df = speckle_utils.get_dataframe(data, return_original_df=False)")
    print("______________________________________________________________________")
    print("updating data")
    print("1. call updateStreamAnalysis --> updateStreamAnalysis(new_data, stream_id, branch_name, geometryGroupPath, match_by_id, openai_key, return_original)")


#==============================================================================

def getSpeckleStream(stream_id,
                     branch_name,
                     client,
                     commit_id=""
                     ):
    """
    Retrieves data from a specific branch of a speckle stream.

    Args:
        stream_id (str): The ID of the speckle stream.
        branch_name (str): The name of the branch within the speckle stream.
        client (specklepy.api.client.Client, optional): A speckle client. Defaults to a global `client`.
        commit_id (str): id of a commit, if nothing is specified, the latest commit will be fetched

    Returns:
        dict: The speckle stream data received from the specified branch.

    This function retrieves the last commit from a specific branch of a speckle stream.
    It uses the provided speckle client to get the branch and commit information, and then 
    retrieves the speckle stream data associated with the last commit.
    It prints out the branch details and the creation dates of the last three commits for debugging purposes.
    """

    print("updated A")

    # set stream and branch
    try:
        branch = client.branch.get(stream_id, branch_name, 3)
        print(branch)
    except:
        branch = client.branch.get(stream_id, branch_name, 1)
        print(branch)

    print("last three commits:")
    [print(ite.createdAt) for ite in branch.commits.items]

    if commit_id == "":
        latest_commit = branch.commits.items[0]
        choosen_commit_id = latest_commit.id
        commit = client.commit.get(stream_id, choosen_commit_id)
        print("latest commit ", branch.commits.items[0].createdAt, " was choosen")
    elif type(commit_id) == type("s"): # string, commit uuid
        choosen_commit_id = commit_id
        commit = client.commit.get(stream_id, choosen_commit_id)
        print("provided commit ", choosen_commit_id, " was choosen")
    elif type(commit_id) == type(1): #int 
        latest_commit = branch.commits.items[commit_id]
        choosen_commit_id = latest_commit.id
        commit = client.commit.get(stream_id, choosen_commit_id)


    print(commit)
    print(commit.referencedObject)
    # get transport
    transport = ServerTransport(client=client, stream_id=stream_id)
    #speckle stream
    res = operations.receive(commit.referencedObject, transport)

    return res
 
def getSpeckleGlobals(stream_id, client):
    """
    Retrieves global analysis information from the "globals" branch of a speckle stream.

    Args:
        stream_id (str): The ID of the speckle stream.
        client (specklepy.api.client.Client, optional): A speckle client. Defaults to a global `client`.

    Returns:
        analysisInfo (dict or None): The analysis information retrieved from globals. None if no globals found.
        analysisGroups (list or None): The analysis groups retrieved from globals. None if no globals found.

    This function attempts to retrieve and parse the analysis information from the "globals" 
    branch of the specified speckle stream. It accesses and parses the "analysisInfo" and "analysisGroups" 
    global attributes, extracts analysis names and UUIDs.
    If no globals are found in the speckle stream, it returns None for both analysisInfo and analysisGroups.
    """
    # get the latest commit
    try:
        # speckle stream globals
        branchGlob = client.branch.get(stream_id, "globals")
        latest_commit_Glob = branchGlob.commits.items[0]
        transport = ServerTransport(client=client, stream_id=stream_id)

        globs = operations.receive(latest_commit_Glob.referencedObject, transport)
        
        # access and parse globals
        #analysisInfo = json.loads(globs["analysisInfo"]["@{0;0;0;0}"][0].replace("'", '"'))
        #analysisGroups = [json.loads(gr.replace("'", '"')) for gr in globs["analysisGroups"]["@{0}"]]

        def get_error_context(e, context=100):
            start = max(0, e.pos - context)
            end = e.pos + context
            error_line = e.doc[start:end]
            pointer_line = ' ' * (e.pos - start - 1) + '^'
            return error_line, pointer_line

        try:
            analysisInfo = json.loads(globs["analysisInfo"]["@{0;0;0;0}"][0].replace("'", '"').replace("None", "null"))
        except json.JSONDecodeError as e:
            print(f"Error decoding analysisInfo: {e}")
            error_line, pointer_line = get_error_context(e)
            print("Error position and surrounding text:")
            print(error_line)
            print(pointer_line)
            analysisInfo = None

        try:
            analysisGroups = [json.loads(gr.replace("'", '"').replace("None", "null")) for gr in globs["analysisGroups"]["@{0}"]]
        except json.JSONDecodeError as e:
            print(f"Error decoding analysisGroups: {e}")
            error_line, pointer_line = get_error_context(e)
            print("Error position and surrounding text:")
            print(error_line)
            print(pointer_line)
            analysisGroups = None



        # extract analysis names 
        analysis_names = []
        analysis_uuid = []
        [(analysis_names.append(key.split("++")[0]),analysis_uuid.append(key.split("++")[1]) ) for key in analysisInfo.keys()]


        # print extracted results
        print("there are global dictionaries with additional information for each analysis")
        print("<analysisGroups> -> ", [list(curgrp.keys()) for curgrp in analysisGroups])
        print("<analysis_names> -> ", analysis_names)                       
        print("<analysis_uuid>  -> ", analysis_uuid)
    except Exception as e:  # catch exception as 'e'
        analysisInfo = None
        analysisGroups = None
        print("No GlOBALS FOUND")
        print(f"Error: {e}")  # print error description
  
    return analysisInfo, analysisGroups



#function to extract non geometry data from speckle 
def get_dataframe(objects_raw, return_original_df=False):
    """
    Creates a pandas DataFrame from a list of raw Speckle objects.

    Args:
        objects_raw (list): List of raw Speckle objects.
        return_original_df (bool, optional): If True, the function also returns the original DataFrame before any conversion to numeric. Defaults to False.

    Returns:
        pd.DataFrame or tuple: If return_original_df is False, returns a DataFrame where all numeric columns have been converted to their respective types, 
                               and non-numeric columns are left unchanged. 
                               If return_original_df is True, returns a tuple where the first item is the converted DataFrame, 
                               and the second item is the original DataFrame before conversion.

    This function iterates over the raw Speckle objects, creating a dictionary for each object that excludes the '@Geometry' attribute. 
    These dictionaries are then used to create a pandas DataFrame. 
    The function attempts to convert each column to a numeric type if possible, and leaves it unchanged if not. 
    Non-convertible values in numeric columns are replaced with their original values.
    """
    # dataFrame
    df_data = []
    # Iterate over speckle objects
    for obj_raw in objects_raw:
        obj = obj_raw.__dict__
        df_obj = {k: v for k, v in obj.items() if k != '@Geometry'}
        df_data.append(df_obj)

    # Create DataFrame and GeoDataFrame
    df = pd.DataFrame(df_data)
    # Convert columns to float or int if possible, preserving non-convertible values <-
    df_copy = df.copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].fillna(df_copy[col], inplace=True)

    if return_original_df:
        return df, df_copy
    else:
        return df
    

def updateStreamAnalysis(
          client,
          new_data,
          stream_id,
          branch_name,
          geometryGroupPath=None,
          match_by_id="",
          openai_key ="",
          return_original = False
      ):
  

    """
    Updates Stream Analysis by modifying object attributes based on new data.

    Args:
        new_data (pandas.DataFrame): DataFrame containing new data.
        stream_id (str): Stream ID.
        branch_name (str): Branch name.
        geometry_group_path (list, optional): Path to geometry group. Defaults to ["@Data", "@{0}"].
        match_by_id (str, optional): key for column that should be used for matching. If empty, the index is used.
        openai_key (str, optional): OpenAI key. If empty no AI commit message is generated Defaults to an empty string.
        return_original (bool, optional): Determines whether to return original speckle stream objects. Defaults to False.

    Returns:
        list:  original speckle stream objects as backup if return_original is set to True.

    This function retrieves the latest commit from a specified branch, obtains the 
    necessary geometry objects, and matches new data with existing objects using 
    an ID mapper. The OpenAI GPT model is optionally used to create a commit summary 
    message. Changes are sent back to the server and a new commit is created, with 
    the original objects returned as a backup if return_original is set to True. 
    The script requires active server connection, necessary permissions, and relies 
    on Speckle and OpenAI's GPT model libraries.
    """

    if geometryGroupPath == None:
        geometryGroupPath = ["@Speckle", "Geometry"]

    branch = client.branch.get(stream_id, branch_name, 2)

    latest_commit = branch.commits.items[0]
    commitID = latest_commit.id 

    commit = client.commit.get(stream_id, commitID)

    # get objects
    transport = ServerTransport(client=client, stream_id=stream_id)

    #speckle stream
    res = operations.receive(commit.referencedObject, transport)

    # get geometry objects (they carry the attributes)
    objects_raw = res[geometryGroupPath[0]][geometryGroupPath[1]]
    res_new = copy.deepcopy(res)

    # map ids 
    id_mapper = {}
    if match_by_id != "":
        for i, obj in enumerate(objects_raw):
            id_mapper[obj[match_by_id]] = i
    else:
        for i, obj in enumerate(objects_raw):
            id_mapper[str(i)] = i

    # iterate through rows (objects)
    for index, row in new_data.iterrows():
        #determin target object 
        if match_by_id != "":
            local_id = row[match_by_id]
        else:
            local_id = index
        target_id = id_mapper[local_id]     

        #iterate through columns (attributes)
        for col_name in new_data.columns:
            res_new[geometryGroupPath[0]][geometryGroupPath[1]][target_id][col_name] = row[col_name]


    # ======================== OPEN AI FUN ===========================
    try:
        answer_summary = gptCommitMessage(objects_raw, new_data,openai_key)
        if answer_summary == None:
            _, answer_summary = compareStats(get_dataframe(objects_raw),new_data)
    except:
        _, answer_summary = compareStats(get_dataframe(objects_raw),new_data)
    # ================================================================

    new_objects_raw_speckle_id = operations.send(base=res_new, transports=[transport])

    # You can now create a commit on your stream with this object
    commit_id = client.commit.create(
        stream_id=stream_id,
        branch_name=branch_name,
        object_id=new_objects_raw_speckle_id,
        message="Updated item in colab -" + answer_summary,
        )

    print("Commit created!")
    if return_original:
        return objects_raw #as back-up

def custom_describe(df):
    # Convert columns to numeric if possible
    df = df.apply(lambda x: pd.to_numeric(x, errors='ignore'))

    # Initial describe with 'include = all'
    desc = df.describe(include='all')

    # Desired statistics
    desired_stats = ['count', 'unique', 'mean', 'min', 'max']

    # Filter for desired statistics
    result = desc.loc[desired_stats, :].copy()
    return result

def compareStats(df_before, df_after):
  """
    Compares the descriptive statistics of two pandas DataFrames before and after some operations.

    Args:
        df_before (pd.DataFrame): DataFrame representing the state of data before operations.
        df_after (pd.DataFrame): DataFrame representing the state of data after operations.

    Returns:
        The CSV string includes column name, intervention type, and before and after statistics for each column.
        The summary string provides a count of updated and new columns.

    This function compares the descriptive statistics of two DataFrames: 'df_before' and 'df_after'. 
    It checks the columns in both DataFrames and categorizes them as either 'updated' or 'new'.
    The 'updated' columns exist in both DataFrames while the 'new' columns exist only in 'df_after'.
    For 'updated' columns, it compares the statistics before and after and notes the differences.
    For 'new' columns, it lists the 'after' statistics and marks the 'before' statistics as 'NA'.
    The function provides a summary with the number of updated and new columns, 
    and a detailed account in CSV format of changes in column statistics.
  """
   
  desc_before = custom_describe(df_before)
  desc_after = custom_describe(df_after)

  # Get union of all columns
  all_columns = set(desc_before.columns).union(set(desc_after.columns))

  # Track number of updated and new columns
  updated_cols = 0
  new_cols = 0

  # Prepare DataFrame output
  output_data = []

  for column in all_columns:
      row_data = {'column': column}
      stat_diff = False  # Track if there's a difference in stats for a column

      # Check if column exists in both dataframes
      if column in desc_before.columns and column in desc_after.columns:
          updated_cols += 1
          row_data['interventionType'] = 'updated'
          for stat in desc_before.index:
              before_val = round(desc_before.loc[stat, column], 1) if pd.api.types.is_number(desc_before.loc[stat, column]) else desc_before.loc[stat, column]
              after_val = round(desc_after.loc[stat, column], 1) if pd.api.types.is_number(desc_after.loc[stat, column]) else desc_after.loc[stat, column]
              if before_val != after_val:
                  stat_diff = True
                  row_data[stat+'_before'] = before_val
                  row_data[stat+'_after'] = after_val
      elif column in desc_after.columns:
          new_cols += 1
          stat_diff = True
          row_data['interventionType'] = 'new'
          for stat in desc_after.index:
              row_data[stat+'_before'] = 'NA'
              after_val = round(desc_after.loc[stat, column], 1) if pd.api.types.is_number(desc_after.loc[stat, column]) else desc_after.loc[stat, column]
              row_data[stat+'_after'] = after_val

      # Only add to output_data if there's actually a difference in the descriptive stats between "before" and "after".
      if stat_diff:
          output_data.append(row_data)

  output_df = pd.DataFrame(output_data)
  csv_output = output_df.to_csv(index=False)
  print (output_df)
  # Add summary to beginning of output
  summary = f"Summary:\n  Number of updated columns: {updated_cols}\n  Number of new columns: {new_cols}\n\n"
  csv_output = summary + csv_output

  return csv_output, summary



# Function to call ChatGPT API
def ask_chatgpt(prompt, model="gpt-3.5-turbo", max_tokens=300, n=1, stop=None, temperature=0.3):
    import openai
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpfull assistant,."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        n=n,
        stop=stop,
        temperature=temperature,
    )
    return response.choices[0].message['content']




def gptCommitMessage(objects_raw, new_data,openai_key):
    # the idea is to automatically create commit messages. Commits coming through this channel are all
    # about updating or adding a dataTable. So we can compare the descriptive stats of a before and after
    # data frame 
    #try:
    try:
        import openai
        openai.api_key = openai_key
    except NameError as ne:
        if str(ne) == "name 'openai' is not defined":
            print("No auto commit message: openai module not imported. Please import the module before setting the API key.")
        elif str(ne) == "name 'openai_key' is not defined":
            print("No auto commit message: openai_key is not defined. Please define the variable before setting the API key.")
        else:
            raise ne

    report, summary = compareStats(get_dataframe(objects_raw),new_data)

    # prompt
    prompt = f"""Given the following changes in my tabular data structure, generate a 
    precise and informative commit message. The changes involve updating or adding 
    attribute keys and values. The provided summary statistics detail the changes in 
    the data from 'before' to 'after'. 
    The CSV format below demonstrates the structure of the summary:

    Summary:
    Number of updated columns: 2
    Number of new columns: 1
    column,interventionType,count_before,count_after,unique_before,unique_after,mean_before,mean_after,min_before,min_after,max_before,max_after
    A,updated,800,800,2,3,,nan,nan,nan,nan,nan
    B,updated,800,800,3,3,,nan,nan,nan,nan,nan
    C,new,NA,800,NA,4,NA,nan,NA,nan,NA,nan

    For the commit message, your focus should be on changes in the data structure, not the interpretation of the content. Be precise, state the facts, and highlight significant differences or trends in the statistics, such as shifts in mean values or an increase in unique entries.

    Based on the above guidance, draft a commit message using the following actual summary statistics:

    {report}

    Your commit message should follow this structure:

    1. Brief description of the overall changes.
    2. Significant changes in summary statistics (count, unique, mean, min, max).
    3. Conclusion, summarizing the most important findings with the strucutre:
    # changed columns: , comment: ,
    # added Columns:  , comment: ,
    # Chaged statistic: ,  coment: ,

    Mark the beginning of the conclusion with ">>>" and ensure to emphasize hard facts and significant findings. 
    """

    try:
        answer = ask_chatgpt(prompt)
        answer_summery = answer.split(">>>")[1]
        if answer == None:
            answer_summery = summary
    except:
        answer_summery = summary
    return answer_summery

def specklePolyline_to_BokehPatches(speckle_objs, pth_to_geo="curves", id_key="ids"):
  """
  Takes a list of speckle objects, extracts the polyline geometry at the specified path, and returns a dataframe of x and y coordinates for each polyline.
  This format is compatible with the Bokeh Patches object for plotting.
  
  Args:
    speckle_objs (list): A list of Speckle Objects
    pth_to_geo (str): Path to the geometry in the Speckle Object
    id_key (str): The key to use for the uuid in the dataframe. Defaults to "uuid"
    
  Returns:
    pd.DataFrame: A Pandas DataFrame with columns "uuid", "patches_x" and "patches_y"
  """
  patchesDict = {"uuid":[], "patches_x":[], "patches_y":[]}
  
  for obj in speckle_objs:
    obj_geo = obj[pth_to_geo]
    obj_pts = Polyline.as_points(obj_geo)
    coorX = []
    coorY = []
    for pt in obj_pts:
      coorX.append(pt.x)
      coorY.append(pt.y)
    
    patchesDict["patches_x"].append(coorX)
    patchesDict["patches_y"].append(coorY)
    patchesDict["uuid"].append(obj[id_key])

  return pd.DataFrame(patchesDict)



def rebuildAnalysisInfoDict(analysisInfo):
    """rebuild the analysisInfo dictionary to remove the ++ from the keys

    Args:
        analysisInfo (list): a list containing the analysisInfo dictionary

    Returns:
        dict: a dictionary containing the analysisInfo dictionary with keys without the ++

    """
    analysisInfoDict = {}
    for curKey in analysisInfo[0]:
        newkey = curKey.split("++")[0]
        analysisInfoDict[newkey] = analysisInfo[0][curKey]
    return analysisInfoDict


def specklePolyline2Patches(speckle_objs, pth_to_geo="curves", id_key=None):
    """
    Converts Speckle objects' polyline information into a format suitable for Bokeh patches.

    Args:
        speckle_objs (list): A list of Speckle objects.
        pth_to_geo (str, optional): The path to the polyline geometric information in the Speckle objects. Defaults to "curves".
        id_key (str, optional): The key for object identification. Defaults to "uuid".

    Returns:
        DataFrame: A pandas DataFrame with three columns - "uuid", "patches_x", and "patches_y". Each row corresponds to a Speckle object.
                    "uuid" column contains the object's identifier.
                    "patches_x" and "patches_y" columns contain lists of x and y coordinates of the polyline points respectively.

    This function iterates over the given Speckle objects, retrieves the polyline geometric information and the object's id from each Speckle object, 
    and formats this information into a format suitable for Bokeh or matplotlib patches. The formatted information is stored in a dictionary with three lists 
    corresponding to the "uuid", "patches_x", and "patches_y", and this dictionary is then converted into a pandas DataFrame.
    """
    patchesDict = {"patches_x":[], "patches_y":[]}
    if id_key != None:
        patchesDict[id_key] = []

    for obj in speckle_objs:
        obj_geo = obj[pth_to_geo]
        
        coorX = []
        coorY = []
        
        if isinstance(obj_geo, Mesh):
            # For meshes, we'll just use the vertices for now
            for pt in obj_geo.vertices:
                coorX.append(pt.x)
                coorY.append(pt.y)
        else:
            # For polylines, we'll use the existing logic
            obj_pts = Polyline.as_points(obj_geo)
            for pt in obj_pts:
                coorX.append(pt.x)
                coorY.append(pt.y)

        patchesDict["patches_x"].append(coorX)
        patchesDict["patches_y"].append(coorY)
        if id_key != None:
            patchesDict[id_key].append(obj[id_key])

    return pd.DataFrame(patchesDict)


#================= NOTION INTEGRATION ============================
headers = {
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json"
}

def get_page_id(token, database_id, name):
    headers['Authorization'] = "Bearer " + token
    # Send a POST request to the Notion API
    response = requests.post(f"https://api.notion.com/v1/databases/{database_id}/query", headers=headers)

    # Load the response data
    data = json.loads(response.text)

    # Check each page in the results
    for page in data['results']:
        # If the name matches, return the ID
        if page['properties']['name']['title'][0]['text']['content'] == name:
            return page['id']

    # If no match was found, return None
    return None

def add_or_update_page(token, database_id, name, type, time_updated, comment, speckle_link):
    # Format time_updated as a string 'YYYY-MM-DD'
    date_string = time_updated.strftime('%Y-%m-%d')

    # Construct the data payload
    data = {
        'parent': {'database_id': database_id},
        'properties': {
            'name': {'title': [{'text': {'content': name}}]},
            'type': {'rich_text': [{'text': {'content': type}}]},
            'time_updated': {'date': {'start': date_string}},
            'comment': {'rich_text': [{'text': {'content': comment}}]},
            'speckle_link': {'rich_text': [{'text': {'content': speckle_link}}]}
        }
    }

    # Check if a page with this name already exists
    page_id = get_page_id(token, database_id, name)

    headers['Authorization'] = "Bearer " + token
    if page_id:
        # If the page exists, send a PATCH request to update it
        response = requests.patch(f"https://api.notion.com/v1/pages/{page_id}", headers=headers, data=json.dumps(data))
    else:
        # If the page doesn't exist, send a POST request to create it
        response = requests.post("https://api.notion.com/v1/pages", headers=headers, data=json.dumps(data))
    
    print(response.text)

# Use the function
#add_or_update_page('your_token', 'your_database_id', 'New Title', 'New Type', datetime.now(), 'This is a comment', 'https://your-link.com')
