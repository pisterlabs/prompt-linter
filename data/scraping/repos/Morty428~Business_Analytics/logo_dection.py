"""
Created for Neuro Insights

@author: Matthew Mortensen
@email: matthew.m.mortensen@gmail.com

parts of the code have been sanitized due to NDA
"""
from google.cloud import videointelligence_v1 as vi
from industry_classification import openai_categorizer
import pandas as pd

# function call to use logo detection from gooogle cloud video intelligence
#needs video intelligence API to be activated in google with .json key credentials
#input needed; string or string array of the video uri from GCP ex.'gs://bucket_name/bucket_folder/video_name.mp4'
#input needed; string or string array to create the output json file ex. 'gs://bucket_name/bucket_folder/video_name.json'
def detect_logos(video_uri: str, output: str) -> vi.VideoAnnotationResults: 
    video_client = vi.VideoIntelligenceServiceClient() #calling video intelligence API client
    features = [vi.Feature.LOGO_RECOGNITION] #feature list, if needed can add multiple feauters at once
    request = vi.AnnotateVideoRequest(  #request annotation from API needs input_uri, features, output_uri
        input_uri=video_uri,
        features=features,
        output_uri=output
    )

    print(f'Processing video: "{video_uri}"...')
    operation = video_client.annotate_video(request)
    #output is a json object that can then be looped through
    return operation.result().annotation_results[0]  # Single video


#function to get a dataframe output for logo detection and industy classification for a given list of videos
#note if given a large batch of videos it will take hours to process and can run into errors
#input needed; the list of video names to be processed
#input needed; the list of uri's for the videos from get_GCP_uri
#input needed; the list of output_uri for the video from get_GCP_uri
#input needed; file path for customized json named no_logo.json
def logo_output(names: str, in_uri: str, output_uri: str, file_path: str):
    #create empty dataframes
    
    #dataframe makeup sanitized
    df = pd.DataFrame(columns=[])
    df2 = pd.DataFrame(columns=[])
    path = file_path
    file_names = names
    uri = in_uri
    out_uri = output_uri    
    for file, out, name in zip(uri, out_uri, file_names): #loop through uri, out_uri, file_name lists
        video_uri = file
        output = out
        results = detect_logos(video_uri, output) ###detect logo function call
        if 'logo_recognition_annotations' in results: #if statement to check if a logo is detected
            #if a logo is detected loop through the annotation results and populate the logo dataframe
            annotations = results.logo_recognition_annotations #starting location of results
            #to calc total frames add the end_time_offset seconds and microseconds
            end = results.segment.end_time_offset.seconds + (results.segment.end_time_offset.microseconds /1e6)
            #calculating total number of frames
            tot = end * 25 # 25 is the fps of the ads given
            total_frame = round(tot,2)
            #lists to hold the values assigned to each column **sanitized**
            for annotation in annotations: 
                #sanitized
                for track in annotation.tracks: #loop to append specific instance of a logo being detected
                    industry = openai_categorizer(description) ###openai industry classification function call
                    #sanitized
                    #confidence is a value between 0 and 1 how confidnet the api is at correct logo detection+
                    confidence = track.confidence 
                    #sanitized
                    
            
            #create new df to hold each videos information then concat to main df
            #sanitized
            
        else: #condition to create a file of any video that did not have any logo detected
            print('*****Video has no logo detected; added to no_logo.json*****')
            #print('*****Video has no logo detected; added to no_logo.csv*****')
            ls9.append(name) #append to ls9 to create a column for the df
            #sanitized
            df2.to_json(path + 'no_logo.json', orient='index', indent=2)
            #df2.to_csv(path + 'no_logo.csv', index=False)

    return df