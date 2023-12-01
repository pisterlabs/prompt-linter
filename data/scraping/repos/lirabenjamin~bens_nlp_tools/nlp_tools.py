import pandas as pd
import spacy
from spacy import displacy
import en_core_web_trf
from tqdm import tqdm
import swifter  # for faster pandas apply
import spacy_transformers

# 1. Add LIWC
def process_dataframe_with_liwc(df: pd.DataFrame, id_column:str, text_column: str, save_to_csv: bool = False, output_filename: str = 'liwc_output.csv') -> pd.DataFrame:
    import pandas as pd
    import subprocess
    import os
    # First, save the DataFrame to a temporary CSV file
    temp_filename = 'temp_for_liwc.csv'
    df.to_csv(temp_filename, index=False)

    # Command to run LIWC on the CSV file
    cmd_to_execute = ["LIWC-22-cli",
                      "--mode", "wc",
                      "--input", temp_filename,
                      "--row-id-indices", str(df.columns.get_loc(id_column)+1),  # Assuming the first column (index 0) is the identifier
                      "--column-indices", str(df.columns.get_loc(text_column)+1),  # Index of the text column
                      "--output", output_filename if save_to_csv else 'liwc_temp_output.csv']

    # Execute the command
    result = subprocess.call(cmd_to_execute)

    # Check if the command was successful
    if result != 0:
        os.remove(temp_filename)
        raise RuntimeError("Error occurred while running LIWC-22. Ensure the LIWC-22 application is running.")

    # Read the LIWC output into a pandas DataFrame
    if os.path.exists(output_filename if save_to_csv else 'liwc_temp_output.csv'):
        liwc_output = pd.read_csv(output_filename if save_to_csv else 'liwc_temp_output.csv')
    else:
        os.remove(temp_filename)
        raise FileNotFoundError(f"Expected output file {output_filename if save_to_csv else 'liwc_temp_output.csv'} not found.")
    
    # Clean up temporary files
    os.remove(temp_filename)
    if not save_to_csv:
        os.remove('liwc_temp_output.csv')

    return liwc_output

# 2. custom dictionary: score with sum/ presence absence, tfidf or not. preprocess or not.
def count_dictionary_matches(data, text_column, output_column = "results", mode="count", dictionary=None):
    import pandas as pd
    import re
    if dictionary is None:
        dictionary = []

    # Helper function to count matches in a text
    def count_matches(text):
        return sum(len(re.findall(pattern, text)) for pattern in dictionary)

    # Calculate matches for each row
    data['matches'] = data[text_column].apply(count_matches)

    # If mode is 'proportion', normalize the counts by word counts
    if mode == "proportion":
        data['word_counts'] = data[text_column].apply(lambda x: len(x.split()))
        data[output_column] = data['matches'] / data['word_counts']
    else:
        data[output_column] = data['matches']

    # Drop intermediate columns
    data.drop(columns=['matches'], inplace=True, errors='ignore')
    data.drop(columns=['word_counts'], inplace=True, errors='ignore')

    return data

# 3. Sanitize text
# Initialize spacy model
nlp = en_core_web_trf.load()

def replace_ner(text, level=1):
    """
    Replace named entities in a text.
    
    Parameters:
    - text (str): Input text
    - level (int): Level of deidentification. 
                   1 = Replace only persons with first and last names.
                   2 = Replace all persons.
                   (More levels can be added as required)
                   
    Returns:
    - str: Deidentified text
    """
    doc = nlp(text)
    clean_text = text
    for ent in reversed(doc.ents):
        if ent.label_ == "PERSON":
            if level == 1 and " " in ent.text:
                clean_text = clean_text[:ent.start_char] + "[PERSON]" + clean_text[ent.end_char:]
            elif level == 2:
                clean_text = clean_text[:ent.start_char] + "[PERSON]" + clean_text[ent.end_char:]
    return clean_text

def deidentify_dataframe(df, input_text_column, output_text_column, id_column=None, level=1, save=False, output_filename="deidentified_data.csv"):
    """
    Deidentify a pandas dataframe.
    
    Parameters:
    - df (pd.DataFrame): Input dataframe
    - text_column (str): Name of the text column to deidentify
    - id_column (str, optional): Name of the ID column
    - level (int): Level of deidentification. Refer to `replace_ner` for levels.
    - save (bool): Whether to save the resulting dataframe
    - output_filename (str): Name of the output file if `save` is True
    
    Returns:
    - pd.DataFrame: Deidentified dataframe
    """
    tqdm.pandas()  # Initialize tqdm for pandas
    df[output_text_column] = df[input_text_column].swifter.progress_bar(enable=True).apply(lambda x: replace_ner(x, level))
    
    if save:
        df.to_csv(output_filename)
        
    return df

# Add topics

# Add GPT rating
def generate_ratings(data: pd.DataFrame, id_col: str, text_col: str, prompt: str, output_dir: str, verbose: bool = False, temperature = 1, keep_details = True) -> pd.DataFrame:
    import openai
    import concurrent.futures
    import os
    import datetime
    import ast
    
    # Check OpenAI API key
    if not openai.api_key:
        raise ValueError("OpenAI API key not set. Please set it before calling this function.")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def rate_conversation(id, conversation):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            temperature=temperature,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Here are the participant comments:\n{conversation}"},
            ]
        )
        result = response.choices[0].message.content
        if verbose:
            print(result)
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        with open(f"{output_dir}/{id}_{now}_temp1.txt", "w") as f:
            f.write(result)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(rate_conversation, data[id_col], data[text_col])
    
    def read_all_files_to_dataframe(directory):
        all_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.txt')]
        df_list = []

        for filename in all_files:
            with open(filename, 'r') as f:
                content = f.read()
                df_list.append({"filename": filename, "content": content})

        return pd.DataFrame(df_list)
    
    
    combined_df = read_all_files_to_dataframe(output_dir)
    if combined_df.empty:
        raise ValueError("No data read from the output directory. Ensure .txt files exist in the directory.")
    combined_df.columns = [id_col, "content"]
    combined_df[id_col] = combined_df[id_col].str.replace(f"{output_dir}/", "")
    combined_df[id_col] = combined_df[id_col].str.replace(".txt", "")
    combined_df[[id_col, "timestamp", "temperature"]] = combined_df[id_col].str.split("_", expand=True)

    # Unroll the dictionary
    combined_df["content"] = combined_df["content"].apply(ast.literal_eval)
    df = pd.DataFrame(combined_df.content.tolist())

    # Combine df and combined_df
    df = pd.concat([combined_df, df], axis=1)
    if not keep_details:
        df = df.drop(["content", "timestamp", "temp"], axis=1)
    
    # Join df with data on id
    df[id_col] = df[id_col].astype(int)
    df = df.merge(data, on=id_col)

    return df

# utils word clouds
def create_word_cloud(data, word_col, size_col, color_col, font_filename=None, title_font_size=30, log_scale=False, out=["png", "pdf"], output_file="wordcloud", plot_title="Word Cloud"):
  import numpy as np
  import matplotlib.pyplot as plt
  from wordcloud import WordCloud
  from PIL import Image
  from matplotlib.colors import LinearSegmentedColormap
  import matplotlib.font_manager as fm
  # Normalize the size and color columns
  data['size_norm'] = data[size_col] / data[size_col].max()
  if log_scale:
    data[color_col] = np.log(data[color_col])
  data['color_norm'] = (data[color_col] - data[color_col].min()) / (data[color_col].max() - data[color_col].min())

    # Create a custom color map from gray to blue
  custom_colors = LinearSegmentedColormap.from_list("custom_colors", ["gray", "darkblue"], N=256)

  # Define a custom color function
  def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        freq = data.loc[data[word_col] == word, 'color_norm'].values[0]
        r, g, b, _ = custom_colors(freq)
        return int(r * 255), int(g * 255), int(b * 255)

    # Create a dictionary with words and their corresponding sizes
  word_sizes = data.set_index(word_col)['size_norm'].to_dict()

    # Mask for the word cloud 
  x, y = np.ogrid[:288*3, :432*3]
  mask = ((x - 144*3) ** 2 / (144*3) ** 2) + ((y - 216*3) ** 2 / (216*3) ** 2) > 1
  mask = 255 * mask.astype(int)

    # Load font if specified
  if font_filename:
        font_path = font_filename
        font_prop = fm.FontProperties(fname=font_path)
  else:
        font_path = None
        font_prop = None

    # Generate the word cloud
  wc = WordCloud(
        background_color='white',
        color_func=color_func,
        prefer_horizontal=1,
        width=800,
        mask=mask,
        font_path=font_path,
        height=600
    ).generate_from_frequencies(word_sizes)

    # Display the word cloud
  plt.figure(figsize=(8, 6), dpi=800)
  plt.imshow(wc, interpolation='bilinear')
  plt.axis("off")
  plt.tight_layout(pad=0)
  plt.title(plot_title, fontweight="bold", fontsize=title_font_size, fontproperties=font_prop)
  if "pdf" in out:
    plt.savefig(f"{output_file}.png", format="png", dpi=600, bbox_inches='tight', pad_inches=0)
  if "png" in out:
    plt.savefig(f"{output_file}.pdf", format="pdf", dpi=600, bbox_inches='tight', pad_inches=0)
  plt.close()