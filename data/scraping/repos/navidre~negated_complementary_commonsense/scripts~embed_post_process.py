import os, ast, traceback
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial import distance
from dotenv import load_dotenv
import openai
load_dotenv(f'{Path().resolve()}/.env')
openai.api_key = os.environ['OPENAI_API_KEY']

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

HOME = str(Path.home())

# Negated
OUR_METHOD_RESULT_PATH = 'experiments/atomic2020_ten_preds/cot_qa_updated_neg_teach_var_temp/sampled_negated_preds_generated_cot_qa_updated_neg_teach_var_temp_evaluated.tsv'
target_file_path = 'experiments/atomic2020_ten_preds/cot_qa_updated_neg_teach_var_temp/sampled_negated_preds_generated_cot_qa_updated_neg_teach_var_temp_evaluated_with_ada_embeddings.csv'
figure_path = 'experiments/atomic2020_ten_preds/cot_qa_updated_neg_teach_var_temp/sampled_negated_preds_generated_cot_qa_updated_neg_teach_var_temp_evaluated_with_ada_embeddings.pdf'
# Normal
NORMAL_OUR_METHOD_RESULT_PATH = 'experiments/atomic2020_ten_preds/cot_qa_updated_neg_teach_var_temp/sampled_normal_preds_generated_cot_qa_updated_neg_teach_var_temp_evaluated.tsv'
normal_target_file_path = 'experiments/atomic2020_ten_preds/cot_qa_updated_neg_teach_var_temp/sampled_normal_preds_generated_cot_qa_updated_neg_teach_var_temp_evaluated_with_ada_embeddings.csv'
normal_figure_path = 'experiments/atomic2020_ten_preds/cot_qa_updated_neg_teach_var_temp/sampled_normal_preds_generated_cot_qa_updated_neg_teach_var_temp_evaluated_with_ada_embeddings.pdf'

# create a function to calculate cosine similarity
def cosine_similarity(x, y):
    return 1 - distance.cosine(x, y)

def get_embedding(text, model="text-embedding-ada-002"):
    # Check if text is a string
    if not isinstance(text, str):
        return None
    text = text.replace("\n", " ")
    return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

def get_embeddings_for_tsv(tsv_path, target_file_path):
    # If target file exists, load it and return
    if os.path.exists(target_file_path):
        print(f"Loading embeddings from {target_file_path}...")
        df = pd.read_csv(target_file_path)
        try:
            # Ignore nan values
            df = df[~df.combined_ada_embedding.isna()]
            if 'combined_ada_embedding' in df.columns:
                df['combined_ada_embedding'] = df.combined_ada_embedding.apply(eval).apply(np.array)
            # If answer_ada_embedding column exists, load it
            if 'answer_ada_embedding' in df.columns:
                df['answer_ada_embedding'] = df.answer_ada_embedding.apply(eval).apply(np.array)
            # If question_ada_embedding column exists, load it
            if 'question_ada_embedding' in df.columns:
                df['question_ada_embedding'] = df.question_ada_embedding.apply(eval).apply(np.array)
        except:
            print("Error loading embeddings from file. Please delete the file and try again.")
            traceback.print_exc()
            import IPython; IPython. embed(); exit(1)
    else:
        # Calculating embeddings
        print(f"Calculating embeddings for {tsv_path}...")
        df = pd.read_csv(tsv_path, sep='\t')
        df["combined"] = (
            "Question: " + df.prompt.str.strip() + "; Answer: " + df.generated_tail.str.strip()
        )
        # Get combined column value from row zero of the dataframe

        # Apply get_embedding to each question
        df['combined_ada_embedding'] = df.combined.apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))
        df['answer_ada_embedding'] = df.generated_tail.apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))
        df['question_ada_embedding'] = df.prompt.apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))
        # Save dataframe to file
        df.to_csv(target_file_path, index=False)
    return df

def plot_embeddings(df, column_name, figure_path):
    # Create a t-SNE model and transform the data
    # matrix = df.ada_embedding.apply(ast.literal_eval).to_list()
    try:
        matrix = df[column_name].apply(np.ndarray.tolist).to_list()
    except:
        print("Error converting embeddings to list.")
        traceback.print_exc()
        import IPython; IPython. embed(); exit(1)
    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)
    vis_dims = tsne.fit_transform(matrix)

    # five colors from green to red
    colors = ['#00FF00', '#00FF7F', '#FF0000', '#FF0000', '#FF0000']
    x = [x for x,y in vis_dims]
    y = [y for x,y in vis_dims]
    color_indices = df.majority_vote.values - 1

    colormap = matplotlib.colors.ListedColormap(colors)
    plt.scatter(x, y, c=color_indices, cmap=colormap, alpha=0.3)
    plt.title(f"{column_name} embeddings visualized using t-SNE")
    plt.savefig(figure_path)
    print(f"Saved figure to {figure_path}")

df = get_embeddings_for_tsv(OUR_METHOD_RESULT_PATH, target_file_path)
normal_df = get_embeddings_for_tsv(NORMAL_OUR_METHOD_RESULT_PATH, normal_target_file_path)

# Now using the embeddings, we can do some cool stuff like clustering, etc.
plot_embeddings(df, 'combined_ada_embedding', figure_path)
plot_embeddings(normal_df, 'combined_ada_embedding', normal_figure_path)

# Write code to plot similarity between question and answer embeddings
# Negated
# Calculate similarity between question and answer embeddings
# create a dataframe
df = pd.DataFrame({'question': [[1, 2, 3], [4, 5, 6], [7, 8, 9]], 'answer': [[4, 5, 6], [7, 8, 9], [1, 2, 3]], 'vote': [1, 2, 3]})

# calculate similarity between question and answer columns
df['similarity'] = df.apply(lambda row: cosine_similarity(row['question'], row['answer']), axis=1)

# create a scatter plot
plt.scatter(x=df['similarity'], y=df['vote'], c=df['vote'], cmap='rainbow', s=df.groupby(['vote', 'similarity']).size()*20)

# add a colorbar
plt.colorbar()

# add labels and title
plt.xlabel('Similarity')
plt.ylabel('Vote')
plt.title('Similarity vs Vote')

# show the plot
plt.show()

# TODO:
# 1. Plot question embeddings minus answer embeddings
# 2. Plot negated answer emdeddings minus normal answer embeddings
# 3. Plot negated question embeddings minus normal question embeddings
# 4. Plot negated answer embeddings minus average answer embeddings of the same question