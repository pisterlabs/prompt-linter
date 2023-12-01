# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

intermediate = "midpattern=true"
folder = f'/Volumes/MATLAB-Drive/Shared/figures/{intermediate}/tables/'
plotfolder=f'/Volumes/MATLAB-Drive/Shared/figures/{intermediate}/cca_regress_python/'
if not os.path.exists(plotfolder):
    os.makedirs(plotfolder)

# Read the CSV file into a dataframe
df = pd.read_csv(os.path.join(folder, 'combined_faxis=Inf_regress.csv'))

# Filter for rows where pvalue_U and pvalue_V are both significant (less than 0.05)
significant_df = df[(df['pvalue_U'] < 0.05) & (df['pvalue_V'] < 0.05)]
significant_df['animal'] = significant_df['filename'].apply(lambda x: os.path.basename(x).split('_')[0])

# remove 60 hz from coherence -- because coherence sensitive to 60 hz noise
notch = significant_df.f.unique()[significant_df.query('field=="Cavg"').set_index('f').coef_U.abs().groupby('f').mean().argmax()]
significant_df = significant_df.query('f < (@notch-5) | f > (@notch+5)')

# Take the absolute value of coef_U and coef_V
significant_df['abs_coef_U']          = np.abs(significant_df['coef_U'])
significant_df['abs_coef_V']          = np.abs(significant_df['coef_V'])
significant_df['abs_coef_difference'] = significant_df['abs_coef_U'] - significant_df['abs_coef_V']
significant_df['coef_difference']     = significant_df['coef_U'] - significant_df['coef_V']
significant_df['abs_coef_U_strat']    = significant_df['abs_coef_U'] + 0.20
significant_df['abs_coef_V_strat']    = significant_df['abs_coef_V'] - 0.20
# Compute the mean of the absolute values of coef_U and coef_V
significant_df['coef_mean'] = significant_df[['abs_coef_U', 'abs_coef_V']].mean(axis=1)
significant_df['smooth_abs_coef_difference'] = significant_df.groupby(['field', 'animal','coef_i']).rolling(5, center=True).mean().reset_index()['abs_coef_difference']

# ------------------------------

# Create a seaborn plot, splitting by 'field' in the columns
# Set sharey=False to not share the y-axis across subplots
g = sns.FacetGrid(significant_df, col="field", col_wrap=5, height=4, aspect=1, sharey=True)
g.map(sns.lineplot, 'f', 'coef_mean', marker='o', color='black', alpha=0.5)
g.map(sns.lineplot, 'f', 'abs_coef_U_strat', marker='o', color='red', alpha=0.5, errorbar=None)
g.map(sns.lineplot, 'f', 'abs_coef_V_strat', marker='o', color='blue', alpha=0.5, errorbar=None)
# Add titles to the subplots
g.set_titles("{col_name}")
# Show the plot
plt.show()
plt.savefig(os.path.join(plotfolder, 'coef_mean_overall.png'))
plt.savefig(os.path.join(plotfolder, 'coef_mean_overall.pdf'))

# ------------------------------

# # Create a seaborn plot, splitting by 'field' in the columns
# # Set sharey=False to not share the y-axis across subplots
# g = sns.FacetGrid(significant_df, col="field", col_wrap=5, height=4, aspect=1, sharey=True)
# g.map(sns.lineplot, 'f', 'smooth_abs_coef_difference', marker='o', color='black', alpha=0.5)
# # Add titles to the subplots
# g.set_titles("{col_name}")
# # Show the plot
# plt.show()
# plt.savefig(os.path.join(plotfolder, 'coef_mean_overall.png'))
# plt.savefig(os.path.join(plotfolder, 'coef_mean_overall.pdf'))

# ------------------------------
# OVERALL_10+_word_LONG_TITLE: "Frequency vs Mean Coefficient, split by 'field' in the columns"
# COLUMNS: 'field'
# FILTER: 'meas' == 'raw'
# Y-AXIS: 'coef_mean'
# X-AXIS: 'f'

# Extract 'animal' from 'filename' and add it as a new column to the DataFrame
significant_df['animal'] = \
        significant_df['filename'].apply(lambda x: os.path.basename(x).split('_')[0])

# Create a seaborn plot, splitting by 'field' in the columns and 'meas' in the rows
g = sns.FacetGrid(significant_df, row="meas", col="field", height=4, aspect=1, sharey=False)
g.map(sns.lineplot, 'f', 'coef_mean', marker='o')

# Add titles to the subplots
g.set_titles("{col_name}")

# Show the plot
plt.show()
plt.savefig(os.path.join(plotfolder, 'meas_coef_mean.png'))

# Create a seaborn plot, splitting by 'field' in the columns and 'animal' in the rows
g = sns.FacetGrid(significant_df, row="animal", col="field", height=4, aspect=1, sharey=False)
g.map(sns.lineplot, 'f', 'coef_mean', marker='o')

# Add titles to the subplots
g.set_titles("{col_name}")

# Show the plot
plt.show()
plt.savefig(os.path.join(plotfolder, 'animal_coef_mean.png'))
plt.savefig(os.path.join(plotfolder, 'animal_coef_mean.pdf'))

# ------------------------------
# OVERALL_10+_word_LONG_TITLE: "Frequency vs Mean Coefficient, split by 'field' in the columns and 'animal' in the rows"
# ROWS: 'animal'
# COLUMNS: 'field'
# FILTER: 'meas' == 'raw'
# Y-AXIS: 'coef_mean'
# X-AXIS: 'f'
#

# Filter the DataFrame for rows where 'meas' is 'raw'
raw_df = significant_df[significant_df['meas'] == 'raw']

# Create a seaborn plot, splitting by 'field' in the columns and 'animal' in the rows
g = sns.FacetGrid(raw_df, row="animal", col="field", height=4, aspect=1, sharey=False)
g.map(sns.lineplot, 'f', 'coef_mean', marker='o')

# Add titles to the subplots
g.set_titles("{col_name}")

# Show the plot
plt.show()
plt.savefig(os.path.join(plotfolder, 'raw_animal_coef_mean.png'))
plt.savefig(os.path.join(plotfolder, 'raw_animal_coef_mean.pdf'))

# ------------------------------

# Create a seaborn plot with 'field' in the columns, 'f' on the x-axis, and 'coef_difference' on the y-axis,
# splitting by 'animal' in the rows
cm = sns.color_palette("PuBuGn_d", 5)
g = sns.FacetGrid(significant_df, col="field", row="coef_i", hue="coef_i", height=4, aspect=1, sharey=False)
g.map(sns.lineplot, 'f', 'coef_mean', marker='o')

# Add titles to the subplots
g.set_titles("{col_name} by coef_i")

# Show the plot
plt.show()
plt.savefig(os.path.join(plotfolder, 'coef_mean_by_component.png'))
plt.savefig(os.path.join(plotfolder, 'coef_mean_by_component.pdf'))

# ------------------------------

# Create a seaborn plot with 'field' in the columns, 'f' on the x-axis, and 'coef_difference' on the y-axis,
# splitting by 'animal' in the rows
cm = sns.color_palette("PuBuGn_d", 5)
g = sns.FacetGrid(significant_df, col="field", row="coef_i", hue="animal", height=4, aspect=1, sharey=False)
g.map(sns.lineplot, 'f', 'coef_mean', marker='o')

# Add titles to the subplots
g.set_titles("{col_name} by coef_i")
for ax in g.axes.ravel():
    ax.set_ylim([0, 0.6])
    # ax.axvline(x=60, color='black', linestyle='--')

# Show the plot
plt.show()
plt.savefig(os.path.join(plotfolder, 'animal_coef_mean_by_component.png'))
plt.savefig(os.path.join(plotfolder, 'animal_coef_mean_by_component.pdf'))

# ------------------------------

# Create a new column for the difference between coef_U and coef_V
significant_df['coef_difference'] = significant_df['coef_U'] - significant_df['coef_V']
# Create a seaborn plot with 'field' in the columns, 'f' on the x-axis, and 'coef_difference' on the y-axis
g = sns.FacetGrid(significant_df, col="field", height=4, aspect=1)
g.map(sns.lineplot, 'f', 'coef_difference', marker='o')
# Add titles to the subplots
g.set_titles("{col_name}")
# Show the plot
plt.show()
plt.savefig(os.path.join(plotfolder, 'cI.png'))

# ------------------------------

# Create a seaborn plot with 'field' in the columns, 'f' on the x-axis, and 'coef_difference' on the y-axis,
# splitting by 'animal' in the rows
g = sns.FacetGrid(significant_df, row="animal", col="field", height=4, aspect=1, sharey=False)
g.map(sns.lineplot, 'f', 'coef_difference', marker='o')

# Add titles to the subplots
g.set_titles("{col_name}")

# Show the plot
plt.show()
plt.savefig(os.path.join(plotfolder, 'animal_coef_difference.png'))

# ------------------------------

# Create a seaborn plot with 'field' in the columns, 'f_bin' on the x-axis, and 'coef_difference' on the y-axis
# Add a horizontal black dashed line at y=0
# Group by 'f_bin', 'field', and 'animal', and compute the mean of 'coef_difference' within each group
# Round the 'f' values to the nearest 5 to create frequency bins
significant_df['f_bin'] = 5 * round(significant_df['f'] / 5)
grouped_df = significant_df.groupby(['f_bin', 'field', 'animal'])['coef_difference'].mean().reset_index()
g = sns.FacetGrid(grouped_df, col="field", height=4, aspect=1, sharey=False)
g.map(sns.lineplot, 'f_bin', 'coef_difference', marker='o')
g.map(plt.axhline, y=0, ls='--', c='black')

# Add titles to the subplots
g.set_titles("{col_name}")

# Show the plot
plt.show()
plt.savefig(os.path.join(plotfolder, 'binned_coef_difference.png'))
plt.savefig(os.path.join(plotfolder, 'binned_coef_difference.pdf'))


# Create a seaborn plot with 'field' in the columns, 'f_bin' on the x-axis, and 'coef_difference' on the y-axis
# Add a horizontal black dashed line at y=0
# Group by 'f_bin', 'field', and 'animal', and compute the mean of 'coef_difference' within each group
# Round the 'f' values to the nearest 5 to create frequency bins
significant_df['f_bin'] = 10 * round(significant_df['f'] / 10)
grouped_df = significant_df.groupby(['f_bin', 'field', 'animal'])['coef_difference'].mean().reset_index()
g = sns.FacetGrid(grouped_df, col="field", height=4, aspect=1, sharey=False)
g.map(sns.lineplot, 'f_bin', 'coef_difference', marker='o')
g.map(plt.axhline, y=0, ls='--', c='black')

# Add titles to the subplots
g.set_titles("{col_name}")

# Show the plot
plt.show()
plt.savefig(os.path.join(plotfolder, 'large_binned_coef_difference.png'))
plt.savefig(os.path.join(plotfolder, 'large_binned_coef_difference.pdf'))

