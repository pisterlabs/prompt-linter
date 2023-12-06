import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collections import Counter


# Load the data
with open('data/clean-reader_prompt_metadata_plus.json', 'r') as f:
    data = json.load(f)

repo_to_promptCount = {}
for file_addr in data:
    addr = file_addr.split('/')
    repo = addr[3]
    filename = addr[4]
    repo_to_promptCount[repo] = repo_to_promptCount.get(repo, 0) + len(data[file_addr])

repo_density = Counter(repo_to_promptCount.values())

# Plot the data
plt.bar(repo_density.keys(), repo_density.values())
plt.xlabel('Number of Prompts')
plt.ylabel('Number of Repositories')
plt.title('Distribution of Prompts per Repository')
plt.savefig('images/prompt_density.png')
plt.show()

# Plot the data on a log scale
plt.bar(repo_density.keys(), repo_density.values())
plt.xlabel('Number of Prompts')
plt.ylabel('Number of Repositories (log scale)')
plt.title('Distribution of Prompts per Repository (semi-log scale)')
plt.yscale('log')
plt.savefig('images/prompt_density_log.png')
plt.show()

# Plot the data on a log scale with a log scale y-axis
plt.bar(repo_density.keys(), repo_density.values())
plt.xlabel('Number of Prompts (log scale)')
plt.ylabel('Number of Repositories (log scale)')
plt.title('Distribution of Prompts per Repository (log scale)')
plt.yscale('log')
plt.xscale('log')
plt.savefig('images/prompt_density_log_log.png')
plt.show()

