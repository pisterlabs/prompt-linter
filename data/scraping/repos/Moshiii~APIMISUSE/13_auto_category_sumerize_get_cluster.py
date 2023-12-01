import os
import json
import openai
from dotenv import load_dotenv
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# read json
embeddings = []
with open('C:\@code\APIMISUSE\data\Solution_list.json',  encoding="utf-8") as f:
    Problem_list = json.load(f)
    embeddings = [x["embeddings"] for x in Problem_list]


# data = np.array(embeddings)
# k_values = range(3, 11)
# sse = []
# for k in k_values:
#     kmeans = KMeans(n_clusters=k)
#     kmeans.fit(data)
#     sse.append(kmeans.inertia_)


# # Plot the Elbow curve
# plt.plot(k_values, sse, 'bx-')
# plt.xlabel('Number of Clusters (K)')
# plt.ylabel('Sum of Squared Distances')
# plt.title('Elbow Method')
# plt.show()
# the best k is 6 where the elbow curve decreasing is slowdown.



# # get all the labels

# data = np.array(embeddings)
# k_values = 6
# kmeans = KMeans(n_clusters=k_values)
# kmeans.fit(data)

# labels = kmeans.labels_
# cluster_centers = kmeans.cluster_centers_
# # save labels to File
# with open('C:\@code\APIMISUSE\data\Solution_list_cluster_labels.json', 'w', encoding="utf-8") as f:
#     json.dump(labels.tolist(), f, indent=4)

# read labels from File
with open('C:\@code\APIMISUSE\data\Solution_list_cluster_labels.json', encoding="utf-8") as f:
    labels = json.load(f)
    labels = np.array(labels)

assignment = 4
cluster_items = []
for idx in range(len(labels)):
    if labels[idx] == assignment:
        cluster_items.append(Problem_list[idx])
        print(Problem_list[idx]["Solution"])


#Solution:
#assigmnet = 0
# replace or update deprecated or outdated functions
#assigmnet = 1
# no solution
#assigmnet = 2
# replace or update deprecated or outdated functions
#assigmnet = 3
# checking the device type, specifying the device, or moving tensors to the correct device (CPU or GPU) using methods such as .to(device) or conditional statements.
#assigmnet = 4




#problem:        
# assignment = 0
# null exception handeling: undefined Variable 

# assignment = 1
# Library and Module Replacement: Function and Method Replacement: Import and Name Changes:

# assignment = 3
# Model and State Handling:

# assignment = 4
# Casting and Conversion Errors: Data Type and Dtype Mismatches:

# assignment = 5
# Device Availability and Compatibility: Device Configuration and Initialization: Device Type and Allocation:

# assignment = 6
# Deprecated Function Replacement: Deprecated Module or Class Replacement: Deprecated Argument Usage:
