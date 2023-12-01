import os
import MemoryObject
import MemoryStreamAccess
import openai
import spacy
import re
from gpt_api_old import AI_entities as AI, AI_trainer
import tiktoken
import uuid
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import matplotlib.pyplot as plt
import numpy as np
from utils import config_retrieval

MAXIMUM_TOKENS_16K = 14000

memory_stream = MemoryStreamAccess.MemoryStreamAccess()

notes_dir_path = r'C:\Users\philippe\Documents\pdf to txt files\notes'

encoder = tiktoken.encoding_for_model("gpt-3.5-turbo-16k")

dir_path = r"C:\Users\philippe\Documents\pdf to txt files\test"

nlp = spacy.load("en_core_web_lg")

def split_into_paragraphs(text):
    # if the input is a list, join it into a string
    if isinstance(text, list):
        text = ' '.join(text)

    doc = nlp(text)
    paragraphs = []
    current_paragraph = []
    for sent in doc.sents:
        current_paragraph.append(sent.text)
        if len(current_paragraph) > 120:
            half = len(current_paragraph) // 2
            paragraphs.append(' '.join(current_paragraph[:half]))
            paragraphs.append(' '.join(current_paragraph[half:]))
            current_paragraph = []
        elif sent.text.endswith('.') or sent.text.endswith('\n'):
            paragraphs.append(' '.join(current_paragraph))
            current_paragraph = []
    if current_paragraph:
        paragraphs.append(' '.join(current_paragraph))
    return [p for p in paragraphs if len(p.split()) > 40]


def split_into_chapters(text):
    # Modified pattern to match only "chapter" or "part" followed by a number in any format
    pattern = r'(?i)(?:\bchapter\b|\bpart\b)\s*(\d+|\d+\.\d+|one|two|three|four|five|six|seven|eight|nine|ten|[IVXLCDM]+)'
    potential_chapters = re.split(pattern, text)

    chapters = []
    current_text = ""
    i = 0
    chapter_count = 0  # Counter for chapters


    while i < len(potential_chapters):
        current_text += potential_chapters[i]
        answer = "NA"
        i += 1
        reset_current_text = False  # Flag to determine if we should reset current_text

        if i < len(potential_chapters):
            # If the text is under 50 words, discard it without user approval
            if len(current_text.split()) < 50:
                reset_current_text = True
                i += 1
                continue
            # If the text has more than 500 words or every 20th chapter, add it to chapters without user approval
            elif len(current_text.split()) > 500 or chapter_count % 20 == 0:
                chapters.append(current_text)
                chapter_count += 1
                reset_current_text = True
            # If the text has less than 500 words, ask for user validation
            else:
                print(current_text)
                user_input = input(f"The chapter seems short. Do you think this is a valid chapter? (yes/no/add): ")
                if user_input.lower() == 'yes':
                    chapters.append(current_text)
                    chapter_count += 1
                    reset_current_text = True
                    answer = 'yes'
                elif user_input.lower() == "no":
                    reset_current_text = True
                    answer = 'no'
                elif user_input.lower() == "add":
                    answer = 'add'
                    AI_trainer.training_dataset_creation(current_text, answer, "Chapter_validation_AI")
                    # If "add" is chosen, we don't reset current_text, allowing it to accumulate with the next segment
                    continue
                AI_trainer.training_dataset_creation(current_text, answer, "Chapter_validation_AI")

            if reset_current_text:
                current_text = ""

            if i < len(potential_chapters):  # Check again to avoid index out of range
                current_text += potential_chapters[i]
            i += 1

    # Handle the last chapter
    if current_text and len(current_text.split()) >= 50:  # Ensure the text is not too short
        print(current_text)
        user_input = input(f"The chapter seems short. Do you think this is a valid chapter? (yes/no): ")
        if user_input.lower() == 'yes' or len(current_text.split()) >= 1000:
            chapters.append(current_text)

    return chapters

def save_notes_to_file(notes, base_filename):
    notes_filename = os.path.join(notes_dir_path, f'notes_for_{base_filename}.txt')
    with open(notes_filename, 'w', errors='ignore') as file:
        file.write('\n'.join(notes))
    print(f"Notes saved to {notes_filename}")


def apply_notetaking_AI_and_save(text, filename):
    text = text.lower()
    chapters = split_into_chapters(text)
    del chapters[0]  # the first part is before the first keyword and is useless

    processed_chapters = []
    chapter_counter = 1  # Counter for processed chapters
    for i, chapter in enumerate(chapters):
        tokens = encoder.encode(chapter)
        if len(tokens) <= MAXIMUM_TOKENS_16K:
            processed_chapters.append("Chapter number " + str(chapter_counter) + ": " + chapter)
            chapter_counter += 1
        else:
            midpoint = len(tokens) // 2
            segment1 = encoder.decode(tokens[:midpoint])
            segment2 = encoder.decode(tokens[midpoint:])

            if segment1:  # Ensure the segment is not empty
                processed_chapters.append("Chapter number " + str(chapter_counter) + ": " + segment1)
                chapter_counter += 1  # Increment the counter

            if segment2:  # Ensure the segment is not empty
                processed_chapters.append("Chapter number " + str(chapter_counter) + ": " + segment2)
                chapter_counter += 1  # Increment the counter

    notes = []
    for i, chapter in enumerate(processed_chapters):
        if len(encoder.encode(chapter)) <= MAXIMUM_TOKENS_16K:  # Only process chapters within the token limit
            note = AI.notetaking_AI(chapter)  # here, chapter is a string

            # Check if the returned note is a list
            if isinstance(note, list):
                note = " ".join(note)  # Convert the list to a single string

            # Prepend the note with the marker
            note = "generated_chapter " + str(i + 1) + ": " + note
            notes.append(note)
        else:
            print(f"Chapter {i + 1} is too large for notetaking.")

        if i == 20:
            print("file check")

    base_filename = os.path.splitext(filename)[0]
    save_notes_to_file(notes, base_filename)

    return processed_chapters


def get_notes_for_chapter(filename, chapter_number):
    notes_filename = os.path.join(notes_dir_path, f'notes_for_{os.path.splitext(filename)[0]}.txt')

    # Check if the file exists
    if not os.path.exists(notes_filename):
        raise FileNotFoundError(f"Error: File {notes_filename} does not exist.")

    with open(notes_filename, 'r', encoding='utf-8', errors='ignore') as file:
        notes = file.read()  # read notes as a single string

    chapter_start_marker = "generated_chapter " + str(chapter_number) + ":"
    chapter_end_marker = "generated_chapter " + str(chapter_number + 1) + ":"

    # Determine the starting and ending indices of the chapter in the notes
    chapter_start = notes.lower().find(chapter_start_marker.lower())
    chapter_end = notes.lower().find(chapter_end_marker.lower())

    # If the chapter_end_marker isn't found, it means we're looking at the last chapter in the file
    if chapter_end == -1:
        chapter_notes = notes[chapter_start:].strip()
    else:
        chapter_notes = notes[chapter_start:chapter_end].strip()

    return chapter_notes



def assign_parent_ID(text, notes, level_of_abstraction,memory_id):
    # Get the subject names from the subject_choice_and_creation_AI function
    subjects = AI.subject_choice_and_creation_AI(text, notes, level_of_abstraction,memory_id)
    mysql_config = {"host": "localhost", "user": "root", "password": "Q144bughL0?Y@JFYxPA0", "database": "externalmemorydb"}
    pinecone_index = "spiky-testing"
    # Initialize the memory stream
    memory_stream = MemoryStreamAccess.MemoryStreamAccess(mysql_config, pinecone_index)
    cursor = memory_stream.mycursor

    # Get the IDs of the subjects from the database
    subject_ids = []
    for subject in subjects:
        cursor.execute("SELECT memory_id FROM spiky_memory_table WHERE content = %s", (subject,))
        result = cursor.fetchone()
        if result:
            subject_ids.append(result[0])

    # Convert the list of IDs to a comma-separated string
    subject_ids_str = ",".join(map(str, subject_ids))

    # Close the database connection
    memory_stream.mydb.close()

    return subject_ids_str

def check_for_unlinked_child_memories():
    #TODO create a process that checks all the level 0 memories to see if they have a parent, if not, find one or delete it potentially.
    #TODO if some already have parents but have less than 5 memories, merge them with other subtrees?
    pass

def fetch_vectors_in_chunks(memory_ids, chunk_size=1000):
    vectors_list = []
    for i in range(0, len(memory_ids), chunk_size):
        chunk = memory_ids[i:i+chunk_size]
        vectors_chunk = memory_stream.index.fetch(ids=chunk)
        for vec_id, vec_data in vectors_chunk['vectors'].items():
            vectors_list.append(vec_data['values'])
    return vectors_list


def hierarchical_clustering(memory_ids, max_clusters=None, distance_threshold=None):
    # Retrieve vectors from Pinecone
    vectors = fetch_vectors_in_chunks(memory_ids)

    # Convert the list of vectors to a numpy ndarray
    vectors_array = np.array(vectors)

    # Compute the linkage matrix
    Z = linkage(vectors_array, method='ward')

    # Decide on the number of clusters based on the provided settings
    if max_clusters:
        clusters = fcluster(Z, max_clusters, criterion='maxclust')
    elif distance_threshold:
        clusters = fcluster(Z, distance_threshold, criterion='distance')
    else:
        raise ValueError("Either max_clusters or distance_threshold must be provided.")

    # Group memory IDs by their cluster
    clustered_memories = {}
    for idx, cluster_id in enumerate(clusters):
        if cluster_id not in clustered_memories:
            clustered_memories[cluster_id] = []
        clustered_memories[cluster_id].append(memory_ids[idx])

    return clustered_memories

def visualize_dendrogram(memory_ids, distance_threshold=None, max_cluster_size=None, min_cluster_size=None):
    """
    Visualize the dendrogram and optionally highlight small clusters.

    Parameters:
    - memory_ids: List of memory IDs corresponding to the data points
    - distance_threshold: Distance threshold to form clusters (default will be set interactively)
    - max_cluster_size: Maximum size for a cluster to be considered "small"
    """

    vectors = fetch_vectors_in_chunks(memory_ids)
    vectors_array = np.array(vectors)
    Z = linkage(vectors_array, method='ward')
    print(len(memory_ids))

    # If no threshold is provided, set it interactively
    if not distance_threshold:
        print("Please observe the dendrogram and set an appropriate distance threshold.")
        plt.figure(figsize=(10, 7))
        dendrogram(Z)
        plt.title("Dendrogram")
        plt.ylabel("Distance")
        plt.xlabel("Memory IDs or Clusters")
        plt.show()
        distance_threshold = float(input("Enter the chosen distance threshold: "))

    # Plot the dendrogram with the chosen threshold
    plt.figure(figsize=(10, 7))
    dendrogram(Z)
    plt.axhline(y=distance_threshold, color='r', linestyle='--')

    plt.show()

    # Get clusters formed at the chosen threshold
    clusters = fcluster(Z, distance_threshold, criterion='distance')

    # Print out data points of clusters and highlight small clusters
    cluster_dict = {}
    for cluster_id, memory_id in zip(clusters, memory_ids):
        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = []
        cluster_dict[cluster_id].append(memory_id)
    if min_cluster_size is None:
        min_cluster_size = 3
    if max_cluster_size is None:
        max_cluster_size = 12
    count=0
    max_count=0
    min_count=0
    for cluster_id, members in cluster_dict.items():
        count+=1
        print(f"Cluster {cluster_id}:")
        for member in members:
            print(f"  - {member}")
        if len(members) <= min_cluster_size:
            print(f"  (Small Cluster)")
            min_count+=1
        if len(members) >= max_cluster_size:
            print(f"  (Big CLuster")
            memories=memory_stream.select_memories("spiky_memory",type_select="fetch rows",corresponding_IDs=members)
            for memory in memories:
                print("new_memory")
                print(memory.content)
                print()
            max_count+=1
        print()
    print("count:" + str(count))
    print("min count:" + str(min_count))
    print("max count:" + str(max_count))
def create_spiky_memory_object(text="no text", memory_type="level0", notes="", memory_ids=None, clustering_setting=None,level_of_abstraction=None):
    memory_object=None
    if memory_type == "level0":
        content = AI.observation_memory_AI(text, notes)
        child_list_ID = [str(uuid.uuid4())]
        memory_object = MemoryObject.MemoryObject(content=content, child_list_ID=child_list_ID)
        text_memory_object = MemoryObject.MemoryObject(memory_id= child_list_ID[0],content=text, parent_ID=memory_object.memory_id)

        memory_stream.add_memory(text_memory_object,AI.get_embedding(text))
        memory_stream.add_memory(memory_object,AI.get_embedding(content))

    elif memory_type == "reflection": # in this process the code gets all the objects that have the correct level of abstraction and no parents, and then regroups them ot create parents that are a level of abstraction above.
        if not level_of_abstraction:
            raise ValueError("For reflection type, level_of_abstraction must be provided.")

        unlinked_memories = memory_stream.select_memories(table_name="spiky_memory", type_select="unlinked_at_level",
                                                          level_of_abstraction=level_of_abstraction)#level of abstraction is the lower level in order to create a higher level
        memory_ids = [memory.memory_id for memory in unlinked_memories]
        clustering_setting = {"max_clusters":20,"distance_threshold":5}
        clustered_memories = hierarchical_clustering(memory_ids, **clustering_setting)

        for cluster_id, clustered_ids in clustered_memories.items():
            # Fetch the contents of the memories associated with the clustered_ids
            clustered_contents = [memory.content for memory in memory_stream.select_memories(table_name="spiky_memory",type_select="fetch rows", corresponding_IDs=clustered_ids)]

            # Generate a name for the reflection
            reflection_name = AI.generate_reflection_name(clustered_contents)  # Modify the function to accept contents

            # Create a memory object for the reflection name
            child_id=[str(uuid.uuid4())]
            reflection_name_memory = MemoryObject.MemoryObject(content=reflection_name,child_list_ID=child_id)
            memory_stream.add_memory(reflection_name_memory, AI.get_embedding(reflection_name))

            # Generate the summary for the reflection
            content = AI.reflection_AI_new(clustered_contents,reflection_name)  # Modify the function to accept contents

            # Create a memory object for the reflection summary with the reflection name as its parent
            reflection_summary_memory = MemoryObject.MemoryObject(memory_id=child_id, content=content, child_list_ID=clustered_ids, parent_ID=reflection_name_memory.memory_id)
            memory_stream.add_memory(reflection_summary_memory, AI.get_embedding(content))

            # Update the parent ID for each memory in this cluster to the reflection summary
            for memory_id in clustered_ids:
                memory = memory_stream.select_memories(table_name="spiky_memory",type_select="fetch rows", corresponding_IDs=[memory_id])[0]
                memory.parent_id = reflection_summary_memory.memory_id
                memory_stream.modify_memory(memory_id, memory)

    if memory_object is None:
        print(f"No spiky memory was created: text [{text}] ")


def start_process():
    for filename in os.listdir(dir_path):
        print(filename)
        if filename.endswith(".txt"):
            with open(os.path.join(dir_path, filename), 'r', encoding='utf-8', errors='ignore') as file:
                text = file.read()
                chapters = apply_notetaking_AI_and_save(text, filename)
                chapter_number = 1
                for chapter in chapters:
                    paragraphs = split_into_paragraphs(chapter)
                    notes = get_notes_for_chapter(filename,chapter_number)
                    for paragraph in paragraphs:
                        create_spiky_memory_object(paragraph, memory_type="level0", notes=notes)
                    # create_spiky_memory_object(memory_type="reflection",level_of_abstraction=0)
                    # check_for_unlinked_child_memories()
                    chapter_number = chapter_number + 1

import mysql.connector

# Set up a connection to the MySQL database
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Q144bughL0?Y@JFYxPA0",
    database="externalmemorydb"
)
cursor = conn.cursor()


for filename in os.listdir(dir_path):
    print(filename)
    if filename.endswith(".txt"):
        with open(os.path.join(dir_path, filename), 'r', encoding='utf-8', errors='ignore') as file:
            text = file.read()
            chapters = split_into_chapters(text)
            for chapter in chapters:
                count = 0
                paragraphs = split_into_paragraphs(chapter)
                memory_ids = []  # List to collect all memory IDs
                for paragraph in paragraphs:
                    count += 1
                    # Query the database to find the ID for the matching text
                    query = "SELECT memory_id FROM spiky_memory WHERE content = %s"
                    cursor.execute(query, (paragraph,))
                    result = cursor.fetchone()
                    if result:
                        memory_id = result[0]
                        memory_ids.append(memory_id)  # Add the memory ID to the list
                        print(memory_id)
                        print(count)
                    else:
                        print("failed match: " + paragraph)
                if memory_ids:
                    visualize_dendrogram(memory_ids, distance_threshold=0.72)

cursor.close()
conn.close()

