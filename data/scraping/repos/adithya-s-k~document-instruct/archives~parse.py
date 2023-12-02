import os
import csv
import warnings
from tqdm import tqdm
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
import tiktoken
import nltk

# Download nltk data (needed for word tokenization)
nltk.download('punkt')

def extract_headings_and_content(markdown_content):
    """
    Extract headings and corresponding content from a markdown content.
    """
    # Use regex to extract headings and content
    pattern = r'(#+)\s*(.*?)\n(.*?)(?=(#+\s*.*?\n)|\Z)'
    matches = re.findall(pattern, markdown_content, re.DOTALL)

    # Extract matched headings and content
    extracted_data = []
    for match in matches:
        heading_level = len(match[0])  # #, ##, ###, ####
        heading_text = match[1].strip()
        heading_content = match[2].strip()

        extracted_data.append((heading_level, heading_text, heading_content))
            
    return extracted_data

def print_folder_structure(root_folder, level=0):
    """
    Print the folder structure with appropriate indentation.
    """
    # Print folder structure
    if level == 0:
        print(os.path.basename(root_folder))
    else:
        print("  " * level + "|-- " + os.path.basename(root_folder) + "/")

    # Walk through root_folder and its subfolders
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.mdx'):
                filepath = os.path.join(root, file)
                print("  " * (level + 1) + "|-- " + file)
                
    for dir in dirs:
        print_folder_structure(os.path.join(root_folder, dir), level + 1)

def get_total_mdx_files(root_folder):
    """
    Count the total number of .mdx files in the directory and its subdirectories.
    """
    total_files = 0
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.mdx'):
                total_files += 1
    return total_files

def traverse_and_extract(root_folder):
    """
    Traverse the directory structure and extract headings from markdown files.
    """
    data = []
    total_files = get_total_mdx_files(root_folder)
    
    pbar = tqdm(total=total_files, desc="Extracting data", unit="file")

    # Walk through root_folder and its subfolders
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.mdx'):
                
                filepath = os.path.join(root, file)

                # Load markdown content using LangChain
                loader = UnstructuredMarkdownLoader(filepath)
                document = loader.load()
                markdown_content = document[0].page_content

                # Split markdown content using LangChain
                headers_to_split_on = [
                    ("#", "Heading 1"),
                    ("##", "Heading 2"),
                    ("###", "Heading 3"),
                    # Add more levels if needed
                ]
                markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
                md_header_splits = markdown_splitter.split_text(markdown_content)

                # Extract headings and content from each split
                for doc in md_header_splits:
                    heading_level = len(doc.metadata)  # Level of heading extracted

                    # Use the file name if heading text is empty
                    heading_text = doc.metadata.get(f"Heading {heading_level}")
                    if not heading_text:
                        heading_text = os.path.splitext(file)[0]

                    heading_content = doc.page_content.strip()

                    # Count number of tokens and words in the content
                    num_tokens = num_tokens_from_string(heading_content, "gpt2")
                    num_words = num_words_from_string(heading_content)

                    data.append((filepath, heading_level, heading_text, heading_content, num_tokens, num_words))
                pbar.update(1)    
            else:
                # Display a warning for files that are not .mdx files
                warnings.warn(f"File '{file}' is not an .mdx file. Only .mdx files will be parsed.")
                pbar.update(1)

    pbar.close()                
    return data

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string)) 
    return num_tokens

def num_words_from_string(string: str) -> int:
    """Returns the number of words in a text string."""
    words = nltk.word_tokenize(string)
    return len(words)

def write_to_csv(data, output_file):
    """
    Write extracted data to a CSV file.
    """
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Removed 'Heading Level' from the header
        writer.writerow(['File', 'Heading Text', 'Content', 'Num Tokens', 'Num Words'])
        # Only write the required columns to the CSV
        writer.writerows([(filepath, heading_text, heading_content, num_tokens, num_words) for filepath, _, heading_text, heading_content, num_tokens, num_words in data])


def calculate_generate_data(num_tokens):
    """
    Calculate the value of generate_data based on the number of tokens.
    """
    return (num_tokens + 299) // 300

def process_csv(input_file):
    # Read the entire CSV into a list in memory
    with open(input_file, 'r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        
        # Read the header and add the new 'Generate Data' column
        header = next(reader)
        header.append('Generate Data')
        
        rows = [header]
        
        # Process each row in the CSV
        for row in reader:
            num_tokens = int(row[3])  # Assuming 'Num Tokens' is the 4th column (0-indexed)
            generate_data = calculate_generate_data(num_tokens)

            # Add the new generate_data value to the row
            row.append(generate_data)
            rows.append(row)
            
    # Write the processed rows back to the same file
    with open(input_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(rows)


if __name__ == "__main__":
    root_folder = "documentation"
    output_csv = "output.csv"

    # Print folder structure
    print("Folder Structure:")
    print_folder_structure(root_folder)

    extracted_data = traverse_and_extract(root_folder)
    write_to_csv(extracted_data, output_csv)
    process_csv(output_csv)