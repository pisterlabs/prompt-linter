#!/bin/bash

# Define the directory you want to check; default is the current directory.
# You can replace '.' with any other target directory.
directory='.'

# Loop through all the .py files found in the specified directory
find "$directory" -name '*.py' -print0 | while IFS= read -r -d $'\0' file; do
    # Extract the base name for the .py file
    base="${file%.py}"

    # Run cspell on the .py file and save the output to a corresponding .txt file
    cspell --unique --wordsOnly "$file" > "${base}.txt"

    echo "Spellcheck complete for $file. Output saved to ${base}.txt"
done