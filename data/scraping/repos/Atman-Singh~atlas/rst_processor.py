import rst2pdf
import os
# from langchain.text_splitter import CharacterTextSplitter

def main():
    # filenames = ['Source 1.txt', 'Source 2.txt']
    # with open(r'C:\Users\Atman S\Documents\GM0 RSTs', 'w') as outfile:
    #     for fname in filenames:
    #         with open(fname) as infile:
    #             for line in infile:
    #                 outfile.write(line)
    directory = r'C:\Users\Atman S\Documents\GM0 RSTs\\'
    for files in os.listdir(directory):
        with open(directory + files, encoding="utf8") as f:
            lines = f.read()
            lines = lines.replace("-", "")
            lines = lines.replace("^", "")
            lines = lines.replace("\n", "")
            lines = lines.replace("Â°", " degrees")
            lines = lines.replace("\n", "")
            lines = lines.replace("~", "")
            lines = lines.replace("=", "")
            print(lines)
            print()
            print()

 
  

    # print(lines)    
    
    
    # text_splitter = CharacterTextSplitter(
    #     separator="\n",
    #     chunk_size=1000,
    #     chunk_overlap=200,
    #     length_function=len
    # )
    # chunks = text_splitter.split_text(lines)
    # print(chunks)

if __name__ == '__main__':
    main()
    