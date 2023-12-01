import sys
import csv
import openai

# check if the user supplied the correct number of command line arguments
if len(sys.argv) != 6:
    print("Usage: python titlegen.py inputfilename.csv  startlineno endlineno outputfilename.csv API_key")
    sys.exit(1)
    # set the api key from the command line argument
openai.api_key = sys.argv[5]
startlineno = int(sys.argv[2])
endlineno = int(sys.argv[3])
lineno = 0
with open(sys.argv[1], "r") as file:
        outfile = open(sys.argv[4], "w")
        writer = csv.writer(outfile)
        reader = csv.reader(file)
        for row in reader:
            lineno += 1
            if (lineno < startlineno or lineno > endlineno):
                continue
            # summarize the text in the 5th column using OpenAi's GPT-3
            # create a variable called prompt and set it as the concatenatenation of the string "Summarize this
            # in one sentence:" and the text in the 5th column
            desc = row[4]
            # extract the first 12 words of the description
            trimmed_desc  = " ".join(desc.split()[:12])
            prompt = "Summarize the following description of a museum item in one English phrase of strictly less than 4 words: " + trimmed_desc

            completion = openai.ChatCompletion.create(
                model = 'gpt-3.5-turbo',
                messages = [
                    {'role': 'user', 'content': prompt}
                ],
                temperature = 0  
            )

            # write the response as the last column in each row of the same csv file
            row.append(completion['choices'][0]['message']['content'])
            print(row)
            # write the row to the csv file

            writer.writerow(row)
        outfile.close()
        file.close()
