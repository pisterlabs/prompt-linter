import os
import sys
import stat
import subprocess
import openai

# 1. Receive an argument $1 that is going to represent the technology that we want to
# bootstrap.
tech: str = sys.argv[1]
project_name: str = sys.argv[2]

# 2. Create the custom instruction based on that argument
print('Creating project {} using {}'.format(project_name, tech))

instruction: str = """Create a bash script that builds up a whole project of {}, including init project, dependencies, directories and files with sample code, call it '{}' and run in once its ready. echo each part of the process""".format(tech, project_name)

# 3. Give the instruction to openai and get a response
print('Getting completion...')
response = openai.Completion.create(
        model="text-davinci-003",
        prompt=instruction,
        temperature=0,
        max_tokens=1000
)
print('Completion ready!\n')
print(response.choices[0].text)

ans = input('Do you want to run the script [y/N]?')

if(ans not in ("y", "Y")):
    sys.exit('Program finished')

# 4. Create a directory called tmp/ and a file tmp/script.sh
print('Configuring script...')
script_file = open(os.path.join(os.getcwd(), "script.sh"), 'w')

# 5. chmod +x tmp/script.sh
st = os.stat('./script.sh')
os.chmod('./script.sh', st.st_mode | stat.S_IEXEC)

# 6. echo response > tmp/script.sh
script_file.write(response.choices[0].text)
script_file.close()
print('Script configured!')



# 7. Run that script
print('Running script...\n')
subprocess.call('./script.sh', shell=True)
