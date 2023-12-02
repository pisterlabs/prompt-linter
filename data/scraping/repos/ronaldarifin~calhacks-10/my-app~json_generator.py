import openai

# Replace 'YOUR_API_KEY' with your actual API key
api_key = ''


def executeChatGptCall(inputToGpt, token):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=inputToGpt,
        max_tokens=token,  # Adjust the number of tokens as needed
        api_key=api_key
    )
    return response.choices[0].text.strip()

def cvToJson(userCV, jsonFormat):
    inputToGpt = f"This is my CV: {userCV} Now, I want you return me a json object about my CV. I only want the json object representation like this {jsonFormat}. Nothing else just the json string."
    return executeChatGptCall(inputToGpt, 1500)



jobDescription = """Key Qualifications
You may meet or have interest in any one of the following qualifications:
Strong object-oriented design skills, coupled with a deep knowledge of data structures and algorithms
Proficiency in one or more of the following developer skills: Java, C/C++, PHP, Python, Ruby, Unix, MySQL, Clojure, Scala, Java Script, CSS, HTML5
Experience in sophisticated methodologies such as Data Modeling, Validation, Processing, Hadoop, MapReduce, Mongo, Pig
Experience with web frameworks such as AngularJS, NodeJS, SproutCore
Proven experience in application development in Objective-C for macOS or iOS a plus
Client-Server protocol & API design Skills
Able to craft multi-functional requirements and translate them into practical engineering tasks
A fundamental knowledge of embedded processors, with in-depth knowledge of real time operating system concepts.
Excellent debugging and critical thinking skills
Excellent analytical and problem-solving skills
Ability to work in a fast paced, team-based environment"""


userCV = """Matthew Kao
Berkeley, CA | matthewkao@berkeley.edu | (925)-435-9524 |https://github.com/Matthew-Kao |www.linkedin.com/mattkao EDUCATION University of California, Berkeley | B.A. Computer Science May 2023 | GPA: 3.7
SKILLS
Languages: Python, C++, C, Java, Javascript, Ruby, Dart
Front-end: HTML5, CSS3, Javascript, React.js, React Native, Flutter
WORK EXPERIENCE DBRIEF.AI
Lead App Developer
Tools: Figma, Github, Firebase, AWS, MongoDB Back-end: Node.js, Next.js, Express, SQL, MongoDB, Rust
Berkeley, CA
May 2022 – December 2022
● LedthecreationoftheDbrief.AImobileappfromscratchusingDartandFlutter
● ConstructedandDesignedtheUserInterfaceusingFigma,employingbestpracticesinUXdesignandincorporatinguserfeedbackto
enhance usability and boost user satisfaction
● ArchitectedthesynchronizationbetweentheApplicationandBackendAPI,enablingdynamiccontentupdatesbasedonthelatest news, driving a 15% growth in daily active users
● LeveragedFirebase’sreal-timedatabasecapabilitiestoenabledynamicdataupdates,enhancingtheapplication’sresponsiveness and providing users with up-to-date information
● Engineered and deployed innovative ‘Tinder-style’ swiping features, granting users the capability to showcase political viewpoints through left/right swipes resulting in a 20% increase in time spent on the app per user
● Guided and Mentored a team of 4 through iterative development cycles, by implementing agile methodologies, leading daily stand-ups, and facilitating cross-functional collaboration to deliver the Application on time
Cryptok Berkeley, CA
Founder Aug 2022 – May 2023
● Collaborating within a team of four, Co-Founded and launched an MVP of a web3 based Video Platform that tokenizes everything on the service based on User Engagement (Likes/Dislikes, Comments, Shares)
● Optimized User Experience and performance by leveraging Flutter for Front-End development, resulting in a 10% decrease in app crashes
● ImplementedSolidityforBack-EndSmartContractsonBlockchainPlatforms,enablingseamlessandtamper-prooftransactions, leading to a 15% reduction in transaction processing time
● TestedtheMVPwith200+studentsandreceivedpositivefeedback(morethan70%willingtoswitchplatforms) \
HIGHLIGHTED PROJECTS
Action Map
Tech Stack: Ruby, Model-View-Controller (MVC), CSS, JavaScript, Rspec Tests, Cucumber Scenarios
Jan 2023 – May 2023
● Responsibleforfixing/debugginganApplicationthatDisplaystheUSMapwheretheStateandCountiesareclickable,whichwill then show the Political Candidates – Allowing users to visualize the Political environment within all levels of government
● Establishedtheinterconnectivitybetweenthevariouscontrollers,models,andviewsthatallowtheapptosearchtheGoogleCivic Information API to look up Political Candidates in specific states
● RefactoredlegacycodeofanoldApplicationbywritingCharacterizationTestsandthenfollowingRed-Green-Refactorprinciples
● ImprovedTestCoverageofeveryFileinthedirectorytomorethan90%byusingRspecandCucumbertests
Pacman Aug 2022 – Dec 2022 Tech Stack: Python, BFS, DFS, A*, Monte Carlo Tree Search, Reinforcement Learning
● Developed a Pacman AI project focused on optimizing point collection and decision-making within the game environment
● Programmed the AI algorithms in Python, leveraging data structures and algorithms to enhance game-solving capabilities
● Executed and compared various search algorithms, including Breadth-First Search, A* Search, Depth-First Search, and Monte Carlo Tree Search to enable Pacman to navigate the maze efficiently while maximizing point acquisition
● Employed Reinforcement Learning techniques, specifically Q-learning, to train the Pacman agent in making intelligent decisions by learning from interactions with the game environment
Scheme Interpreter Jan 2022 – May 2022 Tech Stack: Scheme, Python(Tree-Recursion), Turtle Graphics
● Architected a highly efficient and functional code interpreter for the Scheme language, revolutionizing the development process and enabling seamless execution of complex programs
● Implemented Turtle Graphics to create a Tree-Recursive Program using Python that can Read User Input in Scheme, Evaluate and Display the Accurate output in Scheme Language"""


jsonFormat = """
    {... include name, location, email, phone, links, education, and skills
    "work experience": [
        {
            "employer": "Tokenized",
            "position": "Senior Javascript Developer",
            "location": "Melbourne",
            "summary": "Tokenized is a Bitcoin wallet for issuing, managing and trading digital tokens. I built out the front end which was packaged as an electron app. It was a difficult frontend to build because we store the users keys locally and used them to sign transactions and contracts.",
            "website": "https://tokenized.com/",
            "startDate": "2020-05-05",
            "highlights": ["React", "Redux", "SCSS", "Product"]
        },
        ... // And so on for the rest of the objects
        ... we only care about "work experience and projects only"
    ]
}

"""
# Call the function with your sample data
generatedResume = cvToJson(userCV, jsonFormat= jsonFormat)

# Print the generated resume
print(generatedResume)