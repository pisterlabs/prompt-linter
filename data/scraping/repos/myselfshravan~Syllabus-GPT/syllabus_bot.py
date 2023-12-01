import anthropic
import streamlit as st

client = anthropic.Client(
    api_key=st.secrets["apikey"])
subjects = [
    "Data Communication and Networking (DCN)",
    "Design and Analysis of Algorithms (DAA)",
    "Introduction to Artificial Intelligence (AI)",
    "Operating Systems (OS)",
]
syllabus_dict = {
    "Data Communication and Networking (DCN)": {
        "UNIT-I": "Data communication Fundamentals: Introduction, components, Data Representation, Data Flow; Networks – Network criteria, Physical Structures, Network Models, Categories of networks; Protocols, Standards, Standards organization; The Internet – Brief history, Internet today; Network Models -Layered tasks; The OSI model – Layered architecture, Peer-to-Peer Process, Encapsulation; Layers in the OSI model; TCP/IP Protocol suite; Addressing.",
        "UNIT-II": "Digital Transmission Fundamentals (with problems to solve): Analog & Digital data, Analog & Digital signals (basics); Transmission Impairment – Attenuation, Distortion and Noise; Data rate limits – Nyquist Bit Rate, Shannon Capacity; Performance, Digital Transmission (with problems to solve): Digital-to-Digital conversion - Line coding, Line coding schemes (unipolar, polar, bipolar); Analog-to-Digital conversion - PCM.",
        "UNIT-III": "Error detection & correction(with problems to solve): Introduction, Block coding, Linear Block codes, Cyclic codes – CRC, Polynomials, Checksum, Datalink control: Framing, Flow& error control, Protocols, Noiseless channels (Simplest Protocol, Stop-and-wait protocol); Noisy channels (Stop-and-wait ARQ, Go-Back-N ARQ, Selective Repeat ARQ, Piggybacking).",
        "UNIT-IV": "HDLC – Transfer modes, frames: Point-to-Point Protocol – Framing, transition phases; Multiple Access: Random Access (Aloha, CSMA, CSMA/CD, CSMA/CA), Controlled Access (Reservation, Polling, Token Passing), Channelization (FDMA, TDMA, CDMA)",
        "UNIT-V": "Wired LANs: IEEE standards; Standard Ethernet; Wireless LANs: IEEE802.11 Architecture, MAC sublayer, addressing mechanism, Bluetooth and its architecture; Connecting devices, Backbone networks, Virtual LANs."
    },
    "Design and Analysis of Algorithms (DAA)": {
        "Unit I": "Asymptotic Bounds and Representation problems of Algorithms: Computational Tractability: Some Initial Attempts at Defining Efficiency, Worst-Case Running Times and Brute-Force Search, Polynomial Time as a Definition of Efficiency, Asymptotic Order of Growth: Properties of Asymptotic Growth Rates, Asymptotic Bounds for Some Common Functions, A Survey of Common Running Times: Linear Time, O(n log n) Time, O(nk) Time, Beyond Polynomial Time. Some Representative Problems, A First Problem: Stable Matching.",
        "Unit II": "Graphs & Divide and Conquer: Graph Connectivity and Graph Traversal, Breadth-First Search: Exploring a Connected Component, Depth-First Search, Implementing Graph Traversal Using Queues and Stacks: Implementing Breadth-First Search, Implementing Depth-First Search, An Application of Breadth-First Search: The Problem, Designing the Algorithm, Directed Acyclic Graphs and Topological Ordering, The Merge sort Algorithm.",
        "Unit III": "Greedy Algorithms: Interval Scheduling: The Greedy Algorithm Stays Ahead: Designing a Greedy Algorithm, Analyzing the Algorithm, Scheduling to Minimize Lateness: An Exchange Argument: The Problem, Designing the Algorithm, Designing and Analyzing the Algorithm, Shortest Paths in a Graph: The Problem, Designing the Algorithm, Analyzing the Algorithm, The Minimum Spanning Tree Problem: The Problem, Designing Algorithms, Analyzing the Algorithms, Huffman Codes and Data Compression.",
        "Unit IV": "Dynamic Programming: Weighted Interval Scheduling: A Recursive Procedure: Designing a Recursive Algorithm, Subset Sums and Knapsacks: Adding a Variable: The Problem, Designing the Algorithm, Shortest Paths in a Graph: The Problem, Designing the Algorithm, The Maximum-Flow Problem.",
        "Unit V": "NP and Computational Intractability: Polynomial-Time Reductions NP-Complete Problems: Circuit Satisfiability: A First NP-Complete Problem, General Strategy for Proving New Problems NPComplete, Sequencing Problems: The Traveling Salesman Problem, The Hamiltonian Cycle Problem."
    },
    "Introduction to Artificial Intelligence (AI)": {
        "Unit I": "Introduction: Why study AI? What is AI? The Turing test. Rationality. Branches of AI. Brief history of AI. Challenges for the future. What is an intelligent agent? Doing the right thing (rational action). Performance measure. Autonomy, Environment and agent design, Structure of Agents, Agent types. Uninformed Search: Depth-first, Breadth-first, Uniform-cost, Depth-limited, Iterative deepening. Informed search: Best-first, A* search, Heuristics, Hill climbing, Problem of local extrema.",
        "Unit II": "Game Playing: The minimax algorithm, Resource limitations, Alpha-beta pruning, Constraint satisfaction, Node, arc, path, and k-consistency, Backtracking search, Local search using min-conflicts.",
        "Unit III": "Agents that reason logically 1: Knowledge -based agents, Logic and representation, Propositional (Boolean) logic. Agents that reason logically 2: Inference in propositional logic, Syntax, Semantics, Examples.",
        "Unit IV": "Advanced problem solving paradigm: Planning: types of planning sytem, block world problem, logic based planning, Linear planning using a goal stack, Means-ends analysis, Non linear planning strategies, learning plan.",
        "Unit V": "Knowledge Representation, Expert system Approaches to knowledge representation, knowledge representation using semantic network, Knowledge representation using Frames. Expert system: introduction phases, architecture ES verses Traditional system."
    },
    "Operating Systems (OS)": {
        "Unit I": "Introduction: What operating systems do; Computer System organization; Computer System architecture; Operating System structure; Operating System operations; Operating system structures: operating system services, user operating system Interface, System calls, Types of system calls, Operating system structure, System boot.",
        "Unit II": "Process Management: Basic concept; Process scheduling; Operations on processes; Inter process Communication. Threads: Overview; Multithreading models; Process scheduling: Basic concepts, Scheduling criteria, scheduling algorithms, multiple processor scheduling, Algorithm evaluation.",
        "Unit III": "Process Synchronization: Synchronization, The Critical section problem; Peterson's solution; Synchronization hardware; Semaphores; Classical problems of synchronization; Monitors. Deadlocks: System model; Deadlock characterization; Methods for handling deadlocks; Deadlock prevention; Deadlock avoidance; Deadlock detection and recovery from deadlock.",
        "Unit IV": "Memory Management Strategies: Background; Swapping; Contiguous memory allocation; Paging; Structure of page table; Segmentation. Virtual Memory Management: Background; Demand paging; Copy-on write; Page replacement; Allocation of frames; Thrashing.",
        "Unit V": "File System: File concept; Access methods; Directory structure; File system mounting; file sharing; protection. Secondary Storage Structures: Disk scheduling; FCFS Scheduling, SSTF scheduling, SCAN, C-SCAN scheduling, Look Scheduling, CLOOK scheduling. System Protection: Goals of protection, Principles of protection, Domain of protection, Access matrix."
    }
}
st.set_page_config(page_title="Syllabus GPT - by Shravan", page_icon="📚")
st.title("AI Syllabus Bot 🤖")
st.subheader("Pick the Subject")
subject = st.radio("Subjects", subjects)

st.write(f"### Syllabus of {subject}")
syllabus = syllabus_dict[subject]
with st.expander("Show Units"):
    for unit in syllabus:
        st.write(f"#### {unit}")
        st.write(syllabus[unit])
st.subheader("Pick the Unit")
selected_units = st.radio("Unit", list(syllabus.keys()))
selected_syllabus = syllabus[selected_units]
example_prompts = {
    "0. Null Prompt": "Write Your Own Prompt Here",
    "1.Breakdown Topics": f"Act as a Comprehensive Engineering Course Syllabus guide for {subject} and the topics are {selected_syllabus}. Including all topics covered with brief and detailed explanations, and possible examples.",
    "2. Syllabus Unit": f"Engineering Course Content of {subject} and the topics are {selected_syllabus}. It covers all the topics and explains each of them briefly and in a detailed manner. Provided examples where necessary.",
    "3. Topic Explanation": f"Please provide a brief and detailed explanation of each topic covered in the engineering course content of {subject}. If necessary, you may also provide an example for each topic. The following is a list of topics that you should cover: [{selected_syllabus}]",
    "4. Simplify Information": f"Break down the topics: [{selected_syllabus}] into smaller pieces that are easier to understand. Use analogies and real-life examples to simplify the concept and make it more relatable.",
    "5. Provide Examples": f"Provide examples for each topic: [{selected_syllabus}] to aid understanding. You may also provide examples of real-life applications of the topic.",
    "6. Study Schedule": f"I need help organizing my study time for {subject}. The schedule should include the topics [{selected_syllabus}]. and the time required to study each topic. You may also provide a list of resources that can be used to study the topics. Can you create a study schedule for me, including breaks and practice exercises?",
    "7. Memorization Techniques": f"I need help memorizing the topics [{selected_syllabus}]. Can you provide me with some memorization techniques that I can use to memorize the topics?",
}
st.write("You can either pick a pre writen prompts or write/modify your own prompt to query the selected unit.")
st.caption(f"The response will be based on the following {subject}, {selected_units} ")
st.subheader("Pick the Guide")
guide = st.radio("Guide", list(example_prompts.keys()))
prompt = example_prompts[guide]
st.subheader("Write the Prompt")
text_in = st.text_area("Prompt", prompt)
prompt = f"Consider the {subject} and the topics are {selected_syllabus}. I want you to take the following question and answer based on the syllabus. {text_in}"

check = st.button("Ask the AI")
if check:
    response = client.completion(
        prompt=f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}",
        model="claude-1",
        max_tokens_to_sample=400,
    )
    out = response['completion']
    st.write(out)

hide_streamlit_style = """
                    <style>
                    # MainMenu {visibility: hidden;}
                    footer {visibility: hidden;}
                    footer:after {
                    content:'Made with Passion by Shravan Revanna'; 
                    visibility: visible;
    	            display: block;
    	            position: relative;
    	            # background-color: red;
    	            padding: 15px;
    	            top: 2px;
    	            }
                    </style>
                    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
