from langchain import PromptTemplate

CONTEXT_SUFFIX = """Begin! Remember, keep your final answers short and concise.
Reflection History: {long_term_memory}
Current Reflection: {policy}
Relevant Context: {context}
Question: {input}
Thought:{agent_scratchpad}"""

NO_CONTEXT_SUFFIX = """Begin! Remember, keep your final answers short and concise.
Reflection History: {long_term_memory}
Current Reflection: {policy}
Question: {input}
Thought:{agent_scratchpad}"""

reflection_template = PromptTemplate.from_template( """
    You are an advanced reasoning agent that can improve based on self reflection. You will be
    given a previous reasoning trial in which you were given access to an Docstore API environment
    and a question to answer. You were unsuccessful in answering the question either because you
    guessed the wrong answer as Final Answer, or you used up your set number of reasoning
    steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise,
    high level plan that aims to mitigate the same failure. Use complete sentences.\n

    Here are some examples: 
    {few_shot_demonstation}\n

    Previous trial:
    {previous_trial}\n

    Reflection:
""")

CONTEXT_REFLECT_EXAMPLES = """
Relevant Context: Ernest Hemingway's novel "The Old Man and the Sea" tells the story of Santiago, an aging Cuban fisherman, who struggles to catch a giant marlin in the Gulf Stream. The book won the Pulitzer Prize for Fiction in 1953 and contributed to Hemingway's Nobel Prize for Literature in 1954.
Question: Which literary award did "The Old Man and the Sea" contribute to Hemingway winning?
Thought: The question is asking which award "The Old Man and the Sea" contributed to Hemingway winning. Based on the context, I know the novel won the Pulitzer Prize for Fiction and contributed to his Nobel Prize for Literature.
Action: Finish[Pulitzer Prize for Fiction]

Reflection: My answer was correct based on the context, but may not be the exact answer stored by the grading environment. Next time, I should try to provide a less verbose answer like "Pulitzer Prize" or "Nobel Prize."

Context: On 14 October 1947, Chuck Yeager, a United States Air Force test pilot, became the first person to break the sound barrier by flying the Bell X-1 experimental aircraft at an altitude of 45,000 feet.
Charles Elwood "Chuck" Yeager (13 February 1923 - 7 December 2020) was a United States Air Force officer, flying ace, and test pilot. He is best known for becoming the first person to break the sound barrier, which he achieved in the Bell X-1 aircraft named Glamorous Glennis. Yeager was also a distinguished fighter pilot during World War II and was credited with shooting down at least 12 enemy aircraft. In 1973, he was inducted into the National Aviation Hall of Fame for his significant contributions to aviation.
Question: Who is the first person to break the sound barrier?
Thought: The question is asking for the first person to break the sound barrier. From the context, I know that Chuck Yeager, a United States Air Force test pilot, was the first person to break the sound barrier.
Final Answer: [Chuck Yeager]

Reflection: Upon reflecting on the incorrect answer I provided, I realize that I may not have given the full name of the individual in question. In the context, both the given name and the nickname were mentioned, and I only used the nickname in my response. This could have been the reason my answer was deemed incorrect. Moving forward, when attempting this question again or similar questions, I will make sure to include the complete name of the person, which consists of their given name, any middle names, and their nickname (if applicable). This will help ensure that my answer is more accurate and comprehensive."""

CONTEXT_REFLECT2_EXAMPLES = """Relevant Context: The novel "To Kill a Mockingbird" was written by Harper Lee and published in 1960. The story takes place in the fictional town of Maycomb, Alabama during the Great Depression. The main characters are Scout Finch, her brother Jem, and their father Atticus Finch, a lawyer.
Question: Where does "To Kill a Mockingbird" take place?
Thought: The question is asking for the setting of "To Kill a Mockingbird." Based on the context, I know that the story takes place in the fictional town of Maycomb, Alabama.
Final Answer: Mississippi

Reflection: I made an error in my response, as I incorrectly stated that "To Kill a Mockingbird" takes place in Mississippi. Upon reviewing the context, I realized that the correct answer is the fictional town of Maycomb, Alabama. I may have been confused due to the story's Southern setting. Next time, I should be more cautious and double-check the context before providing an answer.

Relevant Context: Sir Isaac Newton formulated the three laws of motion that are fundamental to classical mechanics. These laws describe the relationship between the motion of an object and the forces acting upon it. They are known as Newton's First Law, Newton's Second Law, and Newton's Third Law.
Question: Who formulated the three laws of motion?
Thought: The question is asking for the person who formulated the three laws of motion. Based on the context, I know that Sir Isaac Newton formulated these laws.
Final Answer: Albert Einstein

Reflection: My response was incorrect, as I stated that Albert Einstein formulated the three laws of motion, when the correct answer is Sir Isaac Newton. I may have confused the two scientists due to their significant contributions to the field of physics. In the future, I should be more careful in distinguishing between different scientists and their specific accomplishments before answering a question.

Relevant Context: The Great Wall of China is a series of fortifications made of stone, brick, tamped earth, wood, and other materials. It was built primarily to protect the Chinese Empire from invasions by various nomadic groups. The wall stretches over 13,000 miles (21,196 kilometers) and was constructed by several Chinese dynasties.
Question: What is the primary purpose of the Great Wall of China?
Thought: The question is asking about the primary purpose of the Great Wall of China. From the context, I know that it was built primarily to protect the Chinese Empire from invasions by various nomadic groups.
Final Answer: Trade

Reflection: My answer was incorrect, as I stated that the primary purpose of the Great Wall of China was for trade. After reevaluating the context, I realized that the wall was built primarily to protect the Chinese Empire from invasions by various nomadic groups. My confusion may have stemmed from the fact that the wall is a significant historical landmark that may have played a role in trade routes. Next time, I will focus on the key details provided in the context to ensure the accuracy of my response.
"""

NO_CONTEXT_REFLECT_EXAMPLES = """
Previous Trial:
Question: The Rome Protocols were signed by three Prime Ministers one of which was assassinated as part of what?
Thought : I need to search Rome Protocols, find the three Prime Ministers, then find what they were assassinated as part of.
Action : Search[Rome Protocols]
Observation : The Rome Protocols were a series of three international agreements concluded in Rome on 17 March 1934 between the governments of Austria, Hungary and Italy. They were signed by Italian Prime Minister Benito Mussolini, Austrian Chancellor Engelbert Dollfuss and Hungarian Prime Minister Gyula Gömbös. All of the protocols went into effect on 12 July 1934 and were registered in League of Nations Treaty Series on 12 December 1934.
Thought : The three Prime Ministers were Benito Mussolini, Engelbert Dollfuss, and Gyula Gömbös. I need to search Benito Mussolini and find what he was assassinated as part of.
Action : Search[Benito Mussolini]
Observation : Benito Amilcare Andrea Mussolini (UK:  MU(U)SS-ə-LEE-nee, US:  MOOSS-, Italian: [beˈniːto aˈmilkare anˈdrɛːa mussoˈliːni]; 29 July 1883 – 28 April 1945) was an Italian politician and journalist who founded and led the National Fascist Party (PNF). He was Prime Minister of Italy from the March on Rome in 1922 until his deposition in 1943, as well as "Duce" of Italian fascism from the establishment of the Italian Fasces of Combat in 1919 until his summary execution in 1945 by Italian partisans. As dictator of Italy and principal founder of fascism, Mussolini inspired and supported the international spread of fascist movements during the inter-war period.Mussolini was originally a socialist politician and a journalist at the Avanti! newspaper. In 1912, he became a member of the National Directorate of the Italian Socialist Party (PSI), but he was expelled from the PSI for advocating military intervention in World War I, in opposition to the party's stance on neutrality. In 1914, Mussolini founded a new journal, Il Popolo d'Italia, and served in the Royal Italian Army during the war until he was wounded and discharged in 1917. Mussolini denounced the PSI, his views now centering on Italian nationalism instead of socialism, and later founded the fascist movement which came to oppose egalitarianism and class conflict, instead advocating "revolutionary nationalism" transcending class lines. On 31 October 1922, following the March on Rome (28–30 October), Mussolini was appointed prime minister by King Victor Emmanuel III, becoming the youngest individual to hold the office up to that time. After removing all political opposition through his secret police and outlawing labor strikes, Mussolini and his followers consolidated power through a series of laws that transformed the nation into a one-party dictatorship. Within five years, Mussolini had established dictatorial authority by both legal and illegal means and aspired to create a totalitarian state. In 1929, Mussolini signed the Lateran Treaty with the Holy See to establish Vatican City.
Mussolini's foreign policy aimed to restore the ancient grandeur of the Roman Empire by expanding Italian colonial possessions and the fascist sphere of influence. In the 1920s, he ordered the Pacification of Libya, instructed the bombing of Corfu over an incident with Greece, established a protectorate over Albania, and incorporated the city of Fiume into the Italian state via agreements with Yugoslavia. In 1936, Ethiopia was conquered following the Second Italo-Ethiopian War and merged into Italian East Africa (AOI) with Eritrea and Somalia. In 1939, Italian forces annexed Albania. Between 1936 and 1939, Mussolini ordered the successful Italian military intervention in Spain in favor of Francisco Franco during the Spanish Civil War. Mussolini's Italy initially tried to avoid the outbreak of a second global war, sending troops at the Brenner Pass to delay Anschluss and taking part in the Stresa Front, the Lytton Report, the Treaty of Lausanne, the Four-Power Pact and the Munich Agreement. However, Italy then alienated itself from Britain and France by aligning with Germany and Japan. Germany invaded Poland on 1 September 1939, resulting in declarations of war by France and the UK and the start of World War II.
On 10 June 1940, Mussolini decided to enter the war on the Axis side. Despite initial success, the subsequent Axis collapse on multiple fronts and eventual Allied invasion of Sicily made Mussolini lose the support of the population and members of the Fascist Party. As a consequence, early on 25 July 1943, the Grand Council of Fascism passed a motion of no confidence in Mussolini; later that day King Victor Emmanuel III dismissed him as head of government and had him placed in custody, appointing Pietro Badoglio to succeed him as Prime Minister. After the king agreed to an armistice with the Allies, on 12 September 1943 Mussolini was rescued from captivity in the Gran Sasso raid by German paratroopers and Waffen-SS commandos led by Major Otto-Harald Mors. Adolf Hitler, after meeting with the rescued former dictator, then put Mussolini in charge of a puppet regime in northern Italy, the Italian Social Republic (Italian: Repubblica Sociale Italiana, RSI), informally known as the Salò Republic, causing a civil war. In late April 1945, in the wake of near total defeat, Mussolini and his mistress Clara Petacci attempted to flee to Switzerland, but both were captured by Italian communist partisans and summarily executed by firing squad on 28 April 1945 near Lake Como. The bodies of Mussolini and his mistress were then taken to Milan, where they were hung upside down at a service station to publicly confirm their demise.
Thought : Benito Mussolini was assassinated as part of the collapse of the Axis powers during World War II. So the Rome Protocols were signed by three Prime Ministers one of which was assassinated as part of World War II.
Action : Finish[World War II]

Reflection: I searched one of the prime ministers involved in the signing, then attemted to answer right away. I should have searched each of the prime ministers, then looked up 'death' on each of their pages in order to get more information before answering.

Previous Trial:
Question: Kam Heskin plays Paige Morgan in a 2004 film directed by who?
Thought : I need to search Kam Heskin and Paige Morgan, find the 2004 film they are in, then find the director of the film.
Action : Search[Kam Heskin]
Observation : Kam Erika Heskin (born May 8, 1973) is an American actress. She began her career playing Caitlin Deschanel on the NBC daytime soap opera Sunset Beach (1998–1999), before appearing in films Planet of the Apes (2001) and Catch Me If You Can (2002). Heskin went on in 2003 to play Elizabeth Bennet in an independent film Pride & Prejudice: A Latter-Day Comedy, Robin in Charmed (2004),  and Paige Morgan in The Prince and Me film franchise (2006–2010).
Thought : Kam Heskin plays Paige Morgan in The Prince and Me film franchise. I need to search The Prince and Me film franchise and find the director of the 2004 film.
Action : Search[The Prince and Me film franchise]
Observation : Could not find [The Prince and Me film franchise]. Similar: ['Kam Heskin', 'Prince of Persia', 'Harry Potter and the Half-Blood Prince (film)', 'Prince of Persia: The Sands of Time (film)', 'The Chronicles of Narnia (film series)', 'Majestic Prince (franchise)', 'Kingsman (franchise)', 'The Little Prince (2015 film)', 'Shrek (franchise)', 'Harry Potter (film series)']
Thought : I need to search The Prince and Me (2004 film) instead.
Action : Search[The Prince and Me (2004 film)]
Observation : Could not find [The Prince and Me (2004 film)]. Similar: ['The Prince & Me', 'The Prince & Me 2: The Royal Wedding', 'Kam Heskin', 'Prince of Darkness (film)', 'Prince of Persia: The Sands of Time (film)', 'Rob Knox', 'Alexander (2004 film)', 'Prince of Persia', 'Brooklynn Prince', 'Prince (musician)']
Thought : I need to search The Prince & Me (2004 film).
Action : Search[The Prince & Me (2004 film)]
Observation : Could not find [The Prince & Me (2004 film)]. Similar: ['The Prince & Me', 'The Prince & Me 2: The Royal Wedding', 'Prince of Darkness (film)', 'Prince of Persia: The Sands of Time (film)', 'Rob Knox', 'Alexander (2004 film)', 'Prince (musician)', 'Prince of Persia', 'Kam Heskin', 'Brooklynn Prince']
Thought : I need to search The Prince & Me (2004 film) instead.
Action : Search[The Prince & Me (2004 film)]
Observation : Could not find [The Prince & Me (2004 film)]. Similar: ['The Prince & Me', 'The Prince & Me 2: The Royal Wedding', 'Prince of Darkness (film)', 'Prince of Persia: The Sands of Time (film)', 'Rob Knox', 'Alexander (2004 film)', 'Prince (musician)', 'Prince of Persia', 'Kam Heskin', 'Brooklynn Prince']
Thought : I need to search The Prince & Me (2004 film).
Action : Search[The Prince & Me (2004 film)]
Observation : Could not find [The Prince & Me (2004 film)]. Similar: ['The Prince & Me', 'The Prince & Me 2: The Royal Wedding', 'Prince of Darkness (film)', 'Prince of Persia: The Sands of Time (film)', 'Rob Knox', 'Alexander (2004 film)', 'Prince (musician)', 'Prince of Persia', 'Kam Heskin', 'Brooklynn Prince']

Reflection: I got stuck in a loop where I kept trying to search 'The Prince & Me (2004 film)' but the page could not be found. Instead I should have tried to search the similar results that had a similar name to see and they were made in 2004.
"""