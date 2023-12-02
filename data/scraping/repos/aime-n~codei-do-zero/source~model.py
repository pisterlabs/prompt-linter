import cohere 
import os
import re

api_key = 'y7Pemp9bBQAX1DUwtP8bFtKssS4qSudAlzhQh87S'
co = cohere.Client(api_key)
# model = 'command-nightly'
model = 'f2e29a92-b7c1-44b0-8344-610a442ac4d2-ft'

def write_to_file(response_text, base_filename="output_response.txt"):
    # Verifica se o arquivo já existe
    if os.path.exists(base_filename):
        # Se o arquivo existir, encontre um novo nome de arquivo
        index = 1
        while os.path.exists(f"{index}_{base_filename}"):
            index += 1
        filename = f"{index}_{base_filename}"
    else:
        filename = base_filename
    
    # Escreve o texto no arquivo
    with open(filename, 'w') as file:
        file.write(response_text)
    print(f"Response written to: {filename}")


prompt = '''
Below, I will give a song on an specified structure. 
In first metadata gives the name of the original song (You must change this on your answer).
In second metadata  gives the tags relationated to the song.
In third metadata gives the wiki's song. If they aren't 'None' they talk some about the context of the song.
The last metadata gives a sequence of three columns.
The strucuture of those three columns is very important. The description about how they works is: First one is the initial time of the chord, the second is the final time of the chord and the third is the chord by itself.

You must follow the exact same structure.

Generate a similar song to the one below. You must change the times of the columns and the chords to change the song by complete. Furthermore you have to change the original topic field and original energy field to new ones, because they will talk about the new song you generated.

the melody  starts here:
0.000	0.427	N
0.427	2.133	G
2.133	3.733	B:7
3.733	5.357	E:min
5.357	6.941	G:7
6.941	8.605	C
8.605	9.336	F
9.336	10.141	C
10.141	11.726	G
11.726	13.342	D/3
13.342	14.901	E:min
14.901	15.720	B:min
15.720	16.524	G
16.524	18.135	C
18.135	19.731	A:min7
19.731	22.940	D
22.940	24.528	G
24.528	26.141	B
26.141	27.713	E:min
27.713	29.304	G:7
29.304	32.493	A:min
32.493	35.724	D
35.724	37.317	C
37.317	38.933	A/3
38.933	40.519	G
40.519	42.098	E:min
42.098	43.712	D
43.712	45.314	B/3
45.314	46.909	E:min
46.909	48.525	D#:aug
48.525	50.125	G/5
50.125	51.717	A/3
51.717	54.869	C:maj6
54.869	58.093	D
58.093	59.685	G
59.685	61.324	B:7
61.324	62.901	E:min
62.901	64.533	G:7
64.533	66.107	C
66.107	67.672	A:min
67.672	68.524	F
68.524	69.317	C
69.317	70.888	G
70.888	72.533	D/3
72.533	74.053	E:min
74.053	74.877	B:min
74.877	75.701	G
75.701	77.293	C
77.293	78.884	A:min7
78.884	82.101	D
82.101	83.676	G
83.676	85.261	B
85.261	86.895	E:min
86.895	88.493	G:7
88.493	91.685	A:min
91.685	94.877	D
94.877	96.473	C
96.473	98.078	A/3
98.078	99.669	G
99.669	101.301	E:min
101.301	102.893	D
102.893	104.493	B/3
104.493	106.061	E:min
106.061	107.680	D#:aug
107.680	109.309	G/5
109.309	110.885	A/3
110.885	114.091	C:maj6
114.091	117.284	D
117.284	118.453	G:(1,5)
118.453	120.510	F:(1,5,9)
120.510	121.245	C
121.245	123.653	Bb
123.653	124.935	G:(1,5)
124.935	126.885	F:(1,5,9)
126.885	128.470	C
128.470	130.084	Bb
130.084	131.677	G
131.677	133.277	D/3
133.277	134.879	E:min
134.879	135.669	B:min
135.669	136.487	G
136.487	138.060	C
138.060	139.668	A:min
139.668	142.861	D
142.861	144.461	G
144.461	146.077	B:7
146.077	147.640	E:min
147.640	149.269	G:7
149.269	152.452	A:min
152.452	155.645	D
155.645	157.246	C
157.246	158.853	A/3
158.853	160.444	G
160.444	162.069	E:min
162.069	163.645	D
163.645	165.280	B/3
165.280	166.882	E:min
166.882	168.467	D#:aug
168.467	170.057	G/5
170.057	171.659	A/3
171.659	174.837	C:maj6
174.837	178.056	D
178.056	179.670	C
179.670	181.253	A/3
181.253	182.821	G
182.821	184.453	E:min
184.453	186.036	D
186.036	187.629	B/3
187.629	189.253	E:min
189.253	190.865	E:min/7
190.865	192.461	G/5
192.461	194.045	A/3
194.045	197.207	C:maj6
197.207	200.453	D
200.453	202.021	C
202.021	203.685	A/3
203.685	205.270	G
205.270	206.861	E:min
206.861	208.453	D
208.453	210.078	B/3
210.078	211.637	E:min
211.637	213.240	D#:aug
213.240	214.877	G/5
214.877	216.437	A/3
216.437	219.645	C:maj6
219.645	222.821	D
222.821	224.421	C
224.421	226.029	A/3
226.029	227.599	G
227.599	229.245	E:min
229.245	230.871	D
230.871	232.453	B/3
232.453	234.005	E:min
234.005	235.661	D#:aug
235.661	237.229	G/5
237.229	238.816	A/3
238.816	242.057	C
242.057	245.197	D
245.197	248.893	N

A possible example that I can give to you is the one below.

0.000	0.278	N
0.278	3.000	Eb
3.000	4.360	Ab
4.360	5.020	G:min
5.020	5.737	C:min
5.737	7.043	F:min
7.043	7.702	Ab
7.702	8.359	Bb
8.359	10.286	Eb
10.286	10.952	Bb
10.952	13.688	Eb
13.688	15.006	Ab
15.006	15.682	G:min
15.682	16.294	C:min
16.294	17.649	F:min
17.649	18.253	Ab:min
18.253	18.932	Bb:7
18.932	20.792	Eb
20.792	21.412	Bb
21.412	22.735	Eb
22.735	23.992	Bb/3
23.992	25.282	C:min
25.282	25.942	G:min
25.942	26.567	Ab
26.567	27.833	Eb
27.833	29.102	Bb/3
29.102	30.393	C:min
30.393	31.656	G/3
31.656	32.257	C:min
32.257	32.886	G:min
32.886	33.498	C:min
33.498	34.149	G:min
34.149	35.384	F:min
35.384	36.000	Ab:min
36.000	36.674	Bb:7
36.674	39.189	Eb
39.189	41.710	Ab:min
41.710	43.000	Eb/3
43.000	44.253	Eb:7
44.253	45.522	Ab:min
45.522	46.759	Gb:7
46.759	48.024	Cb
48.024	49.282	Gb:7
49.282	50.551	Cb
50.551	51.796	Gb:7
51.796	53.053	Cb
53.053	54.269	Bb
54.269	56.816	Eb
56.816	58.061	Ab
58.061	58.723	G:min
58.723	59.331	C:min
59.331	60.555	F:min
60.555	61.200	Ab:min
61.200	61.824	Bb:7
61.824	63.665	Eb
63.665	64.290	Bb
64.290	65.535	Eb
65.535	66.825	Bb/3
66.825	68.066	C:min
68.066	68.691	G:min
68.691	69.333	Ab
69.333	70.574	Eb
70.574	71.838	Bb/3
71.838	73.041	C:min
73.041	74.290	G/3
74.290	74.898	C:min
74.898	75.535	G:min
75.535	76.157	C:min
76.157	76.755	G:min
76.755	78.004	F:min
78.004	78.639	Ab:min
78.639	79.225	Bb:7
79.225	84.351	Eb
84.351	89.470	Eb
89.470	94.506	Bb:7
94.506	97.008	Eb
97.008	99.596	Bb:min7
99.596	102.147	F:7
102.147	103.029	Bb:7
103.029	104.577	C:7
104.577	110.808	F:min
110.808	112.082	Ab:min
112.082	113.331	Bb:7
113.331	114.580	Eb
114.580	115.816	Bb/3
115.816	117.089	C:min
117.089	117.711	G:min
117.711	118.322	Ab
118.322	119.609	Eb
119.609	120.833	Bb/3
120.833	122.057	C:min
122.057	123.314	G/3
123.314	123.983	C:min
123.983	124.548	G:min
124.548	125.183	C:min
125.183	125.770	G:min
125.770	127.029	F:min
127.029	128.241	Ab
128.241	129.518	G:min
129.518	130.780	Ab:min
130.780	131.392	Eb
131.392	132.008	Bb/3
132.008	132.600	C:min
132.600	133.240	G:min
133.240	134.523	F:min
134.523	135.172	Ab:min
135.172	135.722	Bb:7
135.722	138.180	Eb
138.180	139.429	Ab
139.429	140.049	G:min
140.049	140.624	C:min
140.624	141.914	F:min
141.914	142.527	Ab
142.527	143.167	Bb
143.167	144.939	Eb
144.939	145.571	Bb
145.571	148.012	Eb
148.012	149.229	Ab
149.229	149.912	G:min
149.912	150.482	C:min
150.482	151.718	F:min
151.718	152.327	Ab:min
152.327	152.939	Bb:7
152.939	154.727	Eb
154.727	155.359	Bb
155.359	156.596	Eb
156.596	157.809	Bb/3
157.809	159.016	C:min
159.016	159.637	G:min
159.637	160.253	Ab
160.253	161.502	Eb
161.502	162.804	Bb/3
162.804	163.973	C:min
163.973	165.176	G/3
165.176	165.780	C:min
165.780	166.408	G:min
166.408	167.016	C:min
167.016	167.633	G:min
167.633	168.922	F:min
168.922	170.105	Ab:min
170.105	171.355	Bb:7
171.355	172.812	Eb
172.812	173.106	Bb
173.106	173.763	Eb
173.763	175.827	N
'''

prompt_musica_1 = '''
0.000	0.427	N
0.427	2.133	G
2.133	3.733	B:7
3.733	5.357	E:min
5.357	6.941	G:7
6.941	8.605	C
8.605	9.336	F
9.336	10.141	C
10.141	11.726	G
11.726	13.342	D/3
13.342	14.901	E:min
14.901	15.720	B:min
15.720	16.524	G
16.524	18.135	C
18.135	19.731	A:min7
19.731	22.940	D
22.940	24.528	G
24.528	26.141	B
26.141	27.713	E:min
27.713	29.304	G:7
29.304	32.493	A:min
32.493	35.724	D
35.724	37.317	C
37.317	38.933	A/3
38.933	40.519	G
40.519	42.098	E:min
42.098	43.712	D
43.712	45.314	B/3
45.314	46.909	E:min
46.909	48.525	D#:aug
48.525	50.125	G/5
50.125	51.717	A/3
51.717	54.869	C:maj6
54.869	58.093	D
58.093	59.685	G
59.685	61.324	B:7
61.324	62.901	E:min
62.901	64.533	G:7
64.533	66.107	C
66.107	67.672	A:min
67.672	68.524	F
68.524	69.317	C
69.317	70.888	G
70.888	72.533	D/3
72.533	74.053	E:min
74.053	74.877	B:min
74.877	75.701	G
75.701	77.293	C
77.293	78.884	A:min7
78.884	82.101	D
82.101	83.676	G
83.676	85.261	B
85.261	86.895	E:min
86.895	88.493	G:7
88.493	91.685	A:min
91.685	94.877	D
94.877	96.473	C
96.473	98.078	A/3
98.078	99.669	G
99.669	101.301	E:min
101.301	102.893	D
102.893	104.493	B/3
104.493	106.061	E:min
106.061	107.680	D#:aug
107.680	109.309	G/5
109.309	110.885	A/3
110.885	114.091	C:maj6
114.091	117.284	D
117.284	118.453	G:(1,5)
118.453	120.510	F:(1,5,9)
120.510	121.245	C
121.245	123.653	Bb
123.653	124.935	G:(1,5)
124.935	126.885	F:(1,5,9)
126.885	128.470	C
128.470	130.084	Bb
130.084	131.677	G
131.677	133.277	D/3
133.277	134.879	E:min
134.879	135.669	B:min
135.669	136.487	G
136.487	138.060	C
138.060	139.668	A:min
139.668	142.861	D
142.861	144.461	G
144.461	146.077	B:7
146.077	147.640	E:min
147.640	149.269	G:7
149.269	152.452	A:min
152.452	155.645	D
155.645	157.246	C
157.246	158.853	A/3
158.853	160.444	G
160.444	162.069	E:min
162.069	163.645	D
163.645	165.280	B/3
165.280	166.882	E:min
166.882	168.467	D#:aug
168.467	170.057	G/5
170.057	171.659	A/3
171.659	174.837	C:maj6
174.837	178.056	D
178.056	179.670	C
179.670	181.253	A/3
181.253	182.821	G
182.821	184.453	E:min
184.453	186.036	D
186.036	187.629	B/3
187.629	189.253	E:min
189.253	190.865	E:min/7
190.865	192.461	G/5
192.461	194.045	A/3
194.045	197.207	C:maj6
197.207	200.453	D
200.453	202.021	C
202.021	203.685	A/3
203.685	205.270	G
205.270	206.861	E:min
206.861	208.453	D
208.453	210.078	B/3
210.078	211.637	E:min
211.637	213.240	D#:aug
213.240	214.877	G/5
214.877	216.437	A/3
216.437	219.645	C:maj6
219.645	222.821	D
222.821	224.421	C
224.421	226.029	A/3
226.029	227.599	G
227.599	229.245	E:min
229.245	230.871	D
230.871	232.453	B/3
232.453	234.005	E:min
234.005	235.661	D#:aug
235.661	237.229	G/5
237.229	238.816	A/3
238.816	242.057	C
242.057	245.197	D
245.197	248.893	N
'''
def get_musica_1():
    return prompt_musica_1

def get_prompt_padrao():
    return prompt

def get_response_from_promissor_prompt_continuacao(table):
    api_key = 'y7Pemp9bBQAX1DUwtP8bFtKssS4qSudAlzhQh87S'
    co = cohere.Client(api_key)

    prompt_continuacao = f'''
    Given the following chord progression in a song, continue after the end the chord sequence in a musically coherent manner, 
    maintaining the same format, repeat the chord progression and add the continuation:
    {table}
    '''

    model = 'command-nightly'
    # model = 'e0e24dd3-2818-4af4-b847-57a0f244e277-ft'
    # model = 'base'
    response = co.generate(  
        model=model,  
        prompt = prompt_continuacao,
        # max_tokens=200, # This parameter is optional. 
        temperature=0.7,
        max_tokens=500)

    response = response.generations[0].text
    print('Prediction:\n{}'.format(response))
    return response

def get_response_from_promissor_prompt():
    global function_running  # Usa a variável global
    function_running = True
    api_key = 'y7Pemp9bBQAX1DUwtP8bFtKssS4qSudAlzhQh87S'
    co = cohere.Client(api_key)

    # prompt_path = 'prompt\prompt_5_promissorporra - sem resposta.txt'
    # # Reading the entire file content at once
    # with open(prompt_path, 'r') as file:
    #     prompt = file.read()

    model = 'command-nightly'
    # model = 'e0e24dd3-2818-4af4-b847-57a0f244e277-ft'
    # model = 'base'
    response = co.generate(  
        model=model,  
        prompt = prompt,
        # max_tokens=200, # This parameter is optional. 
        temperature=0.7,
        max_tokens=200)

    response = response.generations[0].text
    print('Prediction:\n{}'.format(response))
    function_running = False
    return response

def generate_response(prompt : str):
    global function_running  # Usa a variável global
    function_running = True
    api_key = 'y7Pemp9bBQAX1DUwtP8bFtKssS4qSudAlzhQh87S'
    co = cohere.Client(api_key)
    model = 'command-nightly'
    # model = 'e0e24dd3-2818-4af4-b847-57a0f244e277-ft'
    # model = 'base'
    response = co.generate(  
        model=model,  
        prompt = prompt,
        temperature=0.7,
        max_tokens=200)
    response = response.generations[0].text
    print('Prediction:\n{}'.format(response))
    function_running = False
    return response

def write_to_file(response_text, base_filename="saidas_gustavo.txt"):
    # Verifica se o arquivo já existe
    if os.path.exists(base_filename):
        # Se o arquivo existir, encontre um novo nome de arquivo
        index = 1
        while os.path.exists(f"{index}_{base_filename}"):
            index += 1
        filename = f"{index}_{base_filename}"
    else:
        filename = base_filename
    
    # Escreve o texto no arquivo
    with open(filename, 'w') as file:
        file.write(response_text)
    print(f"Response written to: {filename}")

    response = response.generations[0].text
    print('Prediction: {}'.format(response))
    write_to_file(response, base_filename="output_response.txt")
    write_to_file(prompt, base_filename="prompt.txt")


def filter_table(output):

    # Using a regex to match lines that look like table rows
    table_lines = re.findall(r"([\d.]+)\s+([\d.]+)\s+(\S+)", output)

    # Joining each matching line with a newline character to get the table as a single string
    table_string = "\n".join(["   ".join(line) for line in table_lines])
    return table_string

if __name__ == "__main__":
    table = '''
0.000	0.427	N
0.427	2.133	G
2.133	3.733	B:7
3.733	5.357	E:min
5.357	6.941	G:7
6.941	8.605	C
8.605	9.336	F
9.336	10.141	C
10.141	11.726	G
11.726	13.342	D/3
13.342	14.901	E:min
14.901	15.720	B:min
'''
    get_response_from_promissor_prompt_continuacao(table)