import openai
import re
openai.api_key = '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'
# Sample data structure (list of dictionaries)
all_data = [
    {'code': 'P0101', 'description':	'Mass air flow (MAF) sensor circuit, range or performance problem','fix': ''},
    {'code': 'P0102', 'description':	'Mass air flow (MAF) sensor circuit, low input','fix': '102'},
    {'code': 'P0103', 'description':	'Mass air flow (MAF) sensor circuit, high input','fix': ''},
    {'code': 'P0106', 'description':	'Manifold absolute pressure (MAP) sensor circuit, range or performance problem','fix': ''},
    {'code': 'P0107', 'description':	'Manifold absolute pressure (MAP) sensor circuit, low input','fix': ''},
    {'code': 'P0108', 'description':	'Manifold absolute pressure (MAP) sensor circuit, high input','fix': ''},
    {'code': 'P0112', 'description':	'Intake air temperature (IAT) circuit, low input','fix': ''},
    {'code': 'P0113', 'description':	'Intake air temperature (IAT) circuit, high input','fix': ''},
    {'code': 'P0117', 'description':	'Engine coolant temperature (ECT) circuit, low input','fix': ''},
    {'code': 'P0118', 'description':	'Engine coolant temperature (ECT) circuit, high input','fix': ''},
    {'code': 'P0121', 'description':	'Throttle position sensor (TPS) circuit, range or performance problem','fix': ''},
    {'code': 'P0122', 'description':	'Throttle position sensor (TPS) circuit, low input','fix': ''},
    {'code': 'P0123', 'description':	'Throttle position sensor (TPS) circuit, high input','fix': ''},
    {'code': 'P0125', 'description':	'Insufficient coolant temperature for closed loop fuel control','fix': ''},
    {'code': 'P0131', 'description':	'Oxygen sensor circuit, low voltage (pre-converter sensor, left bank)','fix': ''},
    {'code': 'P0132', 'description':	'Oxygen sensor circuit, high voltage (pre-converter sensor, left bank)','fix': ''},
    {'code': 'P0133', 'description':	'Oxygen sensor circuit, slow response (pre-converter sensor, left bank)','fix': ''},
    {'code': 'P0134', 'description':	'Oxygen sensor circuit - no activity detected (pre-converter sensor, left bank)','fix': ''},
    {'code': 'P0135', 'description':	'Oxygen sensor heater circuit malfunction (pre-converter sensor, left bank)','fix': ''},
    {'code': 'P0137', 'description':	'Oxygen sensor circuit, low voltage (post-converter sensor, left bank)','fix': ''},
    {'code': 'P0138', 'description':	'Oxygen sensor circuit, high voltage (post-converter sensor, left bank)','fix': ''},
    {'code': 'P0140', 'description':	'Oxygen sensor circuit - no activity detected (post-converter sensor, left bank)','fix': ''},
    {'code': 'P0141', 'description':	'Oxygen sensor heater circuit malfunction (post-converter sensor, left bank)','fix': ''},
    {'code': 'P0143', 'description':	'Oxygen sensor circuit, low voltage (#2 post-converter sensor, left bank)','fix': ''},
    {'code': 'P0144', 'description':	'Oxygen sensor circuit, high voltage (#2 post-converter sensor, left bank)','fix': ''},
    {'code': 'P0146', 'description':	'Oxygen sensor circuit - no activity detected (#2 post-converter sensor, left bank)','fix': ''},
    {'code': 'P0147', 'description':	'Oxygen sensor heater circuit malfunction (#2 post-converter sensor, left bank)','fix': ''},
    {'code': 'P0151', 'description':	'Oxygen sensor circuit, low voltage (pre-converter sensor, right bank)','fix': ''},
    {'code': 'P0152', 'description':	'Oxygen sensor circuit, high voltage (pre-converter sensor, right bank)','fix': ''},
    {'code': 'P0153', 'description':	'Oxygen sensor circuit, slow response (pre-converter sensor, right bank)','fix': ''},
    {'code': 'P0154', 'description':	'Oxygen sensor circuit - no activity detected (pre-converter sensor, right bank)','fix': ''},
    {'code': 'P0155', 'description':	'Oxygen sensor heater circuit malfunction (pre-converter sensor, right bank)','fix': ''},
    {'code': 'P0157', 'description':	'Oxygen sensor circuit, low voltage (post-converter sensor, right bank)','fix': ''},
    {'code': 'P0158', 'description':	'Oxygen sensor circuit, high voltage (post-converter sensor, right bank)','fix': ''},
    {'code': 'P0160', 'description':	'Oxygen sensor circuit - no activity detected (post-converter sensor, right bank)','fix': ''},
    {'code': 'P0161', 'description':	'Oxygen sensor heater circuit malfunction (post-converter sensor, right bank)','fix': ''},
    {'code': 'P0171', 'description':	'System too lean, left bank','fix': ''},
    {'code': 'P0172', 'description':	'System too rich, left bank','fix': ''},
    {'code': 'P0174', 'description':	'System too lean, right bank','fix': ''},
    {'code': 'P0175', 'description':	'System too rich, right bank','fix': ''},
    {'code': 'P0300', 'description':	'Engine misfire detected','fix': ''},
    {'code': 'P0301', 'description':	'Cylinder number 1 misfire detected','fix': ''},
    {'code': 'P0302', 'description':	'Cylinder number 2 misfire detected','fix': ''},
    {'code': 'P0303', 'description':	'Cylinder number 3 misfire detected','fix': ''},
    {'code': 'P0304', 'description':	'Cylinder number 4 misfire detected','fix': ''},
    {'code': 'P0305', 'description':	'Cylinder number 5 misfire detected','fix': ''},
    {'code': 'P0306', 'description':	'Cylinder number 6 misfire detected','fix': ''},
    {'code': 'P0307', 'description':	'Cylinder number 7 misfire detected','fix': ''},
    {'code': 'P0308', 'description':	'Cylinder number 8 misfire detected','fix': ''},
    {'code': 'P0325', 'description':	'Knock sensor circuit malfunction','fix': ''},
    {'code': 'P0327', 'description':	'Knock sensor circuit, low output','fix': ''},
    {'code': 'P0336', 'description':	'Crankshaft position sensor circuit, range or performance problem','fix': ''},
    {'code': 'P0337', 'description':	'Crankshaft position sensor, low output','fix': ''},
    {'code': 'P0338', 'description':	'Crankshaft position sensor, high output','fix': ''},
    {'code': 'P0339', 'description':	'Crankshaft position sensor, circuit intermittent','fix': ''},
    {'code': 'P0340', 'description':	'Camshaft position sensor circuit','fix': ''},
    {'code': 'P0341', 'description':	'Camshaft position sensor circuit, range or performance problem','fix': ''},
    {'code': 'P0401', 'description':	'Exhaust gas recirculation, insufficient flow detected','fix': ''},
    {'code': 'P0404', 'description':	'Exhaust gas recirculation circuit, range or performance problem','fix': ''},
    {'code': 'P0405', 'description':	'Exhaust gas recirculation sensor circuit low','fix': ''},
    {'code': 'P0410', 'description':	'Secondary air injection system','fix': ''},
    {'code': 'P0418', 'description':	'Secondary air injection pump relay control circuit','fix': ''},
    {'code': 'P0420', 'description':	'Catalyst system efficiency below threshold, left bank','fix': ''},
    {'code': 'P0430', 'description':	'Catalyst system efficiency below threshold, right bank','fix': ''},
    {'code': 'P0440', 'description':	'Evaporative emission control system malfunction','fix': ''},
    {'code': 'P0441', 'description':	'Evaporative emission control system, purge control circuit malfunction','fix': ''},
    {'code': 'P0442', 'description':	'Evaporative emission control system, small leak detected','fix': ''},
    {'code': 'P0446', 'description':	'Evaporative emission control system, vent system performance','fix': ''},
    {'code': 'P0452', 'description':	'Evaporative emission control system, pressure sensor low input','fix': ''},
    {'code': 'P0453', 'description':	'Evaporative emission control system, pressure sensor high input','fix': ''},
    {'code': 'P0461', 'description':	'Fuel level sensor circuit, range or performance problem','fix': ''},
    {'code': 'P0462', 'description':	'Fuel level sensor circuit, low input','fix': ''},
    {'code': 'P0463', 'description':	'Fuel level sensor circuit, high input','fix': ''},
    {'code': 'P0500', 'description':	'Vehicle speed sensor circuit','fix': ''},
    {'code': 'P0506', 'description':	'Idle control system, rpm lower than expected','fix': ''},
    {'code': 'P0507', 'description':	'Idle control system, rpm higher than expected','fix': ''},
    {'code': 'P0601', 'description':	'Powertrain Control Module, memory error','fix': ''},
    {'code': 'P0602', 'description':	'Powertrain Control module, programming error','fix': ''},
    {'code': 'P0603', 'description':	'Powertrain Control Module, memory reset error','fix': ''},
    {'code': 'P0604', 'description':	'Powertrain Control Module, memory error (RAM)','fix': ''},
    {'code': 'P0605', 'description':	'Powertrain Control Module, memory error (ROM)','fix': ''},
    # Add more data points as needed
]

# Function to find data based on error code
def find_data_by_code(query_code, data_list):
    query_code = query_code.lower()  # Convert query code to lowercase for case-insensitive comparison
    matched_data = next((data for data in data_list if data['code'].lower() == query_code), None)
    return matched_data

# Function to get an embedding vector for a given input using OpenAI's model
def get_vector(input_text):
    response = openai.Embed.create(
        model="text-embedding-ada-002",
        inputs=[input_text],
    )
    return response['data'][0]['embedding']


customer_question = "Code:P0161"

# Sample conversation prompt
prompt = """
i am a mechanic and i get a car's obd reading as"""+customer_question+"""what the problem and can you give fix solutions?"""


# Request the model to continue the conversation as MEC (AI Car Mechanic)
response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=prompt,
    max_tokens=150,
)
generated_response = response['choices'][0]['text']




if "Code:" in customer_question:
    code_position = customer_question.find("Code:")
    error_code_from_customer = customer_question[code_position + len("Code:"):].split()[0]

    print(f"Error Code from Customer: {error_code_from_customer}")
else:
    print("Error Code not found in the customer's question.")


# Find data related to the error code
matched_data = find_data_by_code(error_code_from_customer, all_data)

# Print the matched data
if matched_data:
    print(f"Matched Data: {matched_data}")
else:
    print(f"No data found for code: {error_code_from_customer}")

print("Generated Response:", generated_response)

