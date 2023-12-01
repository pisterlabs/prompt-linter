import openai, names, uuid, random, csv
import config

from tigerGraph import create_new_patient_vertex, check_existing_symptom,\
    check_existing_disease, confirm_diagnosis, provider_add_patient

openai.api_key = config.openai_key

class Patient:
    def __init__(self):
        self.unique_id = str(uuid.uuid4())
        self.dummy_id = self.unique_id[:2]
        self.gender = self._generate_gender()
        self.name = names.get_first_name(gender=self.gender)
        self.username = 'test'
        self.password = f'test{self.dummy_id}'
        self.email = f'test{self.dummy_id}@test.com'
        self.DOB = self._generate_DOB()

    def _generate_gender(self):
        binary = random.choice([0, 1])
        return 'male' if binary == 1 else 'female'

    def _generate_DOB(self):
        return str(random.randint(1922, 2010))

    def display_info(self):
        print("Name:", self.name)
        print("Gender:", self.gender)
        print("Username:", self.username)
        print("Password:", self.password)
        print("Email:", self.email)
        print("Date of Birth:", self.DOB)


def extract_row_by_index(csv_file, n):
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)
        if 0 <= n < len(rows):
            return rows[n]
        else:
            return []

# Example usage
csv_file = '../symptoms_and_diseases.csv'
def populate(n):
    row = 4472
    while row <= n:
        row_values = extract_row_by_index(csv_file, row)
        symptoms = []
        disease = row_values[0]
        for i in range(len(row_values)):
            if i == 0 or row_values[i] == "":
                pass
            else:
                value = row_values[i].lstrip()
                symptoms.append(value)

        patient = Patient()
        patient_id = create_new_patient_vertex(patient.name, patient.username, patient.password, patient.email, patient.DOB)

        # print("Symptoms: ", symptoms)
        symptom_id_list = check_existing_symptom(patient_id, symptoms)

        # print("symptom id list: ", symptom_id_list)
        # print("Disease: ", disease)
        disease_list_data = [disease]
        check_existing_disease(disease_list_data, symptom_id_list)
        care_provider_id = "CP123"
        provider_add_patient(patient_id, care_provider_id)
        confirm_diagnosis(disease, patient_id, care_provider_id)
        print(row)
        row += 1

populate(4920)



