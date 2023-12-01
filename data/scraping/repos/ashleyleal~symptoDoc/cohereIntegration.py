import cohere
import sys

# Retrieve the userInputValue from command-line arguments
user_input_value = sys.argv[1]
# Process the user input in happy.py
# Example: Use the userInputValue as needed
print("User Input:", user_input_value)


from cohere.responses.classify import Example
co = cohere.Client('oMTBwIPop34GB5pxexB9hHbjEyfUYwmg3ZqowZuo') # This is your trial API key
response = co.classify(
  model='large',
  inputs=[user_input_value],
  examples=[Example("Heart health\nCardiology\nEchocardiogram\nElectrocardiogram (ECG/EKG)\n", "Cardiologist"), Example("Angiogram\nHeart disease\nHypertension\n ", "Cardiologist"), Example("Primary care\nCheck-ups\nDiagnosis\nTreatment\n", "General Practitioner/Family Physician"), Example("Preventive medicine\nPatient care\nHealth promotion\n", "General Practitioner/Family Physician"), Example("Child healthcare\nImmunizations\nGrowth and development\nPediatrics\n", "Pediatrician"), Example("Neonatology\nChild psychology\nPediatric diseases\n", "Pediatrician"), Example("Skin care\nDermatology\nAcne\nEczema\nPsoriasis\n", "Dermatologist:"), Example("Dermatological surgery\nDermatopathology\n", "Dermatologist:"), Example("Women's health\nObstetrics\nGynecology\nPrenatal care\n", "Gynecologist/Obstetrician"), Example("Menopause\nPap smear\nReproductive health\n", "Gynecologist/Obstetrician"), Example("Musculoskeletal system\nOrthopedics\nFractures\nJoint replacement\n", "Orthopedic Surgeon"), Example("Arthroscopy\nSports injuries\nOrthopedic surgery\n", "Orthopedic Surgeon"), Example("Mental health\nPsychiatry\nPsychotherapy\nDepression\n", "Psychiatrist"), Example("Anxiety\nSchizophrenia\nBipolar disorder\n", "Psychiatrist"), Example("Eye care\nOphthalmology\nCataracts\nGlaucoma\n", "Ophthalmologist"), Example("Retinal diseases\nLASIK surgery\nOptometry\n", "Ophthalmologist"), Example("Nervous system\nNeurology\nHeadaches\nEpilepsy\n", "Neurologist"), Example("Stroke\nMultiple sclerosis\nNeurological disorders\n", "Neurologist"),
Example("Cancer treatment\nOncology\nChemotherapy\nRadiation therapy\n", "Oncologist"), Example("Tumor\nPalliative care\nHematology\n", "Oncologist"), Example("Ear, nose, and throat\nOtolaryngology\nSinusitis\nTonsillitis\n", "ENT Specialist (Otolaryngologist"), Example("Hearing loss\nRhinoplasty\nLaryngology\n", "ENT Specialist (Otolaryngologist"), Example("Urinary tract\nUrology\nKidney stones\nProstate cancer\n", "Urologist"), Example("Incontinence\nBladder infections\nAndrology\n", "Urologist")])
print('The confidence levels of the labels are: {}'.format(response.classifications))