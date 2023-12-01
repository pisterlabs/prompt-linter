import openai

openai.api_key = ""

def parse_acknowledgements(acknowledgements):
	prompt = """ 
Task 1: Identify the sections related to funding in the text:
i. Identify key phrases often used in such acknowledgements, such as 'funded by', 'grants', 'financial contributions', 'support provided by', 'award', etc.
ii. Extract sentences or phrases that contain these key phrases.

Task 2: Extract the names of the funders:
i. In the sentences or phrases identified in task 1, identify the entities that follow the key phrases. These entities are usually the names of the funders. Verify that these are standard funder names as opposed to identifiers or other, extraneous values.
ii. Standardize the names of the funders, if needed, to maintain consistency in naming conventions.

Task 3: Extract the grant numbers:
i. Besides the names of the funders, extract grant numbers. These can often be found in the same sentences as the funders' names.
ii. Extract extract grant numbers along with the names of the funders.

Task 4: Verify the grant numbers:
i. Validate the format of the extracted grant numbers. Grant numbers often have specific formats that resemble unique identifiers, which can involve alphanumeric strings possibly separated by slashes, hyphens, or other special characters.
ii. Check if the identified grant numbers match the expected format for grant identifiers. For example, they may be expected to contain a certain number of digits or a specific pattern of letters and numbers.
iii. If a "grant number" does not fit the typical format, flag it for further review. It may be an incorrect extraction from the text, and not a grant number at all.


Task 5: Structure and pair the funder and grant information:
i. The extracted information should be organized in a structured format, such as a table or a list, and paired with the grant information.

Task 6: Verify the extracted information:
i. Verify that the extracted information is accurate and complete. For example, check if any funders have been missed, if any funders have been duplicated, if any extraneous entities have been incorrectly identified as funders. Remove such entries.
ii. Cross-verify the extracted entities classified as funders. Entities resembling a string of alphanumeric characters potentially separated by special characters (like the identifiers mentioned) are often grant numbers rather than funder names.
ii. Develop a pattern matching method to identify potential misclassifications. For example, grant numbers often start with a digit followed by a letter (e.g., '3P', '5P', '5R') and continue with more alphanumeric characters. If an entity labeled as a funder matches this pattern, flag it for review.
iii. Once flagged entities are reviewed, if they are confirmed to be grant numbers rather than funder names, remove them from the list of funders and add them to the list of grant numbers. If they are indeed funder names, retain them in the funder list.

Task 7: Return the extracting information:
i. Return the list of Funder names, grant numbers and projects funded using the below format:
{"funder": "{Funder Name}", "grants":[{grant},{...}]}\n
{...}

For example, if provided with the text:

"We are grateful to all the participants who took part in this study, the general practitioners, the Scottish School of Primary Care for their help in recruiting the participants, and the whole team, which includes interviewers, computer and laboratory technicians, clerical workers, research scientists, volunteers, managers, receptionists, and nurses. The Wellcome Trust provides support for Wellcome Trust UK type 2 diabetes case-control collection (GoDARTS; 099177/Z/12/12) and informatics support is provided by the Chief Scientist Office. The Wellcome Trust funds the Scottish Health Informatics Programme, provides core support for the Wellcome Trust Centre for Human Genetics in Oxford and funds the Wellcome Trust Case Control Consortium 2 (084726/Z/08/Z). The research leading to these results has received support from the Innovative Medicines Initiative Joint Undertaking under grant agreement number 115006 (IMI-SUMMIT), resources of which are composed of financial contributions from the European Union's Seventh Framework Programme (FP7/2007-2013) and European Federation of Pharmaceutical Industries and Associations companies' kind contributions. PCS is supported by the Hong Kong Research Grants Council General Research Fund project (grants 777511 and 776513), and European Commission Seventh Framework Programme grant for the European Network of National Schizophrenia Networks Studying Gene-Environmental Interactions (EU-GEI). ERP holds a Wellcome Trust New Investigator award. MIMcC holds a National Institute for Health Research Senior Investigator award and a Wellcome Trust Senior Investigator award (098381). This research was specifically funded by the Wellcome Trust (092272/Z/10/Z) for a Henry Wellcome Post-Doctoral Fellowship to KZ"

You would:

Task 1:
i. Key phrases identified from the text: 'provides support', 'funds', 'support is provided by', 'has received support from', 'resources of which are composed of financial contributions from', 'is supported by', 'holds a', 'was specifically funded by'.

ii. Extract sentences or phrases containing key phrases:

	The Wellcome Trust provides support for Wellcome Trust UK type 2 diabetes case-control collection (GoDARTS; 099177/Z/12/12) and informatics support is provided by the Chief Scientist Office.
	The Wellcome Trust funds the Scottish Health Informatics Programme, provides core support for the Wellcome Trust Centre for Human Genetics in Oxford and funds the Wellcome Trust Case Control Consortium 2 (084726/Z/08/Z).
	The research leading to these results has received support from the Innovative Medicines Initiative Joint Undertaking under grant agreement number 115006 (IMI-SUMMIT), resources of which are composed of financial contributions from the European Union's Seventh Framework Programme (FP7/2007-2013) and European Federation of Pharmaceutical Industries and Associations companies' kind contributions.
	ERP holds a Wellcome Trust New Investigator award.
	MIMcC holds a National Institute for Health Research Senior Investigator award and a Wellcome Trust Senior Investigator award (098381).
	This research was specifically funded by the Wellcome Trust (092272/Z/10/Z) for a Henry Wellcome Post-Doctoral Fellowship to KZ.

Task 2:
i. Identify funders from the sentences/phrases:

	The Wellcome Trust
	Chief Scientist Office
	Innovative Medicines Initiative Joint Undertaking
	European Union's Seventh Framework Programme
	European Federation of Pharmaceutical Industries and Associations
	Hong Kong Research Grants Council General Research Fund
	European Commission Seventh Framework Programme
	National Institute for Health Research

Task 3
i. Identify grant numbers:

	Wellcome Trust UK type 2 diabetes case-control collection (GoDARTS; Grant Number: 099177/Z/12/12)
	Wellcome Trust Case Control Consortium 2 (Grant Number: 084726/Z/08/Z)
	Innovative Medicines Initiative Joint Undertaking (Grant Number: 115006, IMI-SUMMIT)
	European Union's Seventh Framework Programme (FP7/2007-2013)
	Hong Kong Research Grants Council General Research Fund (Grant Numbers: 777511, 776513)
	Wellcome Trust Senior Investigator award (MIMcC, Grant Number: 098381)
	Wellcome Trust (Grant Number: 092272/Z/10/Z) for a Henry Wellcome Post-Doctoral Fellowship to KZ

Task 4
i. Verify the grant numbers  
  
	099177/Z/12/12 - Matches expected format (6 digits / Z / 2 digits / 2 digits)
	084726/Z/08/Z - Matches expected format (6 digits / Z / 2 digits / 2 digits)
	115006 - Matches a simple format (6 digits)
	FP7/2007-2013 - Partial match to expected format (Alphabets and digits / 4 digits - 4 digits)
	777511 - Matches a simple format (6 digits)
	776513 - Matches a simple format (6 digits)
	098381 - Matches a simple format (6 digits)
	092272/Z/10/Z - Matches expected format (6 digits / Z / 2 digits / 2 digits)

ii. All identified grant numbers match expected formats for grant numbers, so none need to be flagged for further review in this case. All these numbers resemble unique identifiers and do not appear to be other strings of text from the acknowledgement section.

Task 5:

Structure and pair the funder and grant information:

	Funder: The Wellcome Trust
	Grant Numbers: 099177/Z/12/12, 084726/Z/08/Z, 098381, 092272/Z/10/Z

	Funder: Chief Scientist Office

	Funder: Innovative Medicines Initiative Joint Undertaking
	Grant Number: 115006

	Funder: European Commission Seventh Framework Programme

	Funder: 115006

	Funder: European Federation of Pharmaceutical Industries and Associations

	Funder: Hong Kong Research Grants Council General Research Fund
	Grant Numbers: 777511, 776513

	Funder: European Commission Seventh Framework Programme

	Funder: National Institute for Health Research

Task 5: Verify the extracted information:

i. The extracted information was checked for accuracy and completeness. One instance of the duplicate funder "Funder: European Commission Seventh Framework Programme" was removed from the structured list. "Funder: 115006" was removed as well as it matched a grant ID instead of a funder name.　The resulting structured list is:
	
	Funder: The Wellcome Trust
	Grant Numbers: 099177/Z/12/12, 084726/Z/08/Z, 098381, 092272/Z/10/Z

	Funder: Chief Scientist Office

	Funder: Innovative Medicines Initiative Joint Undertaking
	Grant Number: 115006

	Funder: European Commission Seventh Framework Programme

	Funder: European Federation of Pharmaceutical Industries and Associations

	Funder: Hong Kong Research Grants Council General Research Fund
	Grant Numbers: 777511, 776513

	Funder: National Institute for Health Research

Task 6:

Return the text:

	{"funder": "The Wellcome Trust", "grants: ["099177/Z/12/12", "084726/Z/08/Z", "098381", "092272/Z/10/Z"]
	{"funder:"Chief Scientist Office", grants:[]}
	{"funder:"Innovative Medicines Initiative Joint Undertaking", grants:["115006"]}
	{"funder:"European Union's Seventh Framework Programme", grants:[]}
	{"funder:"European Federation of Pharmaceutical Industries and Associations", grants:[]}
	{"funder:"Hong Kong Research Grants Council General Research Fund", grants:["777511", "776513"]}
	{"funder:"National Institute for Health Research", grants:[]}

Now, extract the funders and grant numbers from the following acknowledgements text. Respond only in the below format. Do not respond with your tasks. Verify that you have responded in the below format:
{"funder": "{Funder Name}", "grants":[{grant},{...}]}\n
{...}\n

Acknowledgements text: 
	"""
	content = prompt + acknowledgements
	response = openai.ChatCompletion.create(
		model="gpt-3.5-turbo-16k",
		messages=[{"role": "user", "content": content}])
	entities = response.choices[0]['message']['content'].split('\n')
	entities = [entity for entity in entities if entity != '']
	return entities