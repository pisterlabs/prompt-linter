
DATA = [
    {
        "method": "GET",
        "url": "/AllergyIntolerance/:id",
        "parameters": None,
        "example_response": "{\n  \"clinicalStatus\": {\n    \"coding\": [\n      {\n        \"code\": \"active\",\n        \"system\": \"http://terminology.hl7.org/CodeSystem/allergyintolerance-clinical\"\n      }",
        "description": "Retrieve details of a specific AllergyIntolerance based on its ID."
    },
    {
        "method": "GET",
        "url": "/AllergyIntolerance?:parameters",
        "parameters": None,
        "example_response": "{\n  \"entry\": [\n    {\n      \"resource\": {\n        \"clinicalStatus\": {\n          \"coding\": [\n            {\n              \"code\": \"active\",\n              \"system\": \"http://terminology.hl7.org/CodeSystem/allergyintolerance-clinical\"\n            }\n          ]\n        },\n        \"code\": {\n          \"coding\": [\n            {\n              \"code\": \"390952000\",\n              \"display\": \"Dust\",\n              \"system\": \"http://snomed.info/sct\"\n            }\n          ]\n        },\n        \"id\": \"207378\",\n        \"patient\": {\n          \"display\": \"Bob Fhir\",\n          \"reference\": \"Patient/7684393\"\n        },\n        \"reaction\": [\n          {\n            \"manifestation\": [\n              {\n                \"coding\": [\n                  {\n                    \"code\": \"139176003\",\n                    \"display\": \"Cough\",\n                    \"system\": \"http://snomed.info/sct\"\n                  }\n                ],\n                \"text\": \"Cough\"\n              }\n            ],\n            \"severity\": \"moderate\"\n          }\n        ],\n        \"resourceType\": \"AllergyIntolerance\",\n        \"verificationStatus\": {\n          \"coding\": [\n            {\n              \"code\": \"unconfirmed\",\n              \"system\": \"http://terminology.hl7.org/CodeSystem/allergyintolerance-verification\"\n            }\n          ]\n        }\n      }\n    }\n  ],\n  \"resourceType\": \"Bundle\",\n  \"type\": \"searchset\"\n}",
        "description": "Retrieve a list of AllergyIntolerance?s based on provided parameters."
    },
    {
        "method": "GET",
        "url": "/CarePlan/:id",
        "parameters": None,
        "example_response": "{\n  \"category\": [\n    {\n      \"coding\": [\n        {\n          \"code\": \"assess-plan\",\n          \"system\": \"http://hl7.org/fhir/us/core/CodeSystem/careplan-category\"\n        }",
        "description": "Retrieve details of a specific CarePlan based on its ID."
    },
    {
        "method": "GET",
        "url": "/CarePlan",
        "parameters": None,
        "example_response": "{\n  \"entry\": [\n    {\n      \"resource\": {\n        \"category\": [\n          {\n            \"coding\": [\n              {\n                \"code\": \"assess-plan\",\n                \"system\": \"http://hl7.org/fhir/us/core/CodeSystem/careplan-category\"\n              }",
        "description": "Retrieve a list of all CarePlans."
    },
    {
        "method": "GET",
        "url": "/CareTeam/:id",
        "parameters": None,
        "example_response": "{\n  \"id\": \"6824072\",\n  \"participant\": [\n        \"resourceType\": \"CarePlan\",\n        \"status\": \"active\",\n        \"subject\": {\n          \"display\": \"Bob Fhir\",\n          \"reference\": \"Patient/7684393\"\n        }",
        "description": "Retrieve details of a specific CareTeam based on its ID."
    },
    {
        "method": "GET",
        "url": "/CareTeam?:parameters",
        "parameters": None,
        "example_response": "{\n  \"id\": \"6824072\",\n  \"participant\": [\n    {\n      \"member\": {\n        \"display\": \"Joe Provider\",\n        \"reference\": \"Practitioner/13142\"\n      },\n      \"role\": [\n        {\n          \"text\": \"Primary Care Provider\"\n        }\n      ]\n    }\n  ],\n  \"resourceType\": \"CareTeam\",\n  \"status\": \"proposed\",\n  \"subject\": {\n    \"display\": \"Bob Hill Fhir\",\n    \"reference\": \"Patient/7684393\"\n  }\n}",
        "description": "Retrieve a list of CareTeam?s based on provided parameters."
    },
    {
        "method": "GET",
        "url": "/Condition/:id",
        "parameters": None,
        "example_response": "{\n  \"category\": [\n    {\n      \"coding\": [\n        {\n          \"code\": \"439401001\",\n          \"display\": \"Diagnosis\",\n          \"system\": \"http://snomed.info/sct\"\n        }",
        "description": "Retrieve details of a specific Condition based on its ID."
    },
    {
        "method": "GET",
        "url": "/Condition?:parameters",
        "parameters": None,
        "example_response": "{\n  \"entry\": [\n    {\n      \"resource\": {\n        \"category\": [\n          {\n            \"coding\": [\n              {\n                \"code\": \"439401001\",\n                \"display\": \"Diagnosis\",\n                \"system\": \"http://snomed.info/sct\"\n              }\n            ]\n          }\n        ],\n        \"clinicalStatus\": {\n          \"coding\": [\n            {\n              \"code\": \"active\",\n              \"system\": \"http://terminology.hl7.org/CodeSystem/condition-clinical\"\n            }\n          ]\n        },\n        \"code\": {\n          \"coding\": [\n            {\n              \"code\": \"22253000\",\n              \"display\": \"Pain\",\n              \"system\": \"http://snomed.info/sct\"\n            }\n          ]\n        },\n        \"id\": \"27263267\",\n        \"resourceType\": \"Condition\",\n        \"subject\": {\n          \"display\": \"Bob Hill Fhir\",\n          \"reference\": \"Patient/7684393\"\n        },\n        \"verificationStatus\": {\n          \"coding\": [\n            {\n              \"code\": \"confirmed\",\n              \"system\": \"http://terminology.hl7.org/CodeSystem/condition-ver-status\"\n            }\n          ]\n        }\n      }\n    }\n  ],\n  \"resourceType\": \"Bundle\",\n  \"type\": \"searchset\"\n}",
        "description": "Retrieve a list of Condition?s based on provided parameters."
    },
    {
        "method": "GET",
        "url": "/Device/:id",
        "parameters": None,
        "example_response": "{\n  \"distinctIdentifier\": \"115\",\n  \"expirationDate\": \"2022-12-29\",\n  \"id\": \"115\",\n  \"lotNumber\": \"123\",\n  \"manufactureDate\": \"2022-12-09\",\n  \"patient\": {\n    \"display\": \"Bob Fhir\",\n    \"reference\": \"Patient/7684393\"\n  },\n  \"resourceType\": \"Device\",\n  \"serialNumber\": \"456\",\n  \"type\": {\n    \"coding\": [\n      {\n        \"code\": \"unknown\",\n        \"display\": \"Thermal cycler nucleic acid amplification analyser IVD, laboratory, automated\"\n      }\n    ]\n  },\n  \"udiCarrier\": [\n    {\n      \"carrierHRF\": \"A91004001\",\n      \"deviceIdentifier\": \"156642\"\n    }\n  ]\n}",
        "description": "Retrieve details of a specific Device based on its ID."
    },
    {
        "method": "GET",
        "url": "/Device/?:parameters",
        "parameters": None,
        "example_response": "{\"entry\":[{\"resource\":{\"distinctIdentifier\":\"115\",\"expirationDate\":\"2022-12-29\",\"id\":\"115\",\"lotNumber\":\"123\",\"manufactureDate\":\"2022-12-09\",\"patient\":{\"display\":\"Bob Fhir\",\"reference\":\"Patient/7684393\"},\"resourceType\":\"Device\",\"serialNumber\":\"456\",\"type\":{\"coding\":[{\"code\":\"unknown\",\"display\":\"Thermal cycler nucleic acid amplification analyser IVD, laboratory, automated\"}]},\"udiCarrier\":[{\"carrierHRF\":\"A91004001\",\"deviceIdentifier\":\"156642\"}]}},\"resourceType\":\"Bundle\",\"type\":\"searchset\"}",
        "description": "Retrieve a list of Devices based on provided parameters."
    },
    {
        "method": "GET",
        "url": "/DiagnosticReport/:id",
        "parameters": None,
        "example_response": "{\n  \"category\": [\n    {\n      \"coding\": [\n        {\n          \"code\": \"LP29684-5\",\n          \"display\": \"Radiology\",\n        \"udiCarrier\": [\n          {\n            \"carrierHRF\": \"A91004001\",\n            \"deviceIdentifier\": \"156642\"\n          }",
        "description": "Retrieve details of a specific DiagnosticReport based on its ID."
    },
    {
        "method": "GET",
        "url": "/DiagnosticReport",
        "parameters": None,
        "example_response": "{\n  \"category\": [\n    {\n      \"coding\": [\n        {\n          \"code\": \"LP29684-5\",\n          \"display\": \"Radiology\",\n          \"system\": \"http://loinc.org\"\n        }\n      ],\n      \"text\": \"Radiology\"\n    }\n  ],\n  \"code\": {\n    \"coding\": [\n      {\n        \"code\": \"30746-2\",\n        \"display\": \"30746-2\",\n        \"system\": \"http://loinc.org\"\n      }\n    ],\n    \"text\": \"30746-2\"\n  },\n  \"effectiveDateTime\": \"2022-12-01T17:00:00+00:00\",\n  \"encounter\": {\n    \"display\": \"32169294\",\n    \"reference\": \"Encounter/32169294\"\n  },\n  \"id\": \"6978492\",\n  \"issued\": \"2023-01-11T18:52:30+00:00\",\n  \"performer\": [\n    {\n      \"reference\": \"Organization/74\"\n    }\n  ],\n  \"presentedForm\": [\n    {\n      \"contentType\": \"application/pdf\",\n      \"data\": \"foo\"\n    }\n  ],\n  \"resourceType\": \"DiagnosticReport\",\n  \"status\": \"unknown\",\n  \"subject\": {\n    \"display\": \"Bob Fhir\",\n    \"reference\": \"Patient/7684393\"\n  }\n}",
        "description": "Retrieve a list of all DiagnosticReports."
    },
    {
        "method": "GET",
        "url": "/DocumentReference/:id",
        "parameters": None,
        "example_response": "{\n  \"author\": [\n    {\n      \"reference\": \"Organization/74\"\n    }\n  ],\n  \"category\": [\n    {\n      \"coding\": [\n        {\n          \"code\": \"clinical-note\",\n          \"display\": \"Clinical Note\",\n          \"system\": \"http://hl7.org/fhir/us/core/CodeSystem/us-core-documentreference-category\"\n        }\n      ],\n      \"text\": \"Clinical Note\"\n    }\n  ],\n  \"content\": [\n    {\n      \"attachment\": {\n        \"contentType\": \"application/pdf\",\n        \"data\": \"foo\",\n        \"url\": \"provider.myhelo.com/api?resource=clio&method=download&arguments=%7B%22key%22%3A%225afd7ecdeefd4f1b96d6ecf5a6ba3f8bcd15573a20e57310b79eb0055ff296b1fb2e475c9acb4b50cf52f92b52af1b8f53fd39905992f62c7542011441c4aa4c%22%2C%22name%22%3A%22file%22%7D\"\n      },\n      \"format\": {\n        \"code\": \"urn:ihe:iti:xds:2017:mimeTypeSufficient\",\n        \"display\": \"mimeType Sufficient\",\n        \"system\": \"urn:oid:1.3.6.1.4.1.19376.1.2.3\"\n      }\n    }\n  ],\n  \"context\": {\n    \"encounter\": [\n      {\n        \"display\": \"32169294\",\n        \"reference\": \"Encounter/32169294\"\n      }\n    ],\n    \"period\": {\n      \"end\": \"2022-12-01T18:00:00+00:00\",\n      \"start\": \"2022-12-01T17:00:00+00:00\"\n    }\n  },\n  \"custodian\": {\n    \"reference\": \"Organization/74\"\n  },\n  \"date\": \"2023-01-05T19:00:50+00:00\",\n  \"id\": \"232969698\",\n  \"identifier\": [\n    {\n      \"system\": \"urn:ietf:rfc:3986\",\n      \"value\": \"urn:oid:2.16.840.1.113883.19.5.99999.1\"\n    }\n  ],\n  \"resourceType\": \"DocumentReference\",\n  \"status\": \"current\",\n  \"subject\": {\n    \"display\": \"Bob Fhir\",\n    \"reference\": \"Patient/7684393\"\n  },\n  \"type\": {\n    \"coding\": [\n      {\n        \"code\": \"18842-5\",\n        \"display\": \"Discharge summary\",\n        \"system\": \"http://loinc.org\"\n      }\n    ],\n    \"text\": \"Discharge summary\"\n  }\n}",
        "description": "Retrieve details of a specific DocumentReference based on its ID."
    },
    {
        "method": "GET",
        "url": "/DocumentReference?:parameters",
        "parameters": None,
        "example_response": "{\n  \"entry\": [\n    {\n      \"resource\": {\n        \"author\": [\n          {\n            \"reference\": \"Organization/74\"\n          }\n        ],\n        \"category\": [\n          {\n            \"coding\": [\n              {\n                \"code\": \"clinical-note\",\n                \"display\": \"Clinical Note\",\n                \"system\": \"http://hl7.org/fhir/us/core/CodeSystem/us-core-documentreference-category\"\n              }\n            ],\n            \"text\": \"Clinical Note\"\n          }\n        ],\n        \"content\": [\n          {\n            \"attachment\": {\n              \"contentType\": \"application/pdf\",\n              \"data\": \"foo\",\n              \"url\": \"provider.myhelo.com/api?resource=clio&method=download&arguments=%7B%22key%22%3A%225afd7ecdeefd4f1b96d6ecf5a6ba3f8bcd15573a20e57310b79eb0055ff296b1fb2e475c9acb4b50cf52f92b52af1b8f53fd39905992f62c7542011441c4aa4c%22%2C%22name%22%3A%22file%22%7D\"\n            },\n            \"format\": {\n              \"code\": \"urn:ihe:iti:xds:2017:mimeTypeSufficient\",\n              \"display\": \"mimeType Sufficient\",\n              \"system\": \"urn:oid:1.3.6.1.4.1.19376.1.2.3\"\n            }\n          }\n        ],\n        \"context\": {\n          \"encounter\": [\n            {\n              \"display\": \"32169294\",\n              \"reference\": \"Encounter/32169294\"\n            }\n          ],\n          \"period\": {\n            \"end\": \"2022-12-01T18:00:00+00:00\",\n            \"start\": \"2022-12-01T17:00:00+00:00\"\n          }\n        },\n        \"custodian\": {\n          \"reference\": \"Organization/74\"\n        },\n        \"date\": \"2023-01-05T19:00:50+00:00\",\n        \"id\": \"232969698\",\n        \"identifier\": [\n          {\n            \"system\": \"urn:ietf:rfc:3986\",\n            \"value\": \"urn:oid:2.16.840.1.113883.19.5.99999.1\"\n          }\n        ],\n        \"resourceType\": \"DocumentReference\",\n        \"status\": \"current\",\n        \"subject\": {\n          \"display\": \"Bob Fhir\",\n          \"reference\": \"Patient/7684393\"\n        },\n        \"type\": {\n          \"coding\": [\n            {\n              \"code\": \"18842-5\",\n              \"display\": \"Discharge summary\",\n              \"system\": \"http://loinc.org\"\n            }\n          ],\n          \"text\": \"Discharge summary\"\n        }\n      }\n    }\n  ],\n  \"resourceType\": \"Bundle\",\n  \"type\": \"searchset\"\n}",
        "description": "Retrieve a list of DocumentReference?s based on provided parameters."
    },
    {
        "method": "GET",
        "url": "/Encounter/:id",
        "parameters": None,
        "example_response": "{\n  \"class\": {\n    \"code\": \"1\",\n    \"display\": \"Medical Care\",\n    \"system\": \"http://terminology.hl7.org/CodeSystem/v3-ActCode\"\n  },\n  \"hospitalization\": {\n    \"dischargeDisposition\": {\n      \"coding\": [\n        {\n          \"code\": \"100\",\n          \"display\": \"Discharged for Other Reasons\",\n          \"system\": \"http://www.nubc.org/patient-discharge\"\n        }\n      ]\n    }\n  },\n  \"id\": \"32169294\",\n  \"identifier\": [\n    {\n      \"system\": \"https://provider.myhelo.com\",\n      \"value\": \"32169294\"\n    }\n  ],\n  \"location\": [\n    {\n      \"location\": {\n        \"display\": \"Evan Facility\",\n        \"reference\": \"Location/1766\"\n      }\n    }\n  ],\n  \"participant\": [\n    {\n      \"individual\": {\n        \"reference\": \"Practitioner/13142\"\n      },\n      \"period\": {\n        \"end\": \"2022-12-01T18:00:00+00:00\",\n        \"start\": \"2022-12-01T17:00:00+00:00\"\n      },\n      \"type\": [\n        {\n          \"coding\": [\n            {\n              \"code\": \"PART\",\n              \"display\": \"Participation\",\n              \"system\": \"http://terminology.hl7.org/CodeSystem/v3-ParticipationType\"\n            }\n          ],\n          \"text\": \"Participation\"\n        }\n      ]\n    }\n  ],\n  \"period\": {\n    \"end\": \"2022-12-01T18:00:00+00:00\",\n    \"start\": \"2022-12-01T17:00:00+00:00\"\n  },\n  \"reasonCode\": [\n    {\n      \"coding\": [\n        {\n          \"code\": \"22253000\",\n          \"display\": \"Pain\",\n          \"system\": \"http://snomed.info/sct\"\n        }\n      ]\n    }\n  ],\n  \"reasonReference\": [\n    {\n      \"reference\": \"Condition/27263267\"\n    }\n  ],\n  \"resourceType\": \"Encounter\",\n  \"serviceProvider\": {\n    \"reference\": \"Organization/74\"\n  },\n  \"status\": \"planned\",\n  \"subject\": {\n    \"display\": \"Bob Fhir\",\n    \"reference\": \"Patient/7684393\"\n  },\n  \"type\": [\n    {\n      \"coding\": [\n        {\n          \"code\": \"76499\",\n          \"display\": \"Radiographic Procedure\",\n          \"system\": \"http://www.ama-assn.org/go/cpt\"\n        }\n      ],\n      \"text\": \"Radiographic Procedure\"\n    }\n  ]\n}",
        "description": "Retrieve details of a specific Encounter based on its ID."
    },
    {
        "method": "GET",
        "url": "/Encounter?:parameters",
        "parameters": None,
        "example_response": "{ \"entry\": [ { \"resource\": { \"class\": { \"code\": \"1\", \"display\": \"Medical Care\", \"system\": \"http://terminology.hl7.org/CodeSystem/v3-ActCode\" }, \"hospitalization\": { \"dischargeDisposition\": { \"coding\": [ { \"code\": \"100\", \"display\": \"Discharged for Other Reasons\", \"system\": \"http://www.nubc.org/patient-discharge\" } ] } }, \"id\": \"32169294\", \"identifier\": [ { \"system\": \"https://provider.myhelo.com\", \"value\": \"32169294\" } ], \"location\": [ { \"location\": { \"display\": \"Evan Facility\", \"reference\": \"Location/1766\" } } ], \"participant\": [ { \"individual\": { \"reference\": \"Practitioner/13142\" }, \"period\": { \"end\": \"2022-12-01T18:00:00+00:00\", \"start\": \"2022-12-01T17:00:00+00:00\" }, \"type\": [ { \"coding\": [ { \"code\": \"PART\", \"display\": \"Participation\", \"system\": \"http://terminology.hl7.org/CodeSystem/v3-ParticipationType\" } ], \"text\": \"Participation\" } ] } ], \"period\": { \"end\": \"2022-12-01T18:00:00+00:00\", \"start\": \"2022-12-01T17:00:00+00:00\" }, \"reasonCode\": [ { \"coding\": [ { \"code\": \"22253000\", \"display\": \"Pain\", \"system\": \"http://snomed.info/sct\" } ] } ], \"reasonReference\": [ { \"reference\": \"Condition/27263267\" } ], \"resourceType\": \"Encounter\", \"serviceProvider\": { \"reference\": \"Organization/74\" }, \"status\": \"planned\", \"subject\": { \"display\": \"Bob Fhir\", \"reference\": \"Patient/7684393\" }, \"type\": [ { \"coding\": [ { \"code\": \"76499\", \"display\": \"Radiographic Procedure\", \"system\": \"http://www.ama-assn.org/go/cpt\" } ], \"text\": \"Radiographic Procedure\" } ] } } ], \"resourceType\": \"Bundle\", \"type\": \"searchset\" }",
        "description": "Retrieve a list of Encounter?s based on provided parameters."
    },
    {
        "method": "GET",
        "url": "/Goal/:id",
        "parameters": None,
        "example_response": "{ \"description\": { \"text\": \"ASES Shoulder Assessment\" }, \"id\": \"43\", \"lifecycleStatus\": \"active\", \"resourceType\": \"Goal\", \"subject\": { \"display\": \"Bob Fhir\", \"reference\": \"Patient/7684393\" }, \"target\": [ { \"dueDate\": \"2022-12-28\" } ] }",
        "description": "Retrieve details of a specific Goal based on its ID."
    },
    {
        "method": "GET",
        "url": "/Goal?:parameters",
        "parameters": None,
        "example_response": "{ \"entry\": [ { \"resource\": { \"description\": { \"text\": \"ASES Shoulder Assessment\" }, \"id\": \"43\", \"lifecycleStatus\": \"active\", \"resourceType\": \"Goal\", \"subject\": { \"display\": \"Bob Fhir\", \"reference\": \"Patient/7684393\" }, \"target\": [ { \"dueDate\": \"2022-12-28\" } ] } } ], \"resourceType\": \"Bundle\", \"type\": \"searchset\" }",
        "description": "Retrieve a list of Goal?s based on provided parameters."
    },
    {
        "method": "GET",
        "url": "/Group/:id/$export",
        "parameters": None,
        "example_response": "undefined",
        "description": "Retrieve details of a specific Group based on its ID."
    },
    {
        "method": "GET",
        "url": "/Immunization/:id",
        "parameters": None,
        "example_response": "{ \"id\": \"25189\", \"occurrenceDateTime\": \"2022-12-27T00:00:00+00:00\", \"patient\": { \"reference\": \"Patient/7684393\" }, \"primarySource\": false, \"resourceType\": \"Immunization\", \"status\": \"completed\", \"statusReason\": { \"coding\": [ { \"code\": \"IMMUNE\", \"display\": \"immunity\", \"system\": \"http://hl7.org/fhir/v3/ActReason\" } ] }, \"vaccineCode\": { \"coding\": [ { \"code\": \"168\", \"system\": \"urn:oid:2.16.840.1.113883.12.292\" } ], \"text\": \"Fluad\" } }",
        "description": "Retrieve details of a specific Immunization based on its ID."
    },
    {
        "method": "GET",
        "url": "/Immunization?:parameters",
        "parameters": None,
        "example_response": "{ \"entry\": [ { \"resource\": { \"id\": \"25189\", \"occurrenceDateTime\": \"2022-12-27T00:00:00+00:00\", \"patient\": { \"reference\": \"Patient/7684393\" }, \"primarySource\": false, \"resourceType\": \"Immunization\", \"status\": \"completed\", \"statusReason\": { \"coding\": [ { \"code\": \"IMMUNE\", \"display\": \"immunity\", \"system\": \"http://hl7.org/fhir/v3/ActReason\" } ] }, \"vaccineCode\": { \"coding\": [ { \"code\": \"168\", \"system\": \"urn:oid:2.16.840.1.113883.12.292\" } ], \"text\": \"Fluad\" } } }, { \"resource\": { \"id\": \"25196\", \"occurrenceDateTime\": \"2022-12-30T00:00:00+00:00\", \"patient\": { \"reference\": \"Patient/7684393\" }, \"primarySource\": true, \"resourceType\": \"Immunization\", \"status\": \"not-done\", \"statusReason\": { \"coding\": [ { \"code\": \"PATOBJ\", \"display\": \"patient objection\", \"system\": \"http://hl7.org/fhir/v3/ActReason\" } ] }, \"vaccineCode\": { \"coding\": [ { \"code\": \"03\", \"system\": \"urn:oid:2.16.840.1.113883.12.292\" } ], \"text\": \"MMR\" } } } ], \"resourceType\": \"Bundle\", \"type\": \"searchset\" }",
        "description": "Retrieve a list of Immunization?s based on provided parameters."
    },
    {
        "method": "GET",
        "url": "/Location/:id",
        "parameters": None,
        "example_response": "{ \"id\": \"1766\", \"mode\": \"instance\", \"resourceType\": \"Location\", \"status\": \"active\" }",
        "description": "Retrieve details of a specific Location based on its ID."
    },
    {
        "method": "GET",
        "url": "/MedicationRequest/:id",
        "parameters": None,
        "example_response": "{\n  \"authoredOn\": \"2021-06-16\",\n  \"dosageInstruction\": [\n    {\n      \"text\": \"5 Per hour\"\n    }",
        "description": "Retrieve details of a specific MedicationRequest based on its ID."
    },
    {
        "method": "GET",
        "url": "/MedicationRequest?:parameters",
        "parameters": None,
        "example_response": None,
        "description": "Retrieve a list of MedicationRequest?s based on provided parameters."
    },
    {
        "method": "GET",
        "url": "/Observation/:id",
        "parameters": None,
        "example_response": None,
        "description": "Retrieve details of a specific Observation based on its ID."
    },
    {
        "method": "GET",
        "url": "/MedicationRequest/:id",
        "parameters": None,
        "example_response": "{\n  \"authoredOn\": \"2021-06-16\",\n  \"dosageInstruction\": [\n    {\n      \"text\": \"5 Per hour\"\n    }\n  ],\n  \"id\": \"44534646\",\n  \"intent\": \"order\",\n  \"medicationCodeableConcept\": {\n    \"coding\": [\n      {\n        \"code\": \"unknown\",\n        \"display\": \"Unknown\",\n        \"system\": \"http://terminology.hl7.org/CodeSystem/data-absent-reason\"\n      }\n    ],\n    \"text\": \"Unknown\"\n  },\n  \"reportedBoolean\": false,\n  \"requester\": {\n    \"reference\": \"Organization/74\"\n  },\n  \"resourceType\": \"MedicationRequest\",\n  \"status\": \"active\",\n  \"subject\": {\n    \"display\": \"Evan Tester\",\n    \"reference\": \"Patient/2227174\"\n  }\n}",
        "description": "Retrieve details of a specific MedicationRequest based on its ID."
    },
    {
        "method": "GET",
        "url": "/MedicationRequest?:parameters",
        "parameters": "Name: _id, Type: token, Required: false;\nName: _revinclude, Type: token, Required: false;\nName: intent, Type: token, Required: false;\nName: patient, Type: reference, Required: true;\nName: status, Type: token, Required: false;",
        "example_response": "{\n  \"entry\": [\n    {\n      \"resource\": {\n        \"authoredOn\": \"2021-06-16\",\n        \"dosageInstruction\": [\n          {\n            \"text\": \"5 Per hour\"\n          }\n        ],\n        \"id\": \"44534646\",\n        \"intent\": \"order\",\n        \"medicationCodeableConcept\": {\n          \"coding\": [\n            {\n              \"code\": \"unknown\",\n              \"display\": \"Unknown\",\n              \"system\": \"http://terminology.hl7.org/CodeSystem/data-absent-reason\"\n            }\n          ],\n          \"text\": \"Unknown\"\n        },\n        \"reportedBoolean\": false,\n        \"requester\": {\n          \"reference\": \"Organization/74\"\n        },\n        \"resourceType\": \"MedicationRequest\",\n        \"status\": \"active\",\n        \"subject\": {\n          \"display\": \"Evan Tester\",\n          \"reference\": \"Patient/2227174\"\n        }\n      }\n    }\n  ],\n  \"resourceType\": \"Bundle\",\n  \"type\": \"searchset\"\n}",
        "description": "Retrieve a list of MedicationRequest?s based on provided parameters."
    },
    {
        "method": "GET",
        "url": "/Observation/:id",
        "parameters": None,
        "example_response": "{\n  \"category\": [\n    {\n      \"coding\": [\n        {\n          \"code\": \"social-history\",\n          \"display\": \"Social History\",\n          \"system\": \"http://terminology.hl7.org/CodeSystem/observation-category\"\n        }\n      ]\n    }\n  ],\n  \"code\": {\n    \"coding\": [\n      {\n        \"code\": \"72166-2\",\n        \"display\": \"Tobbaco smoking status\",\n        \"system\": \"http://loinc.org\"\n      }\n    ]\n  },\n  \"id\": \"6262602\",\n  \"issued\": \"2022-12-29T13:46:06+00:00\",\n  \"resourceType\": \"Observation\",\n  \"status\": \"final\",\n  \"subject\": {\n    \"display\": \"Bob Hill Fhir\",\n    \"reference\": \"Patient/7684393\"\n  },\n  \"valueCodeableConcept\": {\n    \"text\": \"Smokes Cigarettes 5 per week\"\n  }\n}",
        "description": "Retrieve details of a specific Observation based on its ID."
    },
    {
        "method": "GET",
        "url": "/Practitioner/:id",
        "parameters": None,
        "example_response": "{\n  \"address\": [\n    {\n      \"city\": \"INDIANAPOLIS\",\n      \"line\": [\n        \"8450 NORTHWEST BLVD\"\n      ],\n      \"postalCode\": \"462781381\",\n      \"state\": \"IN\",\n      \"use\": \"work\"\n    }",
        "description": "Retrieve details of a specific Practitioner based on its ID."
    },
    {
        "method": "GET",
        "url": "/Practitioner?:parameters",
        "parameters": None,
        "example_response": None,
        "description": "Retrieve a list of Practitioner?s based on provided parameters."
    },
    {
        "method": "GET",
        "url": "/Procedure/:id",
        "parameters": None,
        "example_response": None,
        "description": "Retrieve details of a specific Procedure based on its ID."
    },
    {
        "method": "GET",
        "url": "/Procedure?:parameters",
        "parameters": None,
        "example_response": None,
        "description": "Retrieve a list of Procedure?s based on provided parameters."
    },
    {
        "method": "GET",
        "url": "/Provenance/:id",
        "parameters": None,
        "example_response": None,
        "description": "Retrieve details of a specific Provenance based on its ID."
    }
]

import requests
import json
import streamlit as st
import pandas as pd
import openai
import numpy as np
import faiss
import re

from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

from dotenv import load_dotenv
import os

stop_words = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", 
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", 
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", 
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that", 
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", 
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", 
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", 
    "at", "by", "for", "with", "about", "against", "between", "into", "through", 
    "during", "before", "after", "above", "below", "to", "from", "up", "down", 
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once"
])


load_dotenv()

openai_api_key = os.getenv("openai")
openai.api_key = openai_api_key

def basic_lemmatizer(word):
    endings = ['ing', 'ed', 'es', 's']
    for ending in endings:
        if word.endswith(ending):
            return word[:-len(ending)]
    return word

def preprocess_url(url):

    url = re.sub(r'https?://', '', url)
    tokens = re.split(r'[:/?&=]', url)
    tokens = [token for token in tokens if not token.startswith(':')]
   
    return ' '.join(tokens)

def preprocess_description(description):
    if description is None:
        return ""
    
    description = description.lower()
    description = re.sub(r'[^\w\s]', '', description)
    description = " ".join([basic_lemmatizer(word) for word in description.split() if word not in stop_words])
    
    return description

def preprocess(text, text_type='description'):
    if text_type == 'url':
        return preprocess_url(text)
    else:
        return preprocess_description(text)

# Buggy on streamlit  
# file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'myHelo_API.json')
# with open(file_path) as f:
#     DATA = json.load(f)


def get_embeddings(texts):
    # Preprocess the texts
    # texts = [preprocess(text) for text in texts]
    # Get the embeddings from OpenAI
    embeddings = []
    for text in texts:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        embeddings.append(response['data'][0]['embedding'])
    return np.array(embeddings)

def generate_response(api_details):
    summary = "I found the following API details:\n"
    for detail in api_details:
        summary += f"Method: {detail['Method']}, URL: {detail['URL']}, Parameters: {detail['Parameters']}, Description: {detail['Description']}\n"
    
    # Generate a more coherent and user-friendly message with OpenAI
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages= [
            {"role": "system", "content": "You are a helpful assistant that provides really short information about myhELO API."},
            {"role": "user", "content": summary}
        ]
    )
    
    return response.choices[0].message.content

def getResponse(question):
    # Define the number of similar events to retrieve
    k = 3

    # Load the data
    data = pd.DataFrame(DATA)
    data = data[["method", "url", "parameters", "example_response", "description"]]
    data = data.reset_index(drop=True)

    urls = [preprocess(url, text_type='url') for url in data["url"].tolist()]
    descriptions = [preprocess(desc, text_type='description') for desc in data["description"].tolist()]

    print("Data processed")
    
    # Get the embeddings for the descriptions and urls
    description_embeddings = get_embeddings(urls)
    url_embeddings = get_embeddings(descriptions)

    print("Embeddings generated")
    
    # Concatenate the description and url embeddings
    weight_for_url = 3  # for example
    weighted_url_embeddings = weight_for_url * url_embeddings
    concatenated_embeddings = np.concatenate((weighted_url_embeddings, description_embeddings), axis=1)

    print("Embeddings concatenated")

    # concatenated_embeddings = np.concatenate((description_embeddings, url_embeddings), axis=1)
    
    # Index using FAISS
    index = faiss.IndexFlatL2(concatenated_embeddings.shape[1])
    index.add(concatenated_embeddings)

    print("Index created")
    
    # Preprocess the question and get its embedding
    # query_embedding = get_embeddings([question])
    # query_embedding = np.repeat(query_embedding, 2, axis=1)
    query_embedding = get_embeddings([question])

    print("Query embedding generated")
    weighted_query_embedding = np.concatenate((weight_for_url * query_embedding, query_embedding), axis=1)

    print("Query embedding concatenated")

    # Search for the most similar events to the query
    D, I = index.search(weighted_query_embedding, k)

    print("Search completed")
    api_details = []
    for idx in I[0][:4]:
        row = data.iloc[idx]
        method = row['method']
        url = row['url']
        parameters = row['parameters'] if row['parameters'] else "None"
        if row['example_response']:
            example_response = row['example_response']
        else:
            example_response = "None"
        # example_response = row['example_response'][:50] + "..." if len(row['example_response']) > 50 else row['example_response']
        description = row['description']

        response = generate_response(api_details)

        print("Response generated")

        api_details.append({
            "Method": method,
            "URL": url,
            "Parameters": parameters,
            "Example Response": example_response,
            "Description": description
        })

        print("Done!")

    return question, api_details, response

    
    # Formatting the output
    # content = f"**Question:** {question}\n\n---\n\n### Here are the top most similar API calls to your question:\n"
    
    # for idx in I[0][:4]:
    #     row = data.iloc[idx]
    #     method = row['method']
    #     url = row['url']
    #     parameters = row['parameters'] if row['parameters'] else "None"
    #     example_response = row['example_response'][:50] + "..." if row['example_response'] else "None"  # Displaying only the first 50 characters for brevity
    #     content += f"- **Method:** {method}\n  **URL:** {url}\n  **Parameters:** {parameters}\n  **Example Response:** {example_response}\n\n"
    
    # return content


        





if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("user"):

    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    question, api_details, response = getResponse(prompt)

    with st.chat_message("assistant"):
        # st.subheader(f"Question: {question}")
        # st.write("---")
        st.markdown(response)
        st.subheader("Top similar API calls to your question:")
        for detail in api_details:
            st.text(f"Method: {detail['Method']}")
            st.text(f"URL: {detail['URL']}")
            st.text(f"Parameters: {detail['Parameters']}")
            st.text(f"Description: {detail['Description']}")
            
            with st.expander("Click to view example response"):
                st.text(detail['Example Response'])
                
            st.write("---")

    st.session_state.messages.append({"role": "assistant", "content": "Displayed top similar API calls."})

with st.sidebar:

    st.markdown("""
# MVP showcasing implementation of Chatbot for myhELO API
### Not made for public use    
#### Here you can see API Docs            
                """)
        
    st.json(DATA)


st.markdown("""
## Ask a question about api call
                """)