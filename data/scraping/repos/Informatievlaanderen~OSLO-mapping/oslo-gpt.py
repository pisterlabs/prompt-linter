import os
import openai

openai.api_key = os.getenv(
    "")

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Transformeer de onderstaande JSON-bestanden naar OSLO-compatibele JSON-LD-bestanden. Gebruik de bestaande OSLO SHACL-shapes om de context van het JSON-LD-bestand op te bouwen en de data te converteren naar OSLO-conforme klassen en subklassen.

JSON bestand 1:

"1-Aalst Station": {
"Locatie": {
"CRS": "EPSG:31370",
"Geometrie": "POLYGON ((126966 181444))"
},
"Attributen": {
"ID": "1",
"Naam": "Aalst Station",
"Provincie": "Oost-Vlaanderen",
"VVR": "Aalst",
"Gemeente": "Aalst",
"Status": "Goedgekeurd",
"Categorie BVR": "Interregionaal",
"Beheer": ["Lokaal bestuur"],
"Mobiliteitsaanbod": {
"Deelsystemen": [
{
"Naam": "VOM deelmobiliteit",
"Aanbod": ["Deelwagens, Deelfietsen"]
}
],
"Lijnbus halte": "true",
"Tramhalte": "true",
"Kernnet": "true",
"Aanvullend net": "true",
"Functioneel net": "true",
"Metrohalte": "true",
"Treinstation": "true",
"VOM flex halte": "true",
"Deelwagen VOM": "true",
"Deelfiets VOM": "true",
"Fietslockers": "true",
"Deelsteps": "true",
"Fietspomp": "true",
"Park and ride": "true",
"Kiss and ride": "false"
},
"Aanvullend aanbod": {
"Wachtaccomodatie": "true",
"Vuilnisbak": "true",
"Pakketautomaat": "true",
"Bagagelocker": "true",
"Fietshersteldienst": "true",
"Sanitair": "false",
"Rode brievenbus": "true",
"AED": "false",
"Geld automaat": "true",
"Wifi": "false",
"Oplaadpunt smartphones": "true",
"Drinkerwatervoorziening": "true",
"Vergaderruimte": "true",
"Voedings-en krantenwinkel": "true",
"Eet- en drankgelegenheid": "true",
"Uitleenpunt kinderwagens": "true",
"Spin-off centrumdiensten": "true"
}
}
}

output json-ld bestand:

  {
"@context": [
    "https://data.vlaanderen.be/doc/applicatieprofiel/mobiliteit/vervoersknooppunten/erkendestandaard/2022-12-01/context/OSLO-Vervoersknooppunten-ap.jsonld",
    "https://data.vlaanderen.be/doc/applicatieprofiel/infrastructuurelementen/erkendestandaard/2021-09-30/context/infrastructuurelementen-ap.jsonld",
    "https://data.vlaanderen.be/doc/applicatieprofiel/generiek-basis/zonderstatus/2019-07-01/context/generiek-basis.jsonld",
    "https://raw.githubusercontent.com/GeertThijs/MyFiles/master/ContextfileOrganisatie.jsonld",
    {
      "dcterms": "http://purl.org/dc/terms/",
      "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
      "geosparql": "http://www.opengis.net/ont/geosparql#",
      "NetwerkElement.netwerk": {
        "@reverse": "https://data.vlaanderen.be/ns/netwerk#bestaatUit"
      },
      "cl-idt": "https://example.com/concept/identificatortype/",
      "cl-plt": "https://example.com/concept/plaatstype/",
      "cl-hcl": "https://example.com/concept/hoppinpuntclassificatie/",
      "cl-hst": "https://example.com/concept/hoppinpuntstatus/",
      "cl-mdt": "https://example.com/concept/mobiliteitsdiensttype/",
      "cl-tkt": "https://example.com/concept/transportknooptype/",
      "cl-trt": "https://example.com/concept/transporttype/",
      "cl-avd": "https://example.com/concept/aanvullendedienst/",
      "cl-ine": "https://example.com/concept/infrastructuurelement/"
    }
  ],
"@graph": [
{
"@id": "_:vkn001",
"@type": "Hoppinpunt",
"GeregistreerdVervoersknooppunt.registratie": {
"@type": "Identificator",
"Identificator.identificator": {
"@value": "1",
"@type": "cl-idt:hoppinpunt"
}
},
"Vervoersknooppunt.locatie": [
{
"@type": "Plaats",
"Plaats.plaatsnaam": {
"@value": "Aalst",
"@language": "nl"
},
"dcterms:type": "cl-plt:vervoersregio"
},
{
"@type": "Plaats",
"Plaats.plaatsnaam": {
"@value": "Aalst",
"@language": "nl"
},
"dcterms:type": "cl-plt:gemeente"
},
{
"@type": "Punt",
"Geometrie.gml": {
"@value": "<gml:Point srsName=\"http:\\//www.opengis.net/def/crs/EPSG/0/31370\"><gml:coordinates>129234,71 181910,43</gml:coordinates><gml:Point>",
"@type": "geosparql:gmlliteral"
}
},
{
"@type": "Polygon",
"Geometrie.gml": {
"@value": "<gml:Polygon srsName=\"http:\\//www.opengis.net/def/crs/EPSG/0/31370\"><gml:coordinates>126966 181444,126958.387953251 181405.731656763,126936.710678119 181373.289321881,126904.268343237 181351.612046749,126866 181344,126827.731656763 181351.612046749,126795.289321881 181373.289321881,126773.612046749 181405.731656763,126766 181444,126773.612046749 181482.268343236,126795.289321881 181514.710678119,126827.731656763 181536.387953251,126866 181544,126904.268343237 181536.387953251,126936.710678119 181514.710678119,126958.387953251 181482.268343236,126966 181444</gml:coordinates><gml:Polygon>",
"@type": "geosparql:gmlliteral"
}
}

],
"Vervoersknooppunt.naam": {
"@value": "Station Aalst",
"@language": "nl"
},
"Vervoersknooppunt.classificatie": "cl-hcl:interregionaal",
"Vervoersknooppunt.status": [
"cl-hst:uniekeverantwoordingsnota",
"cl-hst:lopende"
],
"Vervoersknooppunt.dienst": [
"_:av001"
],
"Vervoersknooppunt.dienst": [
{
"Mobiliteitsdienst.naam": {
"@value": "Lijnbushalte",
"@language": "nl"
},
"Mobiliteitsdienst.type": "cl-mdt:lijnbushalte",
"Mobiliteitsdienst.aanbieder": {
"@type": "Aanbieder",
"Aanbieder.registratie": {
"@type": "Identificator",
"Identificator.identificator": {
"@value": "0242.069.537",
"@type": "cl-idt:kbonummer"
}
},
"Aanbieder.voorkeursnaam": {
"@value": "De Lijn",
"@language": "nl"
}
}
},
{
"Mobiliteitsdienst.naam": {
"@value": "Tramhalte",
"@language": "nl"
},
"Mobiliteitsdienst.type": "cl-mdt:tramhalte",
"Mobiliteitsdienst.aanbieder": {
"@type": "Aanbieder",
"Aanbieder.registratie": {
"@type": "Identificator",
"Identificator.identificator": {
"@value": "0242.069.537",
"@type": "cl-idt:kbonummer"
}
},
"Aanbieder.voorkeursnaam": {
"@value": "De Lijn",
"@language": "nl"
}
}
},

{
"Mobiliteitsdienst.naam": {
"@value": "Metro",
"@language": "nl"
},
"Mobiliteitsdienst.type": "cl-mdt:metrohalte",
"Mobiliteitsdienst.aanbieder": {
"@type": "Aanbieder",
"Aanbieder.registratie": {
"@type": "Identificator",
"Identificator.identificator": {
"@value": "0247.499.953",
"@type": "cl-idt:kbonummer"
}
},
"Aanbieder.voorkeursnaam": {
"@value": "MIVB",
"@language": "nl"
}
}
},

{
"Mobiliteitsdienst.naam": {
"@value": "Deelwagen",
"@language": "nl"
},
"Mobiliteitsdienst.type": "cl-mdt:deelwagen",
"Mobiliteitsdienst.aanbieder": {
"@type": "Aanbieder",
"Aanbieder.registratie": {
"@type": "Identificator",
"Identificator.identificator": {
"@value": "0479.561.664",
"@type": "cl-idt:kbonummer"
}
},
"Aanbieder.voorkeursnaam": {
"@value": "Cambio",
"@language": "nl"
}
},
"Mobiliteitsdienst.uitgevoerdMet": {
"@type": "Uitvoerder",
"Uitvoerder.contactinfo": {
"@type": "Contactinfo",
"Contactinfo.adres": {
"@type": "Adresvoorstelling",
"gemeentenaam": {
"@value": "Gent",
"@language": "nl"
},
"straatnaam": {
"@value": "Koningin Maria Hendrikaplein",
"@language": "nl"
},
"huisnummer": "65",
"busnummer": "b",
"postcode": "9000"

},
"Contactinfo.telefoon": "093884565"

},
"Uitvoerder.registratie": {
"@type": "Identificator",
"Identificator.identificator": {
"@value": "0479.561.664",
"@type": "cl-idt:kbonummer"
}
},
"Uitvoerder.voorkeursnaam": {
"@value": "Cambio Vlaanderen",
"@language": "nl"
}
}
},
{
"Mobiliteitsdienst.naam": {
"@value": "Deelfiets",
"@language": "nl"
},
"Mobiliteitsdienst.type": "cl-mdt:deelfiets",
"Mobiliteitsdienst.aanbieder": {
"@type": "Aanbieder",
"Aanbieder.registratie": {
"@type": "Identificator",
"Identificator.identificator": {
"@value": "0656.746.814",
"@type": "cl-idt:kbonummer"
}
},
"Aanbieder.voorkeursnaam": {
"@value": "Blue bike",
"@language": "nl"
},
"Mobiliteitsdienst.uitgevoerdMet": {
"@type": "Uitvoerder",
"Uitvoerder.contactinfo": {
"@type": "Contactinfo",
"Contactinfo.telefoon": "0000000000"

},
"Uitvoerder.registratie": {
"@type": "Identificator",
"Identificator.identificator": {
"@value": "0656.746.814",
"@type": "cl-idt:kbonummer"
}
},
"Uitvoerder.voorkeursnaam": {
"@value": "Blue bike",
"@language": "nl"
}
}}},
{
"Mobiliteitsdienst.naam": {
"@value": "Deelwagenvervoeropmaat",
"@language": "nl"
},
"Mobiliteitsdienst.type": "cl-mdt:deelwagenvervoeropmaat",
"Mobiliteitsdienst.aanbieder": {
"@type": "Aanbieder",
"Aanbieder.registratie": {
"@type": "Identificator",
"Identificator.identificator": {
"@value": "0000.000.000",
"@type": "cl-idt:kbonummer"
}
},
"Aanbieder.voorkeursnaam": {
"@value": "deelwagenvervoeropmaat",
"@language": "nl"
},
"Mobiliteitsdienst.uitgevoerdMet": {
"@type": "Uitvoerder",
"Uitvoerder.contactinfo": {
"@type": "Contactinfo",
"Contactinfo.telefoon": "0000000000"

},
"Uitvoerder.registratie": {
"@type": "Identificator",
"Identificator.identificator": {
"@value": "0000.000.000",
"@type": "cl-idt:kbonummer"
}
},
"Uitvoerder.voorkeursnaam": {
"@value": "deelwagenvervoeropmaat",
"@language": "nl"
}
}}},
{
"Mobiliteitsdienst.naam": "Treinvervoer",
"Mobiliteitsdienst.type": "cl-mdt:treinvervoer",
"Mobiliteitsdienst.aanbieder": {
"@type": "Aanbieder",
"Aanbieder.registratie": {
"@type": "Identificator",
"Identificator.identificator": {
"@value": "0203.430.576",
"@type": "cl-idt:kbonummer"
}
},
"Aanbieder.voorkeursnaam": {
"@value": "NMBS",
"@language": "nl"
}
},
"Mobiliteitsdienst.uitgevoerdMet": {
"@type": "Uitvoerder",
"Uitvoerder.contactinfo": {
"@type": "Contactinfo",
"Contactinfo.telefoon": "0000000000"

},
"Uitvoerder.registratie": {
"@type": "Identificator",
"Identificator.identificator": {
"@value": "0203.430.576",
"@type": "cl-idt:kbonummer"
}
},
"Uitvoerder.voorkeursnaam": {
"@value": "NMBS",
"@language": "nl"
}
}
},

{
"Mobiliteitsdienst.naam": "Vervoeropmaatvast",
"Mobiliteitsdienst.type": "cl-mdt:vervoeropmaatvast",
"Mobiliteitsdienst.aanbieder":  {
"@type": "Aanbieder",
"Aanbieder.registratie": {
"@type": "Identificator",
"Identificator.identificator": {
"@value": "0000.000.000",
"@type": "cl-idt:kbonummer"
}
},
"Aanbieder.voorkeursnaam": {
"@value": "e.g. Vervoer Op Maat bvba",
"@language": "nl"
}
},
"Mobiliteitsdienst.uitgevoerdMet": {
"@type": "Uitvoerder",
"Uitvoerder.contactinfo": {
"@type": "Contactinfo",
"Contactinfo.telefoon": "0000000000"

},
"Uitvoerder.registratie": {
"@type": "Identificator",
"Identificator.identificator": {
"@value": "0000.000.000",
"@type": "cl-idt:kbonummer"
}
},
"Uitvoerder.voorkeursnaam": {
"@value": "e.g. Vervoer Op Maat bvba",
"@language": "nl"
}
}
},
{
"Mobiliteitsdienst.naam": "Vervoeropmaatflex",
"Mobiliteitsdienst.type": "cl-mdt:vervoeropmaatflex",
"Mobiliteitsdienst.aanbieder": {
"@type": "Aanbieder",
"Aanbieder.registratie": {
"@type": "Identificator",
"Identificator.identificator": {
"@value": "0000.000.000",
"@type": "cl-idt:kbonummer"
}
},
"Aanbieder.voorkeursnaam": {
"@value": "e.g. Vervoer Op Maat bvba",
"@language": "nl"
}
},

"Mobiliteitsdienst.uitgevoerdMet": {
"@type": "Uitvoerder",
"Uitvoerder.contactinfo": {
"@type": "Contactinfo",
"Contactinfo.telefoon": "0000000000"

},
"Uitvoerder.registratie": {
"@type": "Identificator",
"Identificator.identificator": {
"@value": "0000.000.000",
"@type": "cl-idt:kbonummer"
}
},
"Uitvoerder.voorkeursnaam": {
"@value": "e.g. Vervoer Op Maat bvba",
"@language": "nl"
}
}
},

{
"Mobiliteitsdienst.naam": "Fietslocker",
"Mobiliteitsdienst.type": "cl-mdt:fietslocker",
"Mobiliteitsdienst.aanbieder": {
"@type": "Aanbieder",
"Aanbieder.registratie": {
"@type": "Identificator",
"Identificator.identificator": {
"@value": "0203.430.576",
"@type": "cl-idt:kbonummer"
}
},
"Aanbieder.voorkeursnaam": {
"@value": "NMBS",
"@language": "nl"
}
},
"Mobiliteitsdienst.uitgevoerdMet": {
"@type": "Uitvoerder",
"Uitvoerder.contactinfo": {
"@type": "Contactinfo",
"Contactinfo.telefoon": "0000000000"

},
"Uitvoerder.registratie": {
"@type": "Identificator",
"Identificator.identificator": {
"@value": "0203.430.576",
"@type": "cl-idt:kbonummer"
}
},
"Uitvoerder.voorkeursnaam": {
"@value": "NMBS",
"@language": "nl"
}
}
},
{
"Mobiliteitsdienst.naam": "Deelsteps",
"Mobiliteitsdienst.type": "cl-mdt:deelsteps",
"Mobiliteitsdienst.aanbieder":  {
"@type": "Aanbieder",
"Aanbieder.registratie": {
"@type": "Identificator",
"Identificator.identificator": {
"@value": "0000.000.000",
"@type": "cl-idt:kbonummer"
}
},
"Aanbieder.voorkeursnaam": {
"@value": "e.g. Vervoer Op Maat bvba",
"@language": "nl"
}
}
},

{
"Mobiliteitsdienst.naam": "Fietspomp",
"Mobiliteitsdienst.type": "cl-mdt:fietspomp",
"Mobiliteitsdienst.aanbieder": {
"@type": "Aanbieder",
"Aanbieder.registratie": {
"@type": "Identificator",
"Identificator.identificator": {
"@value": "0203.430.576",
"@type": "cl-idt:kbonummer"
}
},
"Aanbieder.voorkeursnaam": {
"@value": "NMBS",
"@language": "nl"
}
},

"Mobiliteitsdienst.uitgevoerdMet": {
"@type": "Uitvoerder",
"Uitvoerder.contactinfo": {
"@type": "Contactinfo",
"Contactinfo.telefoon": "0000000000"

},
"Uitvoerder.registratie": {
"@type": "Identificator",
"Identificator.identificator": {
"@value": "0203.430.576",
"@type": "cl-idt:kbonummer"
}
},
"Uitvoerder.voorkeursnaam": {
"@value": "NMBS",
"@language": "nl"
}
}
},

{
"@type": "AanvullendeDienst",
"AanvullendeDienst.type": "cl-mdt:Fietspomp",
"AanvullendeDienst.beschikbaarOpInfrastructuurelement":
{
"@type": "Infrastructuurelement",
"Infrastructuurelement.type": "cl-ine:Fietspomp"
}

},
{
"Mobiliteitsdienst.naam": "Park and ride",
"Mobiliteitsdienst.type": "cl-mdt:parkandride",
"Mobiliteitsdienst.aanbieder": {
"@type": "Aanbieder",
"Aanbieder.registratie": {
"@type": "Identificator",
"Identificator.identificator": {
"@value": "0207.437.468",
"@type": "cl-idt:kbonummer"
}
},
"Aanbieder.voorkeursnaam": {
"@value": "Stad Aalst",
"@language": "nl"
}
},
"Mobiliteitsdienst.uitgevoerdMet": {
"@type": "Uitvoerder",
"Uitvoerder.contactinfo": {
"@type": "Contactinfo",
"Contactinfo.telefoon": "0000000000"

},
"Uitvoerder.registratie": {
"@type": "Identificator",
"Identificator.identificator": {
"@value": "0207.437.468",
"@type": "cl-idt:kbonummer"
}
},
"Uitvoerder.voorkeursnaam": {
"@value": "stad Aalst",
"@language": "nl"
}
}
}
],
"Vervoersknooppunt.transportobject": {
"@type": "Transportknoop",
"Transportknoop.type": "cl-tkt:treinstation",
"Transportknoop.geometrie": "<gml:Point srsName=\"http:\\//www.opengis.net/def/crs/EPSG/0/31370\"><gml:coordinates>126832,7 181419,33</gml:coordinates><gml:Point>",
"Transportknoop.transportnetwerk": [
{
"@type": "Transportnetwerk",
"Transportnetwerk.transporttype": "cl-trt:inspire-trn",
"Transportnetwerk.geografischeNaam": {
"@value": "KernNet",
"@language": "nl"
}
},
{
"@type": "Transportnetwerk",
"Transportnetwerk.transporttype": "cl-trt:inspire-trn",
"Transportnetwerk.geografischeNaam": {
"@value": "AanvullendNet",
"@language": "nl"
}
},
{
"@type": "Transportnetwerk",
"Transportnetwerk.transporttype": "cl-trt:inspire-trn",
"Transportnetwerk.geografischeNaam": {
"@value": "FunctioneelNet",
"@language": "nl"
}
}
]
},
"Vervoersknooppunt.wegbeheerder": {
"@type": "Organisatie",
"Organisatie.voorkeursnaam": {
"@value": "Aalst",
"@language": "nl"
}
},
"Vervoersknooppunt.dienst": [
{
"@id": "_:av001",
"@type": "AanvullendeDienst",
"AanvullendeDienst.aanbieder": "_:aan099",
"AanvullendeDienst.naam": {
"@value": "Zitbanken wachtzaal",
"@language": "nl"
},
"AanvullendeDienst.type": "cl-avd:wachtacommodatie",
"AanvullendeDienst.beschikbaarOpInfrastructuurelement": [
{
"@type": "Zitbank"
}
],
"rdf:value": 10
},
{
"@id": "_:aan099",
"@type": "Organisatie",
"voorkeursnaam": {
"@value": "Onbekend",
"@language": "nl"
}
},


{
"@id": "_:av002",
"@type": "AanvullendeDienst",
"AanvullendeDienst.aanbieder": "_:aan099",
"AanvullendeDienst.type": "cl-avd:vuilnisbak",
"AanvullendeDienst.beschikbaarOpInfrastructuurelement": [
{
"@type": "vuilnisbak"
}
]
},
{
"@id": "_:av003",
"@type": "AanvullendeDienst",
"AanvullendeDienst.aanbieder": "_:aan099",
"AanvullendeDienst.type": "cl-avd:pakketautomaat",
"AanvullendeDienst.beschikbaarOpInfrastructuurelement": [
{
"@type": "Infrastructuurelement",
"Infrastructuurelement.type": "cl-ine:pakketautomaat"
}
]
},
{
"@id": "_:av004",
"@type": "AanvullendeDienst",
"AanvullendeDienst.aanbieder": "_:aan099",
"AanvullendeDienst.type": "cl-avd:lockers",
"AanvullendeDienst.beschikbaarOpInfrastructuurelement": [
{
"@type": "Infrastructuurelement",
"Infrastructuurelement.type": "cl-ine:lockers"
}
]
},
{
"@id": "_:av005",
"@type": "AanvullendeDienst",
"AanvullendeDienst.aanbieder": "_:aan099",
"AanvullendeDienst.type": "cl-avd:fietshersteldienst",
"AanvullendeDienst.beschikbaarOpInfrastructuurelement": [
{
"@type": "Infrastructuurelement",
"Infrastructuurelement.type": "cl-ine:fietsherstelplaats"
}
]
},
{
"@id": "_:av006",
"@type": "AanvullendeDienst",
"AanvullendeDienst.aanbieder": "_:aan099",
"AanvullendeDienst.type": "cl-avd:sanitair",
"AanvullendeDienst.beschikbaarOpInfrastructuurelement": [
{
"@type": "Infrastructuurelement",
"Infrastructuurelement.type": "cl-ine:sanitair"
}
]
}, {
"@id": "_:av007",
"@type": "AanvullendeDienst",
"AanvullendeDienst.aanbieder": "_:aan099",
"AanvullendeDienst.type": "cl-avd:Rode_brievenbus",
"AanvullendeDienst.beschikbaarOpInfrastructuurelement": [
{
"@type": "Infrastructuurelement",
"Infrastructuurelement.type": "cl-ine:Rode_brievenbus"
}
]

},
{
"@id": "_:av008",
"@type": "AanvullendeDienst",
"AanvullendeDienst.aanbieder": "_:aan099",
"AanvullendeDienst.type": "cl-avd:AED",
"AanvullendeDienst.beschikbaarOpInfrastructuurelement":
{
"@type": "Infrastructuurelement",
"Infrastructuurelement.type": "cl-ine:AED"
}
},
{
"@id": "_:av009",
"@type": "AanvullendeDienst",
"AanvullendeDienst.aanbieder": "_:aan099",
"AanvullendeDienst.type": "cl-avd:Geld_automaat",
"AanvullendeDienst.beschikbaarOpInfrastructuurelement":
{
"@type": "Infrastructuurelement",
"Infrastructuurelement.type": "cl-ine:Geld_automaat"
}
},
{
"@id": "_:av010",
"@type": "AanvullendeDienst",
"AanvullendeDienst.aanbieder": "_:aan099",
"AanvullendeDienst.type": "cl-avd:Wifi",
"AanvullendeDienst.beschikbaarOpInfrastructuurelement":
{
"@type": "Infrastructuurelement",
"Infrastructuurelement.type": "cl-ine:bagagelockers"
}
},
{
"@id": "_:av011",
"@type": "AanvullendeDienst",
"AanvullendeDienst.aanbieder": "_:aan099",
"AanvullendeDienst.type": "cl-avd:Oplaadpunt_smartphones",
"AanvullendeDienst.beschikbaarOpInfrastructuurelement":
{
"@type": "Infrastructuurelement",
"Infrastructuurelement.type": "cl-ine:Oplaadpunt_smartphones"
}

},
{
"@id": "_:av012",
"@type": "AanvullendeDienst",
"AanvullendeDienst.aanbieder": "_:aan099",
"AanvullendeDienst.type": "cl-avd:Drinkwatervoorziening",
"AanvullendeDienst.beschikbaarOpInfrastructuurelement":
{
"@type": "Infrastructuurelement",
"Infrastructuurelement.type": "cl-ine:Drinkwatervoorziening"
}
},
{
"@id": "_:av013",
"@type": "AanvullendeDienst",
"AanvullendeDienst.aanbieder": "_:aan099",
"AanvullendeDienst.type": "cl-avd:Vergaderruimte",
"AanvullendeDienst.beschikbaarOpInfrastructuurelement":
{
"@type": "Infrastructuurelement",
"Infrastructuurelement.type": "cl-ine:Vergaderruimte"
}
},
{
"@id": "_:av014",
"@type": "AanvullendeDienst",
"AanvullendeDienst.aanbieder": "_:aan099",
"AanvullendeDienst.type": "cl-avd:Voedings-en_krantenwinkel",
"AanvullendeDienst.beschikbaarOpInfrastructuurelement":
{
"@type": "Infrastructuurelement",
"Infrastructuurelement.type": "cl-ine:Voedings-en_krantenwinkel"
}
},
{
"@id": "_:av015",
"@type": "AanvullendeDienst",
"AanvullendeDienst.aanbieder": "_:aan099",
"AanvullendeDienst.type": "cl-avd:Eet-en_drankgelegenheid",
"AanvullendeDienst.beschikbaarOpInfrastructuurelement":
{
"@type": "Infrastructuurelement",
"Infrastructuurelement.type": "cl-ine:Eet-en_drankgelegenheid"
}
},
{
"@id": "_:av016",
"@type": "AanvullendeDienst",
"AanvullendeDienst.type": "cl-avd:Uitleenpunt_kinderwagens",
"AanvullendeDienst.beschikbaarOpInfrastructuurelement":
{
"@type": "Infrastructuurelement",
"Infrastructuurelement.type": "cl-ine:Uitleenpunt_kinderwagens"
}
},
{
"@id": "_:av017",
"@type": "AanvullendeDienst",
"AanvullendeDienst.type": "cl-avd:Spin-off_centrumdiensten",
"AanvullendeDienst.beschikbaarOpInfrastructuurelement":
{
"@type": "Infrastructuurelement",
"Infrastructuurelement.type": "cl-ine:Spin-off_centrumdiensten"
}
}
]
}
]
}


JSON bestand 2:

"2-Erembodegem Station": {
  "Locatie": {
 "CRS": "EPSG:31370",
 "Geometrie": "POLYGON ((128068 178856))"
  },
  "Attributen": {
 "ID": "2",
 "Naam": "Erembodegem Station",
 "Provincie": "Oost-Vlaanderen",
 "VVR": "Aalst",
 "Gemeente": "Aalst",
 "Status": "Goedgekeurd",
 "Categorie BVR": "Regionaal",
 "Beheer": ["Lokaal bestuur"],
 "Mobiliteitsaanbod": {
"Deelsystemen": [
   {
  "Naam": "VOM deelmobiliteit",
  "Aanbod": ["Deelfietsen, Deelwagens"]
   }
],
"Lijnbus halte": "false",
"Tramhalte": "false",
"Kernnet": "false",
"Aanvullend net": "false",
"Functioneel net": "true",
"Metrohalte": "false",
"Treinstation": "true",
"VOM flex halte": "true",
"Deelwagen VOM": "false",
"Deelfiets VOM": "false",
"Fietslockers": "false",
"Deelsteps": "false",
"Fietspomp": "false",
"Park and ride": "false",
"Kiss and ride": "false"
 },
 "Aanvullend aanbod": {
"Wachtaccomodatie": "false",
"Vuilnisbak": "false",
"Pakketautomaat": "false",
"Bagagelocker": "false",
"Fietshersteldienst": "false",
"Sanitair": "false",
"Rode brievenbus": "false",
"AED": "false",
"Geld automaat": "false",
"Wifi": "false",
"Oplaadpunt smartphones": "false",
"Drinkerwatervoorziening": "false",
"Vergaderruimte": "false",
"Voedings-en krantenwinkel": "false",
"Eet- en drankgelegenheid": "false",
"Uitleenpunt kinderwagens": "false",
"Spin-off centrumdiensten": "false"
 }
}
}

output json-ld bestand:


  """
  temperature=0,
  max_tokens=10000,
  top_p=1,
  frequency_penalty=0.2,
  presence_penalty=0
)
