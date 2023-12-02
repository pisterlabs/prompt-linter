import json
from fhirpy import SyncFHIRClient
from tabulate import tabulate
from fhirpy.base.searchset import Raw
import uuid
import base64

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader

contentType = "application/fhir+json"


def FormatResource(resource, data, opt):
    rows = []
    if opt == 1:
        if resource == "Patient":
            for rowval in data:
                row = {
                    "id": rowval.get('id'),
                    "lastName": rowval.get_by_path('name.0.family'),
                    "firstName": rowval.get_by_path('name.0.given.0'),
                    "birthDate": rowval.get_by_path('birthDate'),
                    "gender": rowval.get_by_path('gender'),
                    "address": {
                        "line": rowval.get_by_path('address.0.line'),
                        "city": rowval.get_by_path('address.0.city'),
                        "state": rowval.get_by_path('address.0.state'),
                        "postalCode": rowval.get_by_path('address.0.postalCode'),
                        "country": rowval.get_by_path('address.0.country')
                    },
                    "communicationLanguage": rowval.get_by_path('communication.0.language.text'),
                    "phone": rowval.get_by_path('telecom.0.value')
                }
                rows.append(row)
        elif resource == "Observation":
            for rowval in data:
                row = {
                    "id": rowval.get('id'),
                    "category": rowval.get_by_path('category.0.coding.0.code'),
                    "code": rowval.get_by_path('code.coding.0.code'),
                    "value": rowval.get_by_path('valueQuantity.value'),
                    "uom": rowval.get_by_path('valueQuantity.code'),
                    "date": rowval.get('effectiveDateTime'),
                    "patientId": rowval.get_by_path('subject.reference')
                }
                rows.append(row)
        elif resource == 'DocumentReference':
            for rowval in data:
                row = {
                    "id": rowval.get('id'),
                    "patientId": rowval.get_by_path('subject.reference'),
                    "practitionerId": rowval.get_by_path('author.0.reference'),
                    "base64payload": rowval.get_by_path('content.0.attachment.data'),
                    "mimeType": rowval.get_by_path('content.0.attachment.contentType'),
                    "updatedDate": rowval.get_by_path('meta.lastUpdated'),
                }
                rows.append(row)
        elif resource == 'Encounter':
            for rowval in data:
                row = {
                    "id": rowval.get('id'),
                    "status": rowval.get_by_path('subject.reference'),
                    "type": rowval.get_by_path('type.0.text'),
                    "practitionerId": rowval.get_by_path('participant.0.individual.reference'),
                    "practitionerName": rowval.get_by_path('participant.0.individual.display'),
                    "patientId": rowval.get_by_path('subject.reference'),
                    "patientName": rowval.get_by_path('subject.display'),
                    "start": rowval.get_by_path('period.start'),
                    "end": rowval.get_by_path('period.end'),
                    "updatedDate": rowval.get_by_path('meta.lastUpdated'),
                }
                rows.append(row)
    return rows


def GetResource(resource, id, url, api_key):
    # Get Connection
    client = SyncFHIRClient(url=url, extra_headers={
                            "Content-Type": contentType, "x-api-key": api_key})
    data = ""
    try:
        if len(id) > 0:
            data = client.resources(resource).search(_id=id).fetch()
        else:
            data = client.resources(resource).fetch()

    except Exception as e:
        print("Error :" + str(e))

    rows = FormatResource(resource, data, 1)
    # print(rows)
    return json.dumps(rows)


def GetPatientResources(resource, patientId, url, api_key):
    # Get Connection
    cclient = SyncFHIRClient(url=url, extra_headers={
                             "Content-Type": contentType, "x-api-key": api_key})
    try:
        data = cclient.resources(resource).search(patient=patientId).fetch()
    except:
        print("Unable to get Resource Type")
        return
    rows = FormatResource(resource, data, 1)
    # print(rows)
    return json.dumps(rows)


def CreateDocumentForPatient(patientId, practitionerId, base64payload, mimeType, url, api_key):
    headers = {"Content-Type": contentType, "x-api-key": api_key}
    client = SyncFHIRClient(url=url, extra_headers=headers)

    patient = client.resources('Patient').search(_id=patientId).first()
    practitioner = client.resources(
        'Practitioner').search(_id=practitionerId).first()
    docref = client.resource("DocumentReference")

    docref["status"] = "current"
    docref["id"] = str(uuid.uuid4())
    docref["content"] = [{
        "attachment": {
            "contentType": mimeType,
            "data": base64payload
        }
    }]

    base64_bytes = base64payload.encode('utf-8')
    message_bytes = base64.b64decode(base64_bytes)
    summary_content = json.loads(message_bytes.decode('utf-8'))['summary']

    # write to txt files for document search
    lines = [summary_content]
    with open('/home/irisowner/irisdev/summaries/' + docref["id"] + '.txt', 'w') as f:
        f.writelines(lines)

    docref['author'] = [practitioner.to_reference()]
    docref['subject'] = patient.to_reference()

    try:
        resp = docref.save()
    except Exception as e:
        return "Error while creating DocumentReference:" + str(e)

    return json.dumps({"id": docref["id"]})


def QueryDocs(query):
    loader = DirectoryLoader(
        '/home/irisowner/irisdev/summaries/', glob="./*.txt", loader_cls=TextLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    persist_directory = 'db'
    embedding = OpenAIEmbeddings()

    vectordb = Chroma.from_documents(documents=texts,
                                     embedding=embedding,
                                     persist_directory=persist_directory)

    retriever = vectordb.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(),
                                           chain_type="stuff",
                                           retriever=retriever,
                                           return_source_documents=True)

    llm_response = qa_chain(query)
    return json.dumps(process_llm_response(llm_response))


def process_llm_response(llm_response):
    resp = dict()
    resp['response'] = llm_response['result']
    resp['sourceDocIds'] = []
    for source in llm_response["source_documents"]:
        resp['sourceDocIds'].append(
            (source.metadata['source']).split('/')[-1].split('.')[0])
    return resp
