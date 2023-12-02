from langchain.document_loaders import UnstructuredURLLoader

urls = ["https://migrant.info.pl/en/Residence_permit_for_a_fixed_period"]

loader = UnstructuredURLLoader(urls=urls)

data = loader.load()

print(data)