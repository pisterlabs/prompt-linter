import argparse
from .data.loader import HuggingFaceDatasetImporter, LocalDatasetImporter, KaggleDatasetImporter
from langchain.chat_models import ChatOpenAI

def describe_dataset(openai_key, importer_type, destination_path, dataset_key=None,):
    # Process input based on the selected importer
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_key, temperature=0)
    if importer_type == "HuggingFaceDatasetImporter":
        data_ingestor = HuggingFaceDatasetImporter()
        data_ingestor.import_dataset(dataset_key, destination_path)
        data_ingestor.walk_dataset(destination_path)
        out = data_ingestor.ingest(llm=llm)
        return f"Description: {out}", data_ingestor
    elif importer_type == "KaggleDatasetImporter":
        data_ingestor = KaggleDatasetImporter()
        data_ingestor.import_dataset(dataset_key, destination_path)
        data_ingestor.walk_dataset(destination_path)
        out = data_ingestor.ingest(llm=llm)
        return f"Description: {out}", data_ingestor
    elif importer_type == "LocalDatasetImporter":
        data_ingestor = LocalDatasetImporter()
        data_ingestor.walk_dataset(destination_path)
        out = data_ingestor.ingest(llm=llm)
        return f"Description: {out}", data_ingestor
    else:
        return "Invalid selection"

def main():
    parser = argparse.ArgumentParser(description="Menu options for processing input.")
    parser.add_argument("openai_key", type=str, help="OpenAI Key")
    parser.add_argument("importer_type", type=str, choices=["HuggingFaceDatasetImporter", "KaggleDatasetImporter", "LocalDatasetImporter"], help="Importer Type")
    parser.add_argument("--dataset_key", type=str, help="Dataset Key (Optional)")
    parser.add_argument("destination_path", type=str, help="Destination Path")
    args = parser.parse_args()

    output = describe_dataset(args.openai_key, args.importer_type, args.destination_path, args.dataset_key)
    print("Output:", output)

if __name__ == "__main__":
    main()