from ragen.args import build_argument_parser
from ragen.client import OpenAIClient, PgClient, generate_prompt
from ragen.file import Chunk, ChunkGenerator


def main():
    parser = build_argument_parser()
    args = parser.parse_args()

    gen = ChunkGenerator(args.chunk_size)
    chat_client = OpenAIClient(args.model, args.api_key, args.api_base)
    emb_client = OpenAIClient(args.emb_model, args.emb_api_key, args.emb_api_base)
    db_client = PgClient(
        args.db_host, args.db_user, args.db_password, args.db_port, args.emb_dim
    )

    for file in args.data:
        for i, text in enumerate(gen.generate(path=file)):
            chunk = Chunk(
                filename=file, index=i, text=text, emb=emb_client.embeddings(text)
            )
            db_client.insert_chunk(chunk=chunk)
    db_client.indexing()

    print("Welcome to Ragen!")
    print(
        "You are using the {} embedding model and {} language model.".format(
            args.emb_model,
            args.model,
        )
    )
    print("You're in the context of the following files:", args.data)
    print("End the conversation by pressing `Ctrl+C`")
    while True:
        try:
            user_input = input("Enter your question: ")
        except KeyboardInterrupt:
            print("\nBye!")
            break
        request = Chunk(
            filename="", index=0, text=user_input, emb=emb_client.embeddings(user_input)
        )
        user_context = db_client.retrieve_similar_chunk(request, args.top_k)
        chat_client.chat(
            generate_prompt(user_context, request),
        )
