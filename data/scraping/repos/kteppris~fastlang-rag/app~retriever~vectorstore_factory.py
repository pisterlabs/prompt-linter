from omegaconf import OmegaConf
import chromadb
from langchain.vectorstores.chroma import Chroma
from langchain.vectorstores.base import VectorStore
from langchain.embeddings.base import Embeddings

class VectorstoreFactory:

    @staticmethod
    def _init_chroma_vectorstore(config: OmegaConf, encoder: Embeddings) -> Chroma:
        """Get langchain Chroma wrapper for Chroma vectorstore client."""
        client = chromadb.HttpClient(
            host=config.vectorstore.settings.host, 
            port=config.vectorstore.settings.port,
            ssl=config.vectorstore.settings.get("ssl", False)
        )
        return Chroma(
            client=client,
            embedding_function=encoder
        )

    @staticmethod
    def get_vectorstore(config: OmegaConf, encoder: Embeddings) -> VectorStore:
        """
        Generate and return an instance of the specified vectorstore client based on the given configuration.

        Depending on the vectorstore type specified in the configuration, it then initializes 
        the corresponding vectorstore returns it.

        Parameters
        ----------
        config : OmegaConf
            The configuration object that contains settings and specifications for the desired vectorstore.
            This should include the type of the vectorstore, any necessary settings.

        Returns
        -------
        VectorStore 
            An instance of a specified langchain VectorStore client.

        Raises
        ------
        ValueError
            If the vectorstore type specified in the configuration is not recognized or supported.

        Example
        -------
        >>> config = OmegaConf.create({
                "vectorstore": {"type": "chroma", "settings": {...}}
            })
        >>> vectorstore = VectorstoreFactory.get_vectorstore(config)
        """
        vectorstore_initializers = {
            "chroma": VectorstoreFactory._init_chroma_vectorstore
            # Space for other Vectorstores
        }

        vectorstore_type = config.vectorstore.type.lower()
        if vectorstore_type not in vectorstore_initializers:
            raise ValueError(f"Unknown vectorstore type: {config.vectorstore.type}")

        return vectorstore_initializers[vectorstore_type](config, encoder)
