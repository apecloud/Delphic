import logging
import tempfile
import traceback
from pathlib import Path

import qdrant_client
from django.conf import settings
from langchain.llms import FakeListLLM
from llama_index import (
    VectorStoreIndex,
    ServiceContext,
    SimpleDirectoryReader,
)
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams

from config import celery_app
from delphic.indexes.models import Collection, CollectionStatus
from delphic.models.models import get_embedding_model
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import (
    LangchainEmbedding
)

logger = logging.getLogger(__name__)


@celery_app.task
def create_index(collection_id):
    """
    Celery task to create a GPTSimpleVectorIndex for a given Collection object.

    This task takes the ID of a Collection object, retrieves it from the
    database along with its related documents, and saves the document files
    to a temporary directory. Then, it creates a GPTSimpleVectorIndex using
    the provided code and saves the index to the Comparison.model FileField.

    Args:
        collection_id (int): The ID of the Collection object for which the
                             index should be created.

    Returns:
        bool: True if the index is created and saved successfully, False otherwise.
    """
    try:
        # Get the Collection object with related documents
        collection = Collection.objects.prefetch_related("documents").get(
            id=collection_id
        )
        collection.status = CollectionStatus.RUNNING
        collection.save()

        try:
            # Create a temporary directory to store the document files
            with tempfile.TemporaryDirectory() as tempdir:
                tempdir_path = Path(tempdir)

                # Save the document files to the temporary directory
                for document in collection.documents.all():
                    with document.file.open("rb") as f:
                        file_data = f.read()

                    temp_file_path = tempdir_path / document.file.name
                    temp_file_path.parent.mkdir(parents=True, exist_ok=True)
                    with temp_file_path.open("wb") as f:
                        f.write(file_data)

                # Create the GPTSimpleVectorIndex
                loader = SimpleDirectoryReader(
                    tempdir, recursive=True, exclude_hidden=False
                )
                documents = loader.load_data()

                client = qdrant_client.QdrantClient(
                    # you can use :memory: mode for fast and light-weight experiments,
                    # it does not require to have Qdrant deployed anywhere
                    # but requires qdrant-client >= 1.1.1
                    # location=":memory:"
                    # otherwise set Qdrant instance address with:
                    url=settings.QDRANT_URL,
                    # set API KEY for Qdrant Cloud
                    # api_key="<qdrant-api-key>",
                )
                result = client.get_collections()
                for item in result.collections:
                    if item.name == collection.id and collection.status == CollectionStatus.COMPLETE:
                        return

                client.create_collection(
                    collection_name=collection.id,
                    vectors_config=VectorParams(size=settings.EMBEDDING_VECTOR_SIZE, distance=Distance.DOT),
                )
                vector_store = QdrantVectorStore(client=client, collection_name=collection.id)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)

                service_context = ServiceContext.from_defaults(
                    llm=FakeListLLM(responses=["fake"]),
                    embed_model=get_embedding_model(),
                )
                # build index
                VectorStoreIndex.from_documents(
                    documents, service_context=service_context, storage_context=storage_context
                )

            collection.status = CollectionStatus.COMPLETE
            collection.processing = False
            collection.save()

            return True

        except Exception as e:
            logger.error(f"Error creating index for collection {collection_id}: {e}")
            traceback.print_exc()
            collection.status = CollectionStatus.ERROR
            collection.save()

            return False

    except Exception as e:
        logger.error(f"Error loading collection: {e}")
        return False
