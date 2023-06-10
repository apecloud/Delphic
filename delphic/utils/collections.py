import sys
import torch
import logging
import textwrap
import qdrant_client
from pathlib import Path

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from django.conf import settings
from langchain import OpenAI
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from llama_index import LLMPredictor, ServiceContext
from llama_index.llm_predictor import HuggingFaceLLMPredictor
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index import (
    VectorStoreIndex,
    LangchainEmbedding
)
from langchain.llms import HuggingFacePipeline, GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForCausalLM, GenerationConfig
from transformers import LlamaForCausalLM, LlamaTokenizer, pipeline

from delphic.indexes.models import Collection, CollectionStatus

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
logger = logging.getLogger(__name__)


def format_source(source):
    """
    Format a source object as a nicely-structured markdown text.

    Args:
        source (llama_index.schema.Source): The source object to format.

    Returns:
        str: The formatted markdown text for the given source.
    """
    formatted_source = (
        f"- **{source.title}**\n\n{textwrap.indent(source.content, '  ')}\n\n"
    )
    return formatted_source


async def load_collection_model(collection_id: str | int) -> VectorStoreIndex:
    """
    Load the Collection model from cache or the database, and return the index.

    Args:
        collection_id (Union[str, int]): The ID of the Collection model instance.

    Returns:
        GPTSimpleVectorIndex: The loaded index.

    This function performs the following steps:
    1. Retrieve the Collection object with the given collection_id.
    2. Check if a JSON file with the name '/cache/model_{collection_id}.json' exists.
    3. If the JSON file doesn't exist, load the JSON from the Collection.model FileField and save it to
       '/cache/model_{collection_id}.json'.
    4. Call GPTSimpleVectorIndex.load_from_disk with the cache_file_path.
    """
    # Retrieve the Collection object
    collection = await Collection.objects.aget(id=collection_id)
    logger.info(f"load_collection_model() - loaded collection {collection_id}")

    if collection.status == CollectionStatus.COMPLETE:
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

        service_context = ServiceContext.from_defaults(
            llm=get_inference_model(),
            embed_model=get_embedding_model(),
        )

        vector_store = QdrantVectorStore(client=client, collection_name=collection.id)
        index = VectorStoreIndex.from_vector_store(
            vector_store, service_context=service_context
        )
        logger.info(
            "load_collection_model() - Llamaindex loaded and ready for query..."
        )
    else:
        logger.error(
            f"load_collection_model() - collection {collection_id} has no model!"
        )
        raise ValueError("No model exists for this collection!")

    return index


async def query_collection(collection_id: str | int, query_str: str) -> str:
    """
    Query a collection with a given question and return the response as nicely-structured markdown text.

    Args:
        collection_id (Union[str, int]): The ID of the Collection model instance.
        query_str (str): The natural language question to query the collection.

    Returns:
        str: The response from the query as nicely-structured markdown text.

    This function performs the following steps:
    1. Load the GPTSimpleVectorIndex from the Collection with the given collection_id using load_collection_model.
    2. Call index.query with the query_str and get the llama_index.schema.Response object.
    3. Format the response and sources as markdown text and return the formatted text.
    """
    try:
        # Load the index from the collection
        index = await load_collection_model(collection_id)

        # Call index.query and return the response
        response = index.query(query_str)

        # Format the response as markdown
        markdown_response = f"## Response\n\n{response}\n\n"

        if response.source_nodes:
            markdown_sources = f"## Sources\n\n{response.get_formatted_sources()}"
        else:
            markdown_sources = ""

        formatted_response = f"{markdown_response}{markdown_sources}"

    except ValueError:
        formatted_response = "No model exists for this collection!"

    return formatted_response


def load_model(device_type, model_id, model_basename=None):
    """
    Select a model for text generation using the HuggingFace library.
    If you are running this for the first time, it will download a model for you.
    subsequent runs will use the model from the disk.

    Args:
        device_type (str): Type of device to use, e.g., "cuda" for GPU or "cpu" for CPU.
        model_id (str): Identifier of the model to load from HuggingFace's model hub.
        model_basename (str, optional): Basename of the model if using quantized models.
            Defaults to None.

    Returns:
        HuggingFacePipeline: A pipeline object for text generation using the loaded model.

    Raises:
        ValueError: If an unsupported model or device type is provided.
    """

    logging.info(f'Loading Model: {model_id}, on: {device_type}')
    logging.info('This action can take a few minutes!')

    if model_basename is not None:
        # The code supports all huggingface models that ends with GPTQ and have some variation of .no-act.order or .safetensors in their HF repo.
        logging.info('Using AutoGPTQForCausalLM for quantized models')

        if '.safetensors' in model_basename:
            # Remove the ".safetensors" ending if present
            model_basename = model_basename.replace('.safetensors', "")

        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        logging.info('Tokenizer loaded')

        model = AutoGPTQForCausalLM.from_quantized(
            model_id,
            model_basename=model_basename,
            use_safetensors=True,
            trust_remote_code=True,
            device="cuda:0",
            use_triton=False,
            quantize_config=None
        )
    elif device_type.lower() == 'cuda':  # The code supports all huggingface models that ends with -HF or which have a .bin file in their HF repo.
        logging.info('Using AutoModelForCausalLM for full models')
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        logging.info('Tokenizer loaded')

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map='auto',
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            # max_memory={0: "15GB"} # Uncomment this line with you encounter CUDA out of memory errors
        )
        model.tie_weights()
    else:
        logging.info('Using LlamaTokenizer')
        tokenizer = LlamaTokenizer.from_pretrained(model_id)
        model = LlamaForCausalLM.from_pretrained(model_id)

    # Load configuration from the model to avoid warnings
    generation_config = GenerationConfig.from_pretrained(model_id)
    # see here for details: https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig.from_pretrained.returns

    # Create a pipeline for text generation
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.15,
        generation_config=generation_config
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    logging.info('Local LLM Loaded')

    return local_llm


def get_inference_model():
    return load_model("cuda", settings.INFERENCE_MODEL, settings.INFERENCE_MODEL_BASENAME)


def get_embedding_model():
    return LangchainEmbedding(HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL, model_kwargs={"device": 0}
    ))
