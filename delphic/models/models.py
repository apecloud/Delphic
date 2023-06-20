import os
import json
import torch
import logging

from langchain.utilities import TextRequestsWrapper
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from typing import Optional, List, Mapping, Any
from django.conf import settings
from langchain.llms.base import LLM
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from llama_index import (
    LangchainEmbedding
)
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers import LlamaForCausalLM, LlamaTokenizer, pipeline

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

    if model_basename is not None and model_basename != "":
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


class CustomLLM(LLM):

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        prompt_length = len(prompt)
        response = TextRequestsWrapper().post(settings.EXTERNAL_INFERENCE_ENDPOINT, prompt)
        print("prompt: " + prompt)
        print("response: " + response)
        return json.loads(response)[0]["generated_text"][prompt_length:]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": settings.INFERENCE_MODEL}

    @property
    def _llm_type(self) -> str:
        return "custom"


def get_inference_model():
    if settings.EXTERNAL_INFERENCE_ENDPOINT is not None and settings.EXTERNAL_INFERENCE_ENDPOINT != "":
        llm = CustomLLM()
        logging.info("Remote LLM %s Loaded", settings.EXTERNAL_INFERENCE_ENDPOINT)
        return llm

    return load_model(settings.DEVICE_TYPE, settings.INFERENCE_MODEL, settings.INFERENCE_MODEL_BASENAME)


def get_embedding_model():
    if settings.EMBEDDING_MODEL.startswith("hkunlp"):
        model = HuggingFaceInstructEmbeddings
    else:
        model = HuggingFaceEmbeddings

    return LangchainEmbedding(model(
        model_name=settings.EMBEDDING_MODEL, model_kwargs={"device": "cuda"}
    ))


INFERENCE_MODEL = None
if os.getenv("REQUIRE_INFERENCE_MODEL") == "True":
    INFERENCE_MODEL = get_inference_model()

EMBEDDING_MODEL = None
if os.getenv("REQUIRE_EMBEDDING_MODEL") == "True":
    EMBEDDING_MODEL = get_embedding_model()
