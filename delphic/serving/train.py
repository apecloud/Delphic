import bentoml
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from constants import model_name, bento_name


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='auto',
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    # max_memory={0: "28GB"} # Uncomment this line with you encounter CUDA out of memory errors
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

model.tie_weights()

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=2048,
    temperature=0,
    top_p=0.95,
    repetition_penalty=1.15,
    # generation_config=generation_config
)

bentoml.transformers.save_model(name=bento_name, pipeline=pipe)
