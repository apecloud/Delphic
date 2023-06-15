import bentoml

from bentoml.io import Text, JSON
from constants import model_name, bento_name


runner = bentoml.transformers.get(bento_name + ":latest").to_runner()

svc = bentoml.Service(bento_name, runners=[runner])

@svc.api(input=Text(), output=JSON())
async def forward(input_series: str) -> list:
    return await runner.async_run(input_series)
