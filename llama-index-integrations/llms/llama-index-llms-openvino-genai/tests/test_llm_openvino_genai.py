import asyncio
import time

from llama_index.llms.openvino_genai import OpenVINOGenAILLM, OpenVINOGenAIContinousPipeline
from llama_index.core.base.llms.base import BaseLLM
import huggingface_hub as hf_hub
import pytest

def test_embedding_class():
    names_of_base_classes = [b.__name__ for b in OpenVINOGenAILLM.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes

def test_streaming_completion():
    model_id = "OpenVINO/qwen3-0.6b-int4-ov"
    model_path = "qwen3-0.6b-int4-ov"
    hf_hub.snapshot_download(model_id, local_dir=model_path)
    ov_llm = OpenVINOGenAILLM(
        model_path=model_path,
        device="CPU",
    )
    ov_llm.config.max_new_tokens = 100
    response = ov_llm.stream_complete("Who is Paul Graham?")
    intermediate_response = None
    for chunk in response:
        intermediate_response = chunk

    assert len(intermediate_response.text) > 0

@pytest.mark.skip(reason="no need to test")
def consume_stream_complete(model, prompt):
    chunks = []
    for response in model.stream_complete(prompt):
        chunks.append(response.delta or response.text or "")
    return "".join(chunks)

@pytest.mark.skip(reason="no need to test")
async def ContinousPipeline_streaming_completion():
    model_id = "OpenVINO/qwen3-0.6b-int4-ov"
    model_path = "qwen3-0.6b-int4-ov"
    hf_hub.snapshot_download(model_id, local_dir=model_path)
    ov_pipeline = OpenVINOGenAIContinousPipeline(
        model_path=model_path,
        device="CPU",
    )
    ov_pipeline.config.max_new_tokens = 100
    prompt = "Who is Paul Graham?"
    start_time = time.time()
    result = await asyncio.gather(
        asyncio.to_thread(consume_stream_complete, ov_pipeline, prompt),
        asyncio.to_thread(consume_stream_complete, ov_pipeline, prompt),
    )
    end_time = time.time()
    assert len(result) > 0
    print(f"Execution time: {end_time - start_time:.2f} seconds")

def test_continuous_pipeline_streaming_completion():
    asyncio.run(ContinousPipeline_streaming_completion())