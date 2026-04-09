import logging

import copy
from typing import Any, Callable, Optional, Sequence, Union

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms.callbacks import (
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.base.llms.generic_utils import (
    completion_response_to_chat_response,
    stream_completion_response_to_chat_response,
    messages_to_prompt as generic_messages_to_prompt,
)
from llama_index.core.prompts.base import PromptTemplate
import queue
import openvino_genai
import openvino as ov
from threading import Event, Thread

logger = logging.getLogger(__name__)
def get_scheduler_config_genai(config_data, config_name="CB config"):
    import openvino_genai
    user_config = copy.deepcopy(config_data)

    scheduler_config = openvino_genai.SchedulerConfig()
    if user_config:

        if 'cache_eviction_config' in user_config.keys():
            cache_eviction_kwargs = user_config.pop('cache_eviction_config')
            if "aggregation_mode" in cache_eviction_kwargs.keys():
                cache_eviction_kwargs["aggregation_mode"] = getattr(openvino_genai.AggregationMode, cache_eviction_kwargs["aggregation_mode"])

            if "kvcrush_config" in cache_eviction_kwargs.keys() and isinstance(cache_eviction_kwargs["kvcrush_config"], dict):
                crush_config_kwargs = cache_eviction_kwargs["kvcrush_config"]
                if "anchor_point_mode" in crush_config_kwargs.keys():
                    crush_config_kwargs['anchor_point_mode'] = getattr(openvino_genai.KVCrushAnchorPointMode, crush_config_kwargs['anchor_point_mode'])
                cache_eviction_kwargs["kvcrush_config"] = openvino_genai.KVCrushConfig(**crush_config_kwargs)

            scheduler_config.use_cache_eviction = True
            scheduler_config.cache_eviction_config = openvino_genai.CacheEvictionConfig(**cache_eviction_kwargs)

        if 'sparse_attention_config' in user_config.keys():
            sparse_attention_kwargs = user_config.pop('sparse_attention_config')
            if "mode" in sparse_attention_kwargs.keys():
                sparse_attention_kwargs["mode"] = getattr(openvino_genai.SparseAttentionMode, sparse_attention_kwargs["mode"])

            if user_config.pop('use_sparse_attention', True):
                scheduler_config.use_sparse_attention = True
                scheduler_config.sparse_attention_config = openvino_genai.SparseAttentionConfig(**sparse_attention_kwargs)
            else:
                raise RuntimeError("sparse_attention_config cannot be specified when use_sparse_attention is False")

        for param, value in user_config.items():
            setattr(scheduler_config, param, value)

    return scheduler_config


class OpenVINOGenAILLM(CustomLLM):
    r"""
    OpenVINO GenAI LLM.

    Examples:
        `pip install llama-index-llms-openvino-genai`

        ```python
        from llama_index.llms.openvino_genai import OpenVINOgenAILLM

        llm = OpenVINOGenAILLM(
            model_path=model_path,
            device="CPU",
        )

        response = llm.complete("What is the meaning of life?")
        print(str(response))
        ```

    """

    model_path: str = Field(
        default=None,
        description=("The model path to use from local. "),
    )
    system_prompt: str = Field(
        default="",
        description=(
            "The system prompt, containing any extra instructions or context. "
            "The model card on HuggingFace should specify if this is needed."
        ),
    )
    query_wrapper_prompt: PromptTemplate = Field(
        default=PromptTemplate("{query_str}"),
        description=(
            "The query wrapper prompt, containing the query placeholder. "
            "The model card on HuggingFace should specify if this is needed. "
            "Should contain a `{query_str}` placeholder."
        ),
    )
    device: str = Field(
        default="auto", description="The device to use. Defaults to 'auto'."
    )
    is_chat_model: bool = Field(
        default=False,
        description=(
            LLMMetadata.model_fields["is_chat_model"].description
            + " Be sure to verify that you either pass an appropriate tokenizer "
            "that can convert prompts to properly formatted chat messages or a "
            "`messages_to_prompt` that does so."
        ),
    )

    config: Any = Field(
        default=None,
        description=("The LLM generation configurations."),
        exclude=True,
    )

    _pip: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _streamer: Any = PrivateAttr()

    def __init__(
        self,
        model_path: str,
        config: Optional[dict] = {},
        tokenizer: Optional[Any] = None,
        device: Optional[str] = "CPU",
        query_wrapper_prompt: Union[str, PromptTemplate] = "{query_str}",
        is_chat_model: Optional[bool] = False,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: str = "",
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        **kwargs: Any,
    ) -> None:
        class IterableStreamer(openvino_genai.StreamerBase):
            """
            A custom streamer class for handling token streaming
            and detokenization with buffering.

            Attributes:
                tokenizer (Tokenizer): The tokenizer used for encoding
                and decoding tokens.
                tokens_cache (list): A buffer to accumulate tokens
                for detokenization.
                text_queue (Queue): A synchronized queue
                for storing decoded text chunks.
                print_len (int): The length of the printed text
                to manage incremental decoding.

            """

            def __init__(self, tokenizer: Any) -> None:
                """
                Initializes the IterableStreamer with the given tokenizer.

                Args:
                    tokenizer (Tokenizer): The tokenizer to use for encoding
                    and decoding tokens.

                """
                super().__init__()
                self.tokenizer = tokenizer
                self.tokens_cache: list[int] = []
                self.text_queue: Any = queue.Queue()
                self.print_len = 0

            def __iter__(self) -> self:
                """
                Returns the iterator object itself.
                """
                return self

            def __next__(self) -> str:
                """
                Returns the next value from the text queue.

                Returns:
                    str: The next decoded text chunk.

                Raises:
                    StopIteration: If there are no more elements in the queue.

                """
                value = (
                    self.text_queue.get()
                )  # get() will be blocked until a token is available.
                if value is None:
                    raise StopIteration
                return value

            def get_stop_flag(self) -> bool:
                """
                Checks whether the generation process should be stopped.

                Returns:
                    bool: Always returns False in this implementation.

                """
                return False

            def put_word(self, word: Any) -> None:
                """
                Puts a word into the text queue.

                Args:
                    word (str): The word to put into the queue.

                """
                self.text_queue.put(word)

            def put(self, token_id: int) -> bool:
                """
                Processes a token and manages the decoding buffer.
                Adds decoded text to the queue.

                Args:
                    token_id (int): The token_id to process.

                Returns:
                    bool: True if generation should be stopped, False otherwise.

                """
                self.tokens_cache.append(token_id)
                text = self.tokenizer.decode(
                    self.tokens_cache, skip_special_tokens=True
                )

                word = ""
                if len(text) > self.print_len and text[-1] == "\n":
                    word = text[self.print_len :]
                    self.tokens_cache = []
                    self.print_len = 0
                elif len(text) >= 3 and text[-3:] == chr(65533):
                    pass
                elif len(text) > self.print_len:
                    word = text[self.print_len :]
                    self.print_len = len(text)
                self.put_word(word)

                if self.get_stop_flag():
                    self.end()
                    return True
                else:
                    return False

            def write(
                self, token_id: int | list[int]
            ) -> openvino_genai.StreamingStatus:
                """
                Processes a token and manages the decoding buffer.
                Adds decoded text to the queue.

                Args:
                    token_id (int): The token_id to process.

                Returns:
                    bool: True if generation should be stopped, False otherwise.

                """
                if isinstance(token_id, list):
                    self.tokens_cache += token_id
                else:
                    self.tokens_cache.append(token)
                text = self.tokenizer.decode(
                    self.tokens_cache, skip_special_tokens=True
                )
                word = ""
                if len(text) > self.print_len and text[-1] == "\n":
                    word = text[self.print_len :]
                    self.tokens_cache = []
                    self.print_len = 0
                elif len(text) >= 3 and text[-3:] == chr(65533):
                    pass
                elif len(text) > self.print_len:
                    word = text[self.print_len :]
                    self.print_len = len(text)
                self.put_word(word)

                if self.get_stop_flag():
                    self.end()
                    return openvino_genai.StreamingStatus.STOP
                else:
                    return openvino_genai.StreamingStatus.RUNNING

            def end(self) -> None:
                """
                Flushes residual tokens from the buffer
                and puts a None value in the queue to signal the end.
                """
                text = self.tokenizer.decode(
                    self.tokens_cache, skip_special_tokens=True
                )
                if len(text) > self.print_len:
                    word = text[self.print_len :]
                    self.put_word(word)
                    self.tokens_cache = []
                    self.print_len = 0
                self.put_word(None)

            def reset(self) -> None:
                """
                Resets the state.
                """
                self.tokens_cache = []
                self.text_queue = queue.Queue()
                self.print_len = 0

        class ChunkStreamer(IterableStreamer):
            def __init__(self, tokenizer: Any, tokens_len: int = 4) -> None:
                super().__init__(tokenizer)
                self.tokens_len = tokens_len

            def put(self, token_id: int) -> bool:
                if (len(self.tokens_cache) + 1) % self.tokens_len != 0:
                    self.tokens_cache.append(token_id)
                    return False
                return super().put(token_id)

            def write(
                self, token_id: int | list[int]
            ) -> openvino_genai.StreamingStatus:
                if (len(self.tokens_cache) + 1) % self.tokens_len != 0:
                    if isinstance(token_id, list):
                        self.tokens_cache += token_id
                    else:
                        self.tokens_cache.append(token)
                    return openvino_genai.StreamingStatus.RUNNING
                return super().write(token_id)

        """Initialize params."""
        pipe = openvino_genai.LLMPipeline(model_path, device, config, **kwargs)

        config = pipe.get_generation_config()

        tokenizer = tokenizer or pipe.get_tokenizer()
        streamer = ChunkStreamer(tokenizer)

        if isinstance(query_wrapper_prompt, str):
            query_wrapper_prompt = PromptTemplate(query_wrapper_prompt)

        messages_to_prompt = messages_to_prompt or self._tokenizer_messages_to_prompt

        super().__init__(
            tokenizer=tokenizer,
            model_path=model_path,
            device=device,
            query_wrapper_prompt=query_wrapper_prompt,
            is_chat_model=is_chat_model,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
        )

        self._pipe = pipe
        self._tokenizer = tokenizer
        self._streamer = streamer
        self.config = config

    @classmethod
    def class_name(cls) -> str:
        return "OpenVINO_GenAI_LLM"

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
            model_name=self.model_path,
            is_chat_model=self.is_chat_model,
        )

    def _tokenizer_messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str:
        """Use the tokenizer to convert messages to prompt. Fallback to generic."""
        if hasattr(self._tokenizer, "apply_chat_template"):
            messages_dict = [
                {"role": message.role.value, "content": message.content}
                for message in messages
            ]
            return (
                self._tokenizer.apply_chat_template(
                    messages_dict, add_generation_prompt=True
                )
                if isinstance(self._tokenizer, openvino_genai.Tokenizer)
                else self._tokenizer.apply_chat_template(
                    messages_dict, tokenize=False, add_generation_prompt=True
                )
            )

        return generic_messages_to_prompt(messages)

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        """Completion endpoint."""
        full_prompt = prompt
        if not formatted:
            if self.query_wrapper_prompt:
                full_prompt = self.query_wrapper_prompt.format(query_str=prompt)
            if self.completion_to_prompt:
                full_prompt = self.completion_to_prompt(full_prompt)
            elif self.system_prompt:
                full_prompt = f"{self.system_prompt} {full_prompt}"

        if not isinstance(self._tokenizer, openvino_genai.Tokenizer):
            inputs = self._tokenizer(
                full_prompt, add_special_tokens=False, return_tensors="np"
            )
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            full_prompt = openvino_genai.TokenizedInputs(
                ov.Tensor(input_ids), ov.Tensor(attention_mask)
            )

        tokens = self._pipe.generate(full_prompt, self.config, **kwargs)
        if not isinstance(self._tokenizer, openvino_genai.Tokenizer):
            completion_tokens = tokens[0][inputs["input_ids"].size(1) :]
            completion = self._tokenizer.decode(
                completion_tokens, skip_special_tokens=True
            )
        else:
            completion = tokens
        return CompletionResponse(text=completion, raw={"model_output": tokens})

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        """Streaming completion endpoint."""
        full_prompt = prompt
        if not formatted:
            if self.query_wrapper_prompt:
                full_prompt = self.query_wrapper_prompt.format(query_str=prompt)
            if self.system_prompt:
                full_prompt = f"{self.system_prompt} {full_prompt}"

        if not isinstance(self._tokenizer, openvino_genai.Tokenizer):
            inputs = self._tokenizer(
                full_prompt, add_special_tokens=False, return_tensors="np"
            )
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            full_prompt = openvino_genai.TokenizedInputs(
                ov.Tensor(input_ids), ov.Tensor(attention_mask)
            )

        stream_complete = Event()

        def generate_and_signal_complete() -> None:
            """
            Generation function for single thread.
            """
            self._streamer.reset()
            self._pipe.generate(full_prompt, self.config, self._streamer, **kwargs)
            stream_complete.set()
            self._streamer.end()

        t1 = Thread(target=generate_and_signal_complete)
        t1.start()

        # create generator based off of streamer
        def gen() -> CompletionResponseGen:
            text = ""
            for x in self._streamer:
                text += x
                yield CompletionResponse(text=text, delta=x)

        return gen()

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = self.messages_to_prompt(messages)
        completion_response = self.complete(prompt, formatted=True, **kwargs)
        return completion_response_to_chat_response(completion_response)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        prompt = self.messages_to_prompt(messages)
        completion_response = self.stream_complete(prompt, formatted=True, **kwargs)
        return stream_completion_response_to_chat_response(completion_response)


class OpenVINOGenAIContinousPipeline(CustomLLM):
    r"""
    OpenVINO GenAI ContinousPipeline.

    Examples:
        `pip install llama-index-llms-openvino-genai`

        ```python
        from llama_index.llms.openvino_genai import OpenVINOGenAIContinousPipeline

        llm = OpenVINOGenAIContinousPipeline(
            model_path=model_path,
            device="CPU",
        )

        response = llm.complete("What is the meaning of life?")
        print(str(response))
        ```

    """

    model_path: str = Field(
        default=None,
        description=("The model path to use from local. "),
    )
    system_prompt: str = Field(
        default="",
        description=(
            "The system prompt, containing any extra instructions or context. "
            "The model card on HuggingFace should specify if this is needed."
        ),
    )
    query_wrapper_prompt: PromptTemplate = Field(
        default=PromptTemplate("{query_str}"),
        description=(
            "The query wrapper prompt, containing the query placeholder. "
            "The model card on HuggingFace should specify if this is needed. "
            "Should contain a `{query_str}` placeholder."
        ),
    )
    device: str = Field(
        default="auto", description="The device to use. Defaults to 'auto'."
    )
    is_chat_model: bool = Field(
        default=False,
        description=(
            LLMMetadata.model_fields["is_chat_model"].description
            + " Be sure to verify that you either pass an appropriate tokenizer "
            "that can convert prompts to properly formatted chat messages or a "
            "`messages_to_prompt` that does so."
        ),
    )

    config: Any = Field(
        default=None,
        description=("The LLM generation configurations."),
        exclude=True,
    )
    max_num_batched_tokens: int = Field(
        default=8192,
        description=("Maximum number of tokens to batch."),
    )
    _pipe: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _streamer: Any = PrivateAttr(default=None)
    _stop_scheduler: bool = PrivateAttr(default=False)
    _scheduler_thread: Any = PrivateAttr(default=None)
    _handles: Any = PrivateAttr(default_factory=list)
    _request_id: int = PrivateAttr(default=0)

    def __init__(
        self,
        model_path: str,
        config: Optional[dict] = {},
        tokenizer: Optional[Any] = None,
        device: Optional[str] = "CPU",
        query_wrapper_prompt: Union[str, PromptTemplate] = "{query_str}",
        is_chat_model: Optional[bool] = False,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: str = "",
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        max_num_batched_tokens: Optional[int] = 8192,
        **kwargs: Any,
    ) -> None:

        """Initialize params."""
        import openvino.properties.hint as hints
        config = {hints.performance_mode: hints.PerformanceMode.THROUGHPUT,
              hints.num_requests: "4"}
        ov_config = {}
        ov_config["scheduler_config"] = get_scheduler_config_genai({'max_num_batched_tokens': max_num_batched_tokens, 'dynamic_split_fuse': False, 'use_cache_eviction': False, 'cache_eviction_config': {'start_size': 32, 'recent_size': 128, 'max_cache_size': 4096, 'aggregation_mode': 'NORM_SUM'}})
        pipe = openvino_genai.ContinuousBatchingPipeline(model_path, device=device, properties=config, **ov_config)

        config = pipe.get_config()

        tokenizer = tokenizer or pipe.get_tokenizer()
        
        if isinstance(query_wrapper_prompt, str):
            query_wrapper_prompt = PromptTemplate(query_wrapper_prompt)
        
        messages_to_prompt = messages_to_prompt or self._tokenizer_messages_to_prompt
        _stop_scheduler = False
        super().__init__(
            tokenizer=tokenizer,
            model_path=model_path,
            device=device,
            query_wrapper_prompt=query_wrapper_prompt,
            is_chat_model=is_chat_model,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            _stop_scheduler=_stop_scheduler
        )
        self._stop_scheduler = False
        self._pipe = pipe
        self._tokenizer = tokenizer
        self._handles = []
        self._request_id = 0
        self.config = config

        self._scheduler_thread = Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()
        
    
    def _scheduler_loop(self):
        while not self._stop_scheduler:
            try:
                if self._pipe.has_non_finished_requests():
                    self._pipe.step()
            except Exception as e:
                print(f"[Scheduler thread error] {e}")
                # 可以选择休眠一下，避免高速循环打印
                time.sleep(1)
            
    @classmethod
    def class_name(cls) -> str:
        return "OpenVINO_GenAI_LLM"

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
            model_name=self.model_path,
            is_chat_model=self.is_chat_model,
        )
    
    def stop(self):
        self._stop_scheduler = True
        if self._scheduler_thread is not None:
            self._scheduler_thread.join()

    def _tokenizer_messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str:
        """Use the tokenizer to convert messages to prompt. Fallback to generic."""
        if hasattr(self._tokenizer, "apply_chat_template"):
            messages_dict = [
                {"role": message.role.value, "content": message.content}
                for message in messages
            ]
            return (
                self._tokenizer.apply_chat_template(
                    messages_dict, add_generation_prompt=True
                )
                if isinstance(self._tokenizer, openvino_genai.Tokenizer)
                else self._tokenizer.apply_chat_template(
                    messages_dict, tokenize=False, add_generation_prompt=True
                )
            )

        return generic_messages_to_prompt(messages)

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        """Completion endpoint."""
        full_prompt = prompt
        if not formatted:
            if self.query_wrapper_prompt:
                full_prompt = self.query_wrapper_prompt.format(query_str=prompt)
            if self.completion_to_prompt:
                full_prompt = self.completion_to_prompt(full_prompt)
            elif self.system_prompt:
                full_prompt = f"{self.system_prompt} {full_prompt}"

        if not isinstance(self._tokenizer, openvino_genai.Tokenizer):
            inputs = self._tokenizer(
                full_prompt, add_special_tokens=False, return_tensors="np"
            )
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            full_prompt = openvino_genai.TokenizedInputs(
                ov.Tensor(input_ids), ov.Tensor(attention_mask)
            )

        tokens = self._pipe.generate(full_prompt, self.config, **kwargs)
        if not isinstance(self._tokenizer, openvino_genai.Tokenizer):
            completion_tokens = tokens[0][inputs["input_ids"].size(1) :]
            completion = self._tokenizer.decode(
                completion_tokens, skip_special_tokens=True
            )
        else:
            completion = tokens
        return CompletionResponse(text=completion, raw={"model_output": tokens})
    

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = self.messages_to_prompt(messages)
        completion_response = self.complete(prompt, formatted=True, **kwargs)
        return completion_response_to_chat_response(completion_response)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        prompt = self.messages_to_prompt(messages)
        completion_response = self.stream_complete(prompt, formatted=True, **kwargs)
        return stream_completion_response_to_chat_response(completion_response)

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        req_id = self._request_id
        self._request_id += 1
        full_prompt = prompt
        if not formatted:
            if self.query_wrapper_prompt:
                full_prompt = self.query_wrapper_prompt.format(query_str=prompt)
            if self.system_prompt:
                full_prompt = f"{self.system_prompt} {full_prompt}"

        if not isinstance(self._tokenizer, openvino_genai.Tokenizer):
            inputs = self._tokenizer(
                full_prompt, add_special_tokens=False, return_tensors="np"
            )
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            full_prompt = openvino_genai.TokenizedInputs(
                ov.Tensor(input_ids), ov.Tensor(attention_mask)
            )
        handle = self._pipe.add_request(req_id, full_prompt, self.config, **kwargs)
        self._handles.append(handle)
        text = ""
        while True:
            try:
                result = None
                if handle.can_read():
                    result = handle.read()
                if result is None or len(result) == 0:
                    continue
                
                token_ids = result[0].generated_ids

                if len(token_ids) == 0:
                    if result[0].finish_reason!= openvino_genai.GenerationFinishReason.NONE:
                        break
                    continue

                token_text = self._tokenizer.decode(token_ids)

                text += token_text
                yield CompletionResponse(text=text, delta=token_text, raw={"request_id": req_id, "result": result})

                if result[0].finish_reason!= openvino_genai.GenerationFinishReason.NONE:
                    break
            except Exception as e:
                print(f"[astream_complete error] {e}")
                break
