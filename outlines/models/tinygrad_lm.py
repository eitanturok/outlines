import dataclasses
from typing import Optional, Generator, Tuple, Union, List, Iterator

from tinygrad import Tensor, Device, dtypes
from outlines.models.olmoe import build_olmoe_model, build_olmoe_tokenizer, sample
from outlines.generate.api import GenerationParameters, SamplingParameters
from outlines.processors import OutlinesLogitsProcessor

from .transformers import TransformerTokenizer

KVCacheType = Tuple[Tuple["tinygrad.Tensor", "tinygrad.Tensor"], ...]


class TinygradTokenizer(TransformerTokenizer):
    """Represents a tokenizer for models in the `transformers` library using `tinygrad`."""
    def encode(self, prompt: Union[str, List[str]], **kwargs) -> Tuple["tinygrad.Tensor", "tinygrad.Tensor"]:
        input_ids, attention_mask = super().encode(prompt, **kwargs)
        return Tensor(input_ids.numpy()), Tensor(attention_mask.numpy())
    def decode(self, token_ids: "tinygrad.Tensor") -> List[str]:
        token_ids = [x.cast(dtypes.uint).tolist() for x in token_ids]
        return super().decode(token_ids)


class TinygradLM:
    """Represents a `tinygrad` model."""

    def __init__(
        self,
        model: "PreTrainedModel",
        tokenizer: "PreTrainedTokenizer",
    ):
        self.model = model
        self.tokenizer = TinygradTokenizer(tokenizer)

    def generate(
        self,
        prompts: Union[str, List[str]],
        generation_parameters: "GenerationParameters",
        logits_processor,
        sampling_parameters: "SamplingParameters",
    ) -> str:
        """Generate text using `transformers`.

        Parameters
        ----------
        prompts
            A prompt or list of prompts.
        generation_parameters
            An instance of `GenerationParameters` that contains the prompt,
            the maximum number of tokens, stop sequences and seed. All the
            arguments to `SequenceGeneratorAdapter`'s `__cal__` method.
        logits_processor
            The logits processor to use when generating text.
        sampling_parameters
            An instance of `SamplingParameters`, a dataclass that contains
            the name of the sampler to use and related parameters as available
            in Outlines.

        Returns
        -------
        The generated text
        """
        streamer = self.stream(
            prompts, generation_parameters, logits_processor, sampling_parameters
        )
        return "".join(list(streamer))

    def stream(
        self,
        prompts: Union[str, List[str]],
        generation_parameters: "GenerationParameters",
        logits_processor,
        sampling_parameters: "SamplingParameters",
    ) -> Iterator[str]:
        """Temporary stream stand-in which implements stream() signature
        and equivalent behaviour but isn't yielded until generation completes.

        Parameters
        ----------
        prompts
            A prompt or list of prompts.
        generation_parameters
            An instance of `GenerationParameters` that contains the prompt,
            the maximum number of tokens, stop sequences and seed. All the
            arguments to `SequenceGeneratorAdapter`'s `__cal__` method.
        logits_processor
            The logits processor to use when generating text.
        sampling_parameters
            An instance of `SamplingParameters`, a dataclass that contains
            the name of the sampler to use and related parameters as available
            in Outlines.

        Returns
        -------
        The generated text.
        """

        max_tokens, stop_at, seed = dataclasses.astuple(generation_parameters)
        sampler, num_samples, top_p, top_k, temperature = dataclasses.astuple(
            sampling_parameters
        )
        if max_tokens is None:
            max_tokens = int(1e9)

        if not isinstance(prompts, str):
            raise NotImplementedError(
                "tinygrad does not support batch inference."
            )
        if sampler == "beam_search":
            raise NotImplementedError(
                "tinygrad does not support Beam Search."
            )
        if num_samples != 1:
            raise NotImplementedError(
                "tinygrad does not allow to take several samples."
            )
        if top_k is not None:
            raise NotImplementedError("tinygrad does not support top_k.")
        if seed is not None:
            raise NotImplementedError("tinygrad does not support seed.")
        if stop_at is not None:
            raise NotImplementedError("tinygrad does not support stop_at.")

        generate_kwargs = {
            "temp": temperature,
            "top_p": top_p,
            "sampler": sampler,
            "logits_processor": logits_processor,
        }

        # Adapted from
        # https://github.com/ml-explore/mlx-examples/blob/4872727/llms/mlx_lm/utils.py#L267
        prompt_tokens, _ = self.tokenizer.encode(prompts) # returns input_ids, attention_mask
        ic(prompts, prompt_tokens)

        tokens = []

        for (token, prob), n in zip(
            self.generate_step(prompt_tokens, **generate_kwargs),
            range(max_tokens),
        ):
            if token == self.tokenizer.eos_token_id:
                break
            yield self.tokenizer.decode([token])

        yield self.tokenizer.decode([token])

    def generate_step(
        self,
        prompt: "tinygrad.Tensor",
        temp: Optional[float],
        top_p: Optional[float],
        sampler: str,
        logits_processor: "OutlinesLogitsProcessor",
    ) -> Generator[Tuple[int, float], None, None]:
        """
        Adapted from
        https://github.com/ml-explore/mlx-examples/blob/4872727/llms/mlx_lm/utils.py#L129

        A generator producing token ids based on the given prompt from the model.

        Parameters
        ----------
        prompt
            The input prompt.
        temp
            The temperature for sampling, if 0 the argmax is used.
        top_p
            Nulceus sampling, higher means model considers more less likely words.
        sampler
            The sampler string defined by SequenceGeneratorAdapter
        logits_processor
            Augment logits before sampling.
        """

        temperature: float = temp or 1.0

        # kv cache contains processed input IDs, we pass the unprocessed inputs and cache to model()
        unprocessed_input_ids = prompt
        generated_ids: List[int] = []
        start_pos = unprocessed_input_ids.shape[1] - 1
        ic(unprocessed_input_ids.shape, unprocessed_input_ids.numpy())

        while True:

            logits = self.model(unprocessed_input_ids[:, start_pos:], start_pos, return_logits=True)
            logits = logits[:, -1, :]

            if logits_processor is not None:
                # convert to logits_processor 1d expectation, apply, then convert back
                logits_1d = logits.reshape(-1)
                logits_1d = logits_processor(generated_ids, logits_1d)
                logits = logits_1d.reshape(1, -1)

            if sampler == "greedy":
                new_token_single = sample(logits, 0, 0, 0, 0, 0)
            elif sampler == 'multinomial':
                assert top_p is not None
                new_token_single = sample(logits, temperature, 0, top_p, 0, 0)
            else:
                raise ValueError(f"Invalid tinygrad sampler: `{sampler}`")
            new_token = new_token_single.item()
            yield new_token

            generated_ids.append(new_token)
            unprocessed_input_ids = new_token_single
            start_pos += 1

def tinygradlm(
    model_name: str,
    device: Optional[str] = None,
    tokenizer_config: dict = {},
    model_config: dict = {},
    adapter_path: Optional[str] = None,
    lazy: bool = False,
):
    """Instantiate a model from the `tinygrad` library and its tokenizer.

    Signature adapted from
    https://github.com/ml-explore/mlx-examples/blob/4872727/llms/mlx_lm/utils.py#L422

    Parameters
    ----------
    Args:
        path_or_hf_repo (Path): The path or the huggingface repository to load the model from.
        device: (str, optional): The device(s) on which the model should be loaded. This
            overrides the `device_map` entry in `model_kwargs` when provided.
        tokenizer_config (dict, optional): Configuration parameters specifically for the tokenizer.
            Defaults to an empty dictionary.
        model_config(dict, optional): Configuration parameters specifically for the model.
            Defaults to an empty dictionary.
        adapter_path (str, optional): Path to the LoRA adapters. If provided, applies LoRA layers
            to the model. Default: ``None``.
        lazy (bool): If False eval the model parameters to make sure they are
            loaded in memory before returning, otherwise they will be loaded
            when needed. Default: ``False``

    Returns
    -------
    A `TinyGrad` model instance.

    """
    try:
        from tinygrad import nn
    except ImportError:
        raise ImportError(
            "The `tinygrad` library needs to be installed in order to use `tinygrad` models."
        )

    if device is None: device = Device.DEFAULT
    model = build_olmoe_model(device)
    tokenizer = build_olmoe_tokenizer()
    return TinygradLM(model, tokenizer)
