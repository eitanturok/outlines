import dataclasses
import inspect
from typing import Optional, Generator, Tuple, Union, List, Iterator

import torch
from tinygrad import Tensor, Device, nn, dtypes, Variable
from tinygrad.nn.state import load_state_dict

from outlines.models.tokenizer import Tokenizer
from outlines.generate.api import GenerationParameters, SamplingParameters
from outlines.processors import OutlinesLogitsProcessor

from .gpt2 import Transformer, MODEL_PARAMS
from .transformers import TransformerTokenizer
from .transformers import get_llama_tokenizer_types

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

    def forward(
        self,
        input_ids: "tinygrad.Tensor",
        attention_mask: "tinygrad.Tensor",
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple["tinygrad.Tensor", Optional[KVCacheType]]:
        """Compute a forward pass through the transformer model.

        Parameters
        ----------
        input_ids
            The input token ids.  Must be one or two dimensional.
        attention_mask
            The attention mask.  Must be one or two dimensional.
        past_key_values
            A tuple of tuples containing the cached key and value tensors for each
            attention head.

        Returns
        -------
        The computed logits and the new cached key and value tensors.

        """
        try:
            from tinygrad import Variable, Tensor
        except ImportError:
            ImportError(
                "The `torch` library needs to be installed to use `transformers` models."
            )
        assert 0 < input_ids.ndim < 3

        if past_key_values:
            input_ids = input_ids[..., -1].unsqueeze(-1)

        # with torch.inference_mode():
        #     output = self.model(
        #         input_ids,
        #         attention_mask=attention_mask,
        #         return_dict=True,
        #         output_attentions=False,
        #         output_hidden_states=False,
        #         past_key_values=past_key_values,
        #     )

        n_tokens = input_ids.shape[-1]
        ic(input_ids.shape)
        Tensor.training = False
        Tensor.no_grad = True
        _, logits = self.model(input_ids, Variable("start_pos", 1, n_tokens+1).bind(1), return_logits=True)
        return logits, None

    def __call__(
        self,
        input_ids: "tinygrad.Tensor",
        attention_mask: "tinygrad.Tensor",
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple["tinygrad.Tensor", Optional["tinygrad.Tensor"]]:
        logits, kv_cache = self.forward(input_ids, attention_mask, past_key_values)
        next_token_logits = logits[..., -1, :]
        ic(logits.shape, next_token_logits.shape)
        return next_token_logits, kv_cache

    def generate(
        self,
        prompts: Union[str, List[str]],
        generation_parameters: GenerationParameters,
        logits_processor: Optional["OutlinesLogitsProcessor"],
        sampling_parameters: SamplingParameters,
    ) -> Union[str, List[str], List[List[str]]]:
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
        if isinstance(prompts, str):
            # convert to 2d
            input_ids, attention_mask = self.tokenizer.encode([prompts])
        else:
            input_ids, attention_mask = self.tokenizer.encode(prompts)

        device = nn.state.get_parameters(self.model)[0].device
        inputs = {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
        }
        if (
            "attention_mask"
            not in inspect.signature(self.model.forward).parameters.keys()
        ):
            del inputs["attention_mask"]

        generation_kwargs = self._get_generation_kwargs(
            prompts,
            generation_parameters,
            logits_processor,
            sampling_parameters,
        )
        generated_ids = self._generate_output_seq(prompts, inputs, **generation_kwargs)
        ic(generated_ids.shape)

        # if single str input and single sample per input, convert to a 1D output
        if isinstance(prompts, str):
            generated_ids = generated_ids.squeeze(0)

        return self._decode_generation(generated_ids)

    def stream(
        self,
        prompts: Union[str, List[str]],
        generation_parameters: GenerationParameters,
        logits_processor: Optional["OutlinesLogitsProcessor"],
        sampling_parameters: SamplingParameters,
    ) -> Iterator[Union[str, List[str]]]:
        """
        Temporary stream stand-in which implements stream() signature
        and equivalent behaviour but isn't yielded until generation completes.

        TODO: implement following completion of https://github.com/huggingface/transformers/issues/30810
        """
        if isinstance(prompts, str):
            # convert to 2d
            input_ids, attention_mask = self.tokenizer.encode([prompts])
        else:
            input_ids, attention_mask = self.tokenizer.encode(prompts)
        device = nn.state.get_parameters(self.model)[0].device
        inputs = {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
        }
        if (
            "attention_mask"
            not in inspect.signature(self.model.forward).parameters.keys()
        ):
            del inputs["attention_mask"]

        generation_kwargs = self._get_generation_kwargs(
            prompts,
            generation_parameters,
            logits_processor,
            sampling_parameters,
        )
        generated_ids = self._generate_output_seq(prompts, inputs, **generation_kwargs)

        # if single str input and single sample per input, convert to a 1D output
        if isinstance(prompts, str):
            generated_ids = generated_ids.squeeze(0)

        for i in range(generated_ids.size(-1)):
            output_group_ids = generated_ids.select(-1, i).unsqueeze(-1)
            yield self._decode_generation(output_group_ids)

    def _get_generation_kwargs(
        self,
        prompts: Union[str, List[str]],
        generation_parameters: GenerationParameters,
        logits_processor: Optional["OutlinesLogitsProcessor"],
        sampling_parameters: SamplingParameters,
    ) -> dict:
        """
        Convert outlines generation parameters into model.generate kwargs
        """
        from transformers import GenerationConfig, LogitsProcessorList, set_seed

        max_new_tokens, stop_at, seed = dataclasses.astuple(generation_parameters)
        sampler, num_samples, top_p, top_k, temperature = dataclasses.astuple(
            sampling_parameters
        )
        if max_new_tokens is None:
            max_new_tokens = int(2**30)

        # global seed, not desirable
        if seed is not None:
            set_seed(seed)

        if logits_processor is not None:
            logits_processor_list = LogitsProcessorList([logits_processor])
        else:
            logits_processor_list = None

        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            stop_strings=stop_at,
            num_return_sequences=(num_samples or 1),
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            do_sample=(sampler == "multinomial"),
            num_beams=(num_samples if sampler == "beam_search" else 1),
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        return dict(
            logits_processor=logits_processor_list,
            generation_config=generation_config,
            tokenizer=self.tokenizer.tokenizer,
        )

    def _generate_output_seq(
        self, prompts, inputs, generation_config, **generation_kwargs
    ):
        toks = inputs["input_ids"]
        attention_mask = None # inputs["attention_mask"]
        past_key_values = None
        temperature = generation_config.temperature
        start_pos = 0
        ic(generation_config, generation_kwargs)

        for _ in range(max_length):
            if batch_size == 1 and len(toks[0][start_pos:]) == 1:
                tokens = Variable("tokens", 0, VOCAB_SIZE).bind(toks[0][start_pos])
            else:
                tokens = Tensor([x[start_pos:] for x in toks])
            logits, _ = self.model(tokens, Variable("start_pos", 1 if start_pos else 0, MAX_CONTEXT-1).bind(start_pos), temperature, return_logits=True)
            start_pos = len(toks[0])
            tok = generation_kwargs['logits_processor'](logits)
            for i,t in enumerate(tok): toks[i].append(t)


        for _ in range(generation_config.max_new_tokens):
            logits, _ = self(input_ids, attention_mask, past_key_values)
            token = generation_kwargs['logits_processor'](logits)
            input_ids.append(token)


        # encoder-decoder returns output_ids only, decoder-only returns full seq ids
        # if self.model.config.is_encoder_decoder:
        #     generated_ids = output_ids
        # else:
        #     generated_ids = output_ids[:, input_ids.shape[1] :]
        generated_ids = input_ids
        ic(generated_ids)

        # if batch list inputs AND multiple samples per input, convert generated_id to 3D view
        num_samples = generation_config.num_return_sequences or 1

        if num_samples > 1 and isinstance(prompts, list):
            batch_size = input_ids.size(0)
            num_return_sequences = generation_config.num_return_sequences or 1
            generated_ids = generated_ids.view(batch_size, num_return_sequences, -1)

        return generated_ids

    def _decode_generation(self, generated_ids: "tinygrad.Tensor"):
        if len(generated_ids.shape) == 1:
            return self.tokenizer.decode([generated_ids])[0]
        elif len(generated_ids.shape) == 2:
            return self.tokenizer.decode(generated_ids)
        elif len(generated_ids.shape) == 3:
            return [
                self.tokenizer.decode(generated_ids[i])
                for i in range(len(generated_ids))
            ]
        else:
            raise TypeError(
                f"Generated outputs aren't 1D, 2D or 3D, but instead are {generated_ids.shape}"
            )

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

        def sample(logits: "tinygrad.Tensor") -> Tuple["tinygrad.Tensor", float]:
            softmax_logits = logits.softmax()

            if temperature == 0.0 or sampler == "greedy":
                token = logits.argmax(-1)
            else:
                raise ValueError(f"Invalid mlx-lm sampler: `{sampler}`")

            prob = softmax_logits[0, token]
            return token, prob

        # cache = mlx_lm.models.cache.make_prompt_cache(self.model)

        # kv cache contains processed input IDs, we pass the unprocessed inputs and cache to model()
        unprocessed_input_ids = prompt
        generated_ids: List[int] = []

        while True:
            # logits = self.model(unprocessed_input_ids[None], cache=cache)
            logits = self.model(unprocessed_input_ids[None])
            logits = logits[:, -1, :]

            if logits_processor is not None:
                # convert to logits_processor 1d expectation, apply, then convert back
                logits_1d = logits.reshape(-1)
                logits_1d = logits_processor(generated_ids, logits_1d)
                logits = logits_1d.reshape(1, -1)

            new_token_single, prob = sample(logits)
            new_token = new_token_single.item()
            yield new_token, prob

            generated_ids.append(new_token)
            unprocessed_input_ids = new_token_single


def build_gpt2_tokenizer():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return tokenizer

def build_gpt2_model(model_size, device):
    model = Transformer(**MODEL_PARAMS[model_size])
    weights = Tensor.from_url(f'https://huggingface.co/{model_size}/resolve/main/pytorch_model.bin').to(device)
    weights = nn.state.torch_load(weights)

    # special treatment for the Conv1D weights we need to transpose
    transposed = ('attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight')
    for k in weights:
      if k.endswith(transposed):
        weights[k] = weights[k].T
    # lm head and wte are tied
    weights['lm_head.weight'] = weights['wte.weight']

    load_state_dict(model, weights)
    return model

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

    if device is None:
        device = Device.DEFAULT

    model = build_gpt2_model("gpt2", device)
    tokenizer = build_gpt2_tokenizer()

    return TinygradLM(model, tokenizer)
