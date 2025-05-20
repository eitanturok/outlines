import os
from pathlib import Path
from typing import Dict, Union, Optional

from tinygrad.helpers import fetch
from tinygrad.nn.state import torch_load, load_state_dict

from transformers import PreTrainedTokenizer

from outlines.generate.api import GenerationParameters, SamplingParameters
from outlines.processors import OutlinesLogitsProcessor

from .transformers import TransformerTokenizer
from .gpt2 import Transformer, MODEL_PARAMS


class TinygradLM:
    """
    Represents an `tinygrad_lm` model
    """

    def __init__(
        self,
        model: "tinygrad.Tensor",
        tokenizer: "PreTrainedTokenizer",
    ):
        self.model = model
        self.mlx_tokenizer = tokenizer  # returns mlx tensors, used for encode()
        self.tokenizer = tokenizer


def build_gpt2_tokenizer():
    import tiktoken
    return tiktoken.get_encoding("gpt2")

def build_gpt2_model(model_size):
    model = Transformer(**MODEL_PARAMS[model_size])
    weights = torch_load(fetch(f'https://huggingface.co/{model_size}/resolve/main/pytorch_model.bin'))

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

    model = build_gpt2_model("gpt2")
    tokenizer = build_gpt2_tokenizer()

    return TinygradLM(model, tokenizer)
