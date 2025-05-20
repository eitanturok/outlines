import os
from pathlib import Path
from typing import Dict, Union, Optional

from .transformers import TransformerTokenizer

from .llama import Transformer, convert_from_huggingface, fix_bf16
from .llama3 import load
from tinygrad import nn, Tensor
from tinygrad.helpers import fetch
from tinygrad.nn.state import load_state_dict

from transformers import PreTrainedTokenizer

from outlines.generate.api import GenerationParameters, SamplingParameters
from outlines.processors import OutlinesLogitsProcessor


MODELS = {
    "32B": {
        "model_params": {"dim": 5120, "n_heads": 40, "n_kv_heads": 8, "n_layers": 64, "norm_eps": 1e-5, "rope_theta": 1000000, "vocab_size": 152064, "hidden_dim": 27648},
        "total_num_weights": 17,
        "tokenizer": "Qwen/QwQ-32B-Preview"
    }
}

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
        self.tokenizer = TransformerTokenizer(
            tokenizer._tokenizer
        )  # _tokenizer is HF Tokenizer


def download_weights(total_num_weights:int) -> Path:
    model = fetch("https://huggingface.co/Qwen/QwQ-32B-Preview/resolve/main/model.safetensors.index.json?download=true", "model.safetensors.index.json", subdir=(subdir:="qwq_32b_preview"))

    for i in range(1, total_num_weights + 1):
        filename = f"model-{i:05d}-of-{total_num_weights:05d}.safetensors"
        fetch(f"https://huggingface.co/Qwen/QwQ-32B-Preview/resolve/main/{filename}?download=true", filename, subdir=subdir)

    return Path(os.path.dirname(model))


def load_model(model_path:Path, model_params:Dict[str, Union[int, float]]) -> Transformer:
    # build model
    model = Transformer(**model_params, linear=nn.Linear)

    # update layers to add bias
    updated_layers = []
    for layer in model.layers:
        head_dim = model_params["dim"] // model_params["n_heads"]
        layer.attention.wq = nn.Linear(model_params["dim"], model_params["n_heads"] * head_dim, bias=True)
        layer.attention.wk = nn.Linear(model_params["dim"], model_params["n_kv_heads"] * head_dim, bias=True)
        layer.attention.wv = nn.Linear(model_params["dim"], model_params["n_kv_heads"] * head_dim, bias=True)
        updated_layers.append(layer)
    model.layers = updated_layers

    # load weights
    weights = fix_bf16(convert_from_huggingface(load(str(model_path / "model.safetensors.index.json")), model_params["n_layers"], model_params["n_heads"], model_params["n_kv_heads"], permute_layers=False))

    # replace weights in model
    load_state_dict(model, weights, strict=False, consume=True)
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

    from transformers import AutoTokenizer
    model_info = MODELS["32B"]
    model_path = download_weights(model_info["total_num_weights"])
    model = load_model(model_path, model_info["model_params"])
    tokenizer = AutoTokenizer.from_pretrained(model_info["tokenizer"])
    return TinygradLM(model, tokenizer)
