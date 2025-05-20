from pathlib import Path
import json, os
from tinygrad.nn.state import safe_load, torch_load, gguf_load
from tinygrad import Tensor, dtypes, Device

def load(fn:str):
  if fn.endswith('.index.json'):
    with open(fn) as fp: weight_map = json.load(fp)['weight_map']
    parts = {n: load(str(Path(fn).parent / Path(n).name)) for n in set(weight_map.values())}
    return {k: parts[n][k] for k, n in weight_map.items()}
  elif fn.endswith(".gguf"):
    gguf_tensor = Tensor.empty(os.stat(fn).st_size, dtype=dtypes.uint8, device=f"disk:{fn}").to(Device.DEFAULT)
    return gguf_load(gguf_tensor)[1]
  elif fn.endswith(".safetensors"):
    return safe_load(fn)
  else:
    return torch_load(fn)
