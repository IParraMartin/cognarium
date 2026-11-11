# Install

cognarium needs Python 3.11 or later and PyTorch 2.9 or later.

```bash
uv add cognarium
```

or, with pip:

```bash
pip install cognarium
```

The base install includes everything `cg.load("gpt2")` and the dataset registry need,
such as Transformers, the Hugging Face Hub client and pooch.

## From source

```bash
git clone https://github.com/IParraMartin/cognarium
cd cognarium
uv sync --extra dev
uv run pytest
```

## Extras

| Extra | Adds |
|---|---|
| `dev` | The test suite's tools, and the libraries its oracle tests compare against: SciPy, scikit-learn, dit and rsatoolbox |
| `docs` | MkDocs Material and mkdocstrings, to build this site |
| `vision` | Pillow and torchvision, for images and the processors of vision-language models |
| `video` | PyAV, to read video files |
| `all` | `vision` and `video` |

## Devices

`device="auto"` picks CUDA, then Apple MPS, then the CPU. `dtype="auto"` picks a
precision the device supports. Metrics that need `float64`, or that have known problems
on MPS, always compute on the CPU, whatever device the model uses.

```python
import cognarium as cg

subject = cg.load("gpt2", device="auto")
subject.device
```
