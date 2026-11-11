# Multimodal stimuli

A stimulus can carry images, video and sound, alone or mixed, along with text and
features. A Hugging Face model that reads media, such as a vision-language or an omni
model, sees them in its prompt and learns from them in context, as a text model learns
from text. The same trials also run on the other learners: a text model reads the
stimulus's text, and an exemplar or Bayesian learner reads its features. A colour can
therefore be a chip image for one model, RGB values for another and CIELAB coordinates
for an exemplar learner, in one task scored by the same measures.

Media need the extras for their kind:

```bash
pip install "cognarium[vision]"   # images, and the processors of vision-language models
pip install "cognarium[audio]"    # reading and resampling sound
pip install "cognarium[video]"    # reading video files
```

## Media

| Kind | Built from | Notes |
|---|---|---|
| `cg.media.Image` | A path, a PIL image, an array of shape (height, width, channels) or a tensor of shape (channels, height, width) | Float arrays are read from 0 to 1 |
| `cg.media.Video` | A path, frames of shape (frames, height, width, channels), or a list of images | `fps=` places frames in time for models that use it |
| `cg.media.Audio` | A path, or a waveform with its `sampling_rate` | Averaged to mono, and resampled to the rate the model expects |

A `Stimulus` takes any of these, or anything they take, in its `image`, `video` and
`audio` fields. A path is read only when a model needs it. Each medium has a digest of
its content, which results and run keys record in place of the content, so changing an
image starts a new run.

```python
import numpy as np
import cognarium as cg

red = cg.media.swatch([0.9, 0.1, 0.1], size=32)
beep = cg.media.Audio(np.sin(np.linspace(0, 880 * np.pi, 16000)).astype(np.float32), 16000)
stimulus = cg.Stimulus(text="a red square", image=red, audio=beep)
sorted(stimulus.modalities)
```

`cg.media.swatch` draws a flat square of one colour, and `cg.media.lab_to_srgb`
converts CIELAB coordinates to sRGB under the D65 white point, which together turn a
table of colours into chips.

## Placing media in prompts

The prompt template places each field of a stimulus: `{text}` for its text, and
`{image}`, `{video}` and `{audio}` for its media. Media the template doesn't place go
first, so a stimulus with text and an image shows both under the default template.
A text model skips them, and refuses a template that places media, since text can't
hold an image.

```python
template = cg.PromptTemplate(stimulus="Color: {image} -> Label:", response="")
query = cg.Query(
    trial=cg.Trial(cg.Stimulus(image=red), options=("wug", "dax")),
    context=(cg.Observation(cg.Stimulus(image=cg.media.swatch([0.1, 0.1, 0.9])), "dax"),),
    template=template
)
print(query.describe())
```

`describe()` and `paradigm.preview()` write each medium as its kind and the start of its
digest, as in `<image 3f2a9c1b>`.

## Loading a model that reads media

```python
subject = cg.load(
    "Qwen/Qwen3.5-4B-Base",
    device="auto",
    multimodal=True,
    processor_kwargs={"images_kwargs": {"size": {"shortest_edge": 1024, "longest_edge": 1024}}}
)
sorted(subject.modalities)
```

`multimodal=True` loads the model with its processor. Without it, a vision-language
checkpoint loads as a text model. A model object you built yourself goes to
`cg.load(model, processor=processor)`. `processor_kwargs` goes to every processor call;
here it shrinks each image to 32 by 32 pixels, one token on Qwen models, which a flat
colour chip doesn't need more of.

The subject finds the text that stands for each kind of medium in a prompt, which the
processor expands into the model's media tokens: from the chat template if there is one,
otherwise from the processor's own tokens. `media_tokens=` sets it for a processor where
neither works, and `chat=True` sends each prompt through the chat template, as
instruction-tuned models expect.

Options are scored as text after the prompt, exactly as for a text model, so the
logits, the hidden states of the language model's blocks and the logit lens are all
recorded as usual. Multi-token options continue from the prompt's cache. Some models
place media in positions that may not survive that, so the first such query is also
scored with each option run through the whole prompt. If the two disagree, every later
query is scored that way, with a warning.

## Iterated learning with colour chips

A `cg.tasks.Mapping` takes one image, video or sound per item, with features or without.
Here chains pass names for chips of eight hues down generations of a model that sees the
chips, while an exemplar learner reads the same items' CIELAB coordinates.

```python
angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
lab = np.stack([np.full(8, 60.0), 60 * np.cos(angles), 60 * np.sin(angles)], axis=1)
chips = [cg.media.swatch(color) for color in cg.media.lab_to_srgb(lab)]
task = cg.tasks.Mapping(
    lab.tolist(),
    ("wug", "dax", "fep"),
    images=chips,
    template=template,
    sigma=30.0
)
paradigm = cg.paradigms.IteratedLearning(task, generations=2, chains=2, n_prev_examples=4)
results = cg.run(paradigm, subject, record=["logits"])
results.metrics("ib_complexity", "efficiency_loss", by=["generation"])
```

```python
baseline = cg.run(paradigm, cg.subjects.ExemplarLearner())
baseline.metrics("ib_complexity", "efficiency_loss", by=["generation"])
```

`examples/color_naming.py --show image` runs the 330 World Color Survey chips this way,
and without `--show image` the same chains read each chip's RGB values as text.

## Mixing modalities

A stimulus can hold several media, and the template places each one. A model reads
whichever its processor handles, and a run fails before it starts if the template places
a kind the model can't read.

```python
tones = [
    cg.media.Audio(np.sin(np.linspace(0, frequency * np.pi, 8000)).astype(np.float32), 16000)
    for frequency in (200, 400, 600, 800)
]
paired = cg.tasks.Mapping(
    None,
    ("wug", "dax"),
    images=chips[:4],
    audio=tones,
    template=cg.PromptTemplate(stimulus="Sight {image} sound {audio} -> Label:", response=""),
    points=[[frequency] for frequency in (200, 400, 600, 800)]
)
cg.run(cg.paradigms.IteratedLearning(paired, generations=2, chains=1, n_prev_examples=2), subject).to_pandas().head()
```

Items with no features need `points=` for the naming measures, which place each item in
a space where distances mean something.

## What it can't capture

A WCS chip is a Munsell surface seen under daylight, and a chip image is its sRGB value
on whatever screen the training images came from. Chips outside the sRGB gamut are
clipped to its edge, so a few saturated chips are shown as colours a little different
from their CIELAB coordinates, which the measures still use. The model sees a flat square
with no surround, unlike the mounted chips in the survey.

Each prompt goes through the processor alone, and its media go through the model's
encoder with every query that shows them. In iterated learning that means each example
image is encoded again for every item a generation names. Keeping chips small keeps
this cheap.
