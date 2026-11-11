"""Media, stimuli that carry them, prompt templates that place them, and media tasks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

import cognarium as cg
from cognarium.media import Image, Video, as_media


def red(size=4):
    return np.tile(np.array([255, 0, 0], dtype=np.uint8), (size, size, 1))


def test_an_image_has_one_digest_for_the_same_pixels_in_any_form(tmp_path):
    from PIL import Image as PILImage

    array = red()
    path = tmp_path / "red.png"
    PILImage.fromarray(array).save(path)
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    forms = [Image(array), Image(array.astype(float) / 255), Image(tensor), Image(PILImage.fromarray(array))]
    assert all(np.array_equal(np.asarray(form.load()), array) for form in forms)
    assert len({Image(array).digest, Image(array.copy()).digest}) == 1
    assert Image(path).digest == Image(str(path)).digest
    assert np.array_equal(np.asarray(Image(path).load()), array)
    assert Image(array).digest != Image(red(5)).digest


def test_an_image_refuses_what_isnt_one(tmp_path):
    with pytest.raises(ValueError, match="height"):
        Image(np.zeros(5))
    with pytest.raises(FileNotFoundError):
        Image(tmp_path / "missing.png")


def test_a_video_takes_frames_in_any_image_form():
    frames = [red(), red().astype(float) / 255, torch.from_numpy(red()).permute(2, 0, 1), Image(red())]
    video = Video(frames, fps=2.0)
    loaded, fps = video.load()
    assert loaded.shape == (4, 4, 4, 3)
    assert fps == 2.0
    with pytest.raises(ValueError, match="frames"):
        Video(np.zeros((4, 4, 3)))


def test_a_stimulus_turns_its_media_into_media_of_their_kind():
    stimulus = cg.Stimulus(image=red(), video=np.stack([red()] * 2))
    assert isinstance(stimulus.image, Image)
    assert isinstance(stimulus.video, Video)
    assert stimulus.modalities == frozenset({"image", "video"})
    assert list(stimulus.media) == ["image", "video"]
    with pytest.raises(TypeError, match="image field"):
        cg.Stimulus(image=Video(np.stack([red()] * 2)))
    assert as_media("video", stimulus.video) is stimulus.video


def test_a_template_places_media_among_the_text():
    template = cg.PromptTemplate(stimulus="See {image} and watch {video}:", response="")
    stimulus = cg.Stimulus(text="unused", image=red(), video=np.stack([red()] * 2))
    parts = template.query_parts(stimulus)
    assert parts == ("See ", stimulus.image, " and watch ", stimulus.video, ":")
    assert template.placed_media == frozenset({"image", "video"})
    with pytest.raises(ValueError, match="can't be written as text"):
        template.render_query(stimulus)
    with pytest.raises(ValueError, match="no image or video"):
        template.query_parts(cg.Stimulus(text="only text"))
    with pytest.raises(ValueError, match="fields"):
        cg.PromptTemplate(stimulus="{colour}")


def test_media_the_template_doesnt_place_come_first_and_text_prompts_skip_them():
    stimulus = cg.Stimulus(text="a red square", image=red())
    query = cg.Query(trial=cg.Trial(stimulus, options=("yes", "no")), instructions="Answer.")
    assert query.parts() == ("Answer.\n", stimulus.image, "a red square\nResponse:")
    assert query.prompt() == "Answer.\na red square\nResponse:"
    assert query.describe() == f"Answer.\n<image {stimulus.image.digest[:8]}>a red square\nResponse:"


def test_a_query_places_every_observation_with_its_media():
    template = cg.PromptTemplate(stimulus="Color: {image} ->", response="")
    first, second = cg.Stimulus(image=red()), cg.Stimulus(image=red(3))
    query = cg.Query(
        trial=cg.Trial(second, options=("wug", "dax")),
        context=(cg.Observation(first, "wug"),),
        template=template
    )
    assert query.parts() == ("Color: ", first.image, " -> wug\nColor: ", second.image, " ->")


def test_a_mapping_of_images_alone_places_them_by_default():
    images = [cg.media.swatch([value, 0, 0]) for value in (0.0, 0.5, 1.0)]
    task = cg.tasks.Mapping(None, ("wug", "dax"), images=images, points=[[0.0], [0.5], [1.0]])
    assert task.modalities == frozenset({"image"})
    assert task.template.stimulus == "Item: {image} -> Label:"
    assert [stimulus.image for stimulus in task.stimuli] == images
    assert task.config()["media"] == {"image": [image.digest for image in images]}
    assert task.config()["features"] is None
    assert len(task.start(0, 0)) == 3


def test_a_mapping_mixes_features_and_media_item_by_item():
    rgb = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    clips = [np.stack([red()] * 2), np.stack([red(3)] * 3)]
    task = cg.tasks.Mapping(rgb, ("wug", "dax"), images=[cg.media.swatch(color) for color in rgb], video=clips)
    assert task.modalities == frozenset({"text", "features", "image", "video"})
    assert task.stimuli[1].features.tolist() == [0.0, 0.0, 1.0]
    assert set(task.config()["media"]) == {"image", "video"}
    with pytest.raises(ValueError, match="one entry per item"):
        cg.tasks.Mapping(rgb, ("wug", "dax"), images=[cg.media.swatch(rgb[0])])
    with pytest.raises(ValueError, match="at least one item"):
        cg.tasks.Mapping(None, ("wug", "dax"))


def test_a_text_mapping_keeps_its_config_and_run_key(tmp_path):
    task = cg.tasks.Mapping([[0.1], [0.9]], ("wug", "dax"))
    assert "media" not in task.config()
    assert task.config()["features"] == [[0.1], [0.9]]


def test_a_swatch_is_a_flat_square_of_its_colour():
    swatch = cg.media.swatch([1.0, 0.5, 0.0], size=8)
    pixels = np.asarray(swatch.load())
    assert pixels.shape == (8, 8, 3)
    assert (pixels == [255, 128, 0]).all()
    with pytest.raises(ValueError, match="three values"):
        cg.media.swatch([2.0, 0.0, 0.0])


def test_cielab_converts_to_srgb():
    lab = pd.DataFrame([[100, 0, 0], [0, 0, 0], [53.24, 80.09, 67.20], [50, 0, 0]]).to_numpy(dtype=float)
    rgb = cg.media.lab_to_srgb(lab)
    assert rgb[0] == pytest.approx([1, 1, 1], abs=1e-3)
    assert rgb[1] == pytest.approx([0, 0, 0], abs=1e-6)
    assert rgb[2] == pytest.approx([1, 0, 0], abs=5e-3)
    assert rgb[3] == pytest.approx([0.466] * 3, abs=2e-3)
