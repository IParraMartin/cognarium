"""Hugging Face models that read images, video and audio along with text."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import cognarium as cg
from cognarium.errors import MissingAbilityError
from cognarium.subjects.multimodal import HFMultimodalSubject, find_media_tokens

RATE = 16000
COLOR = cg.PromptTemplate(stimulus="Color: {image} -> Label:", response="")
MIXED = cg.PromptTemplate(stimulus="Sight {image} sound {audio} clip {video} ->", response="")


def vision_subject(vision_model_and_processor, **settings):
    model, processor = vision_model_and_processor
    return cg.load(model, processor=processor, device="cpu", **settings)


def omni_subject(omni_model_and_processor, **settings):
    model, processor = omni_model_and_processor
    return cg.load(model, processor=processor, device="cpu", **settings)


def chip(value):
    return cg.media.swatch([value, 0.2, 1 - value])


def tone(frequency, seconds=0.3):
    time = np.arange(int(RATE * seconds)) / RATE
    return cg.media.Audio((0.3 * np.sin(2 * np.pi * frequency * time)).astype(np.float32), RATE)


def clip(value, frames=4):
    return cg.media.Video(np.stack([np.asarray(chip((value + step / 8) % 1).load()) for step in range(frames)]), fps=2.0)


def color_query(value, options=("wug", "dax"), context=()):
    return cg.Query(trial=cg.Trial(cg.Stimulus(image=chip(value)), options=options), context=context, template=COLOR)


def manual_scores(model, processor, text, images, options):
    scores = []
    for option in options:
        inputs = processor(text=[text], images=images, return_tensors="pt")
        prompt_length = inputs["input_ids"].shape[1]
        tokens = processor.tokenizer(f" {option}", add_special_tokens=False)["input_ids"]
        ids = torch.cat([inputs["input_ids"], torch.tensor([tokens])], dim=1)
        aligned = torch.cat([inputs["mm_token_type_ids"], torch.zeros((1, len(tokens)), dtype=torch.long)], dim=1)
        with torch.inference_mode():
            logits = model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                mm_token_type_ids=aligned,
                pixel_values=inputs["pixel_values"],
                image_grid_thw=inputs["image_grid_thw"]
            ).logits
        log_probs = logits[0].float().log_softmax(dim=-1).double()
        positions = torch.arange(prompt_length - 1, prompt_length - 1 + len(tokens))
        scores.append(log_probs[positions, torch.tensor(tokens)].sum().item())
    return torch.tensor(scores, dtype=torch.float64)


def test_load_gives_a_multimodal_subject_that_finds_its_media_tokens(vision_model_and_processor, omni_model_and_processor):
    vision = vision_subject(vision_model_and_processor)
    assert isinstance(vision, HFMultimodalSubject)
    assert vision.media_tokens == {
        "image": "<|vision_start|><|image_pad|><|vision_end|>",
        "video": "<|vision_start|><|video_pad|><|vision_end|>"
    }
    assert vision.modalities == frozenset({"text", "image", "video"})
    assert vision.state_layers() == ("layer_00", "layer_01")
    omni = omni_subject(omni_model_and_processor)
    assert omni.media_tokens["audio"] == "<|audio_bos|><|AUDIO|><|audio_eos|>"
    assert omni.modalities == frozenset({"text", "image", "video", "audio"})


def test_media_tokens_come_from_the_chat_template_without_token_attributes(vision_model_and_processor):
    _, processor = vision_model_and_processor

    class Templated:
        image_processor = processor.image_processor
        tokenizer = processor.tokenizer
        chat_template = "x"

        def apply_chat_template(self, messages, tokenize):
            return "".join(
                part["text"] if part["type"] == "text" else f"<start_of_{part['type']}>"
                for part in messages[0]["content"]
            )

    assert find_media_tokens(Templated()) == {"image": "<start_of_image>"}
    assert find_media_tokens(processor, {"image": "<img>"})["image"] == "<img>"
    with pytest.raises(ValueError, match="unknown kinds"):
        find_media_tokens(processor, {"smell": "<smell>"})


def test_options_are_scored_with_the_image_in_the_prompt(vision_model_and_processor):
    model, processor = vision_model_and_processor
    subject = vision_subject(vision_model_and_processor)
    query = color_query(0.3)
    scores = subject.respond([query]).scores[0]
    text = "Color: <|vision_start|><|image_pad|><|vision_end|> -> Label:"
    expected = manual_scores(model, processor, text, [query.trial.stimulus.image.load()], ("wug", "dax"))
    assert scores == pytest.approx(expected, abs=1e-5)
    other = subject.respond([color_query(0.9)]).scores[0]
    assert not torch.allclose(scores, other)


def test_a_query_scores_the_same_alone_and_in_any_batch(vision_model_and_processor):
    context = (cg.Observation(cg.Stimulus(image=chip(0.1)), "wug"), cg.Observation(cg.Stimulus(image=chip(0.7)), "dax"))
    queries = [color_query(value, context=context if value > 0.5 else ()) for value in (0.0, 0.4, 0.6, 0.95)]
    alone = torch.cat([vision_subject(vision_model_and_processor).respond([query]).scores for query in queries])
    together = vision_subject(vision_model_and_processor, batch_size=3).respond(queries).scores
    assert together == pytest.approx(alone, abs=1e-5)


@pytest.mark.parametrize("build", ["vision", "omni"])
def test_multi_token_options_from_the_cache_match_the_whole_prompt(build, vision_model_and_processor, omni_model_and_processor):
    if build == "vision":
        subject = vision_subject(vision_model_and_processor)
        query = color_query(0.5, options=("blue green", "red orange", "violet"))
    else:
        subject = omni_subject(omni_model_and_processor)
        stimulus = cg.Stimulus(image=chip(0.5), audio=tone(300), video=clip(0.2))
        query = cg.Query(trial=cg.Trial(stimulus, options=("blue green", "red orange")), template=MIXED)
    cached = subject.respond([query]).scores
    assert subject._cache_trusted is True
    subject._cache_trusted = False
    whole = subject.respond([query]).scores
    assert cached == pytest.approx(whole, abs=1e-4)


def test_a_cache_that_disagrees_is_dropped_with_a_warning(vision_model_and_processor, monkeypatch):
    subject = vision_subject(vision_model_and_processor)
    query = color_query(0.5, options=("blue green", "red orange"))
    whole = subject.respond([query]).scores
    subject = vision_subject(vision_model_and_processor)
    real = cg.subjects.HFSubject._score_after_prompt
    monkeypatch.setattr(
        cg.subjects.HFSubject,
        "_score_after_prompt",
        lambda self, index, prompt, options: real(self, index, prompt, options) + 1.0
    )
    with pytest.warns(UserWarning, match="whole prompt"):
        scores = subject.respond([query, query]).scores
    assert subject._cache_trusted is False
    assert scores == pytest.approx(torch.cat([whole, whole]), abs=1e-5)


def test_iterated_learning_passes_chip_images_down_chains_with_every_recording(vision_model_and_processor, tmp_path):
    values = np.linspace(0, 1, 8)
    rgb = [[value, 0.2, 1 - value] for value in values]
    task = cg.tasks.Mapping(rgb, ("wug", "dax"), images=[chip(value) for value in values], template=COLOR)
    paradigm = cg.paradigms.IteratedLearning(task, generations=2, chains=2, n_prev_examples=4, memory=2)
    subject = vision_subject(vision_model_and_processor)
    results = cg.run(paradigm, subject, out=tmp_path / "run", record=["logits", "hidden_states"], progress=False)
    table = results.to_pandas()
    assert len(table) == 2 * 2 * 8
    assert set(table["response"]) <= {"wug", "dax"}
    assert results.logits(0).shape[0] == 2
    assert set(results.hidden.layers) == {"layer_00", "layer_01"}
    assert not results.lens().empty
    assert not results.metrics("accuracy", "fidelity", "entropy").empty
    assert results.manifest["subject"]["class"] == "HFMultimodalSubject"
    rerun = cg.run(
        paradigm,
        vision_subject(vision_model_and_processor),
        out=tmp_path / "run",
        record=["logits", "hidden_states"],
        progress=False
    )
    assert rerun.manifest["key"] == results.manifest["key"]
    exemplar = cg.run(paradigm, cg.subjects.ExemplarLearner(), out=tmp_path / "exemplar", progress=False)
    assert len(exemplar.to_pandas()) == len(table)


def test_chains_can_pass_on_a_mix_of_image_sound_and_video(omni_model_and_processor, tmp_path):
    task = cg.tasks.Mapping(
        None,
        ("wug", "dax"),
        images=[chip(index / 5) for index in range(6)],
        audio=[tone(200 + 60 * index, 0.2 + 0.05 * index) for index in range(6)],
        video=[clip(index / 5) for index in range(6)],
        template=MIXED,
        points=[[index] for index in range(6)]
    )
    paradigm = cg.paradigms.IteratedLearning(task, generations=2, chains=1, n_prev_examples=3)
    results = cg.run(paradigm, omni_subject(omni_model_and_processor), out=tmp_path, record=["logits"], progress=False)
    assert len(results.to_pandas()) == 12
    assert not results.metrics("ib_complexity", by=["generation"]).empty


def test_media_tokens_and_processor_settings_are_part_of_the_run(vision_model_and_processor):
    subject = vision_subject(
        vision_model_and_processor,
        processor_kwargs={"images_kwargs": {"size": {"shortest_edge": 1024, "longest_edge": 1024}}}
    )
    config = subject.config()
    assert config["processor"] == "Qwen3VLProcessor"
    assert config["media_tokens"]["image"] == "<|vision_start|><|image_pad|><|vision_end|>"
    assert config["processor_kwargs"] == {"images_kwargs": {"size": {"shortest_edge": 1024, "longest_edge": 1024}}}
    small = subject._encode(color_query(0.5))
    assert small.media["image_grid_thw"].tolist() == [[1, 2, 2]]


def test_free_responses_read_the_media_and_repeat_with_the_seed(vision_model_and_processor):
    subject = vision_subject(vision_model_and_processor)
    query = cg.Query(trial=cg.Trial(cg.Stimulus(image=chip(0.4))), template=COLOR, response_seed=(3,))
    first = subject.generate([query, query], max_new_tokens=4)
    assert first[0] == first[1]
    assert subject.generate([query], max_new_tokens=4) == first[:1]


def test_a_text_subject_refuses_a_template_that_places_media(byte_model_and_tokenizer, tmp_path):
    model, tokenizer = byte_model_and_tokenizer
    task = cg.tasks.Mapping(None, ("wug", "dax"), images=[chip(0.1), chip(0.9)], template=COLOR)
    paradigm = cg.paradigms.IteratedLearning(task, generations=1, chains=1, n_prev_examples=1)
    with pytest.raises(MissingAbilityError, match="multimodal=True"):
        cg.run(paradigm, cg.load(model, tokenizer=tokenizer, device="cpu"), out=tmp_path, progress=False)


def test_a_subject_refuses_media_its_model_cant_read(vision_model_and_processor):
    query = cg.Query(trial=cg.Trial(cg.Stimulus(audio=tone(300)), options=("a", "b")), template=cg.PromptTemplate("{audio}"))
    with pytest.raises(MissingAbilityError, match="audio"):
        vision_subject(vision_model_and_processor).respond([query])


def test_load_by_name_with_multimodal_reads_a_saved_model(vision_model_and_processor, tmp_path):
    model, processor = vision_model_and_processor
    model.save_pretrained(tmp_path / "model")
    processor.save_pretrained(tmp_path / "model")
    subject = cg.load(str(tmp_path / "model"), device="cpu", dtype="float32", multimodal=True)
    assert isinstance(subject, HFMultimodalSubject)
    expected = vision_subject(vision_model_and_processor).respond([color_query(0.2)]).scores
    assert subject.respond([color_query(0.2)]).scores == pytest.approx(expected, abs=1e-5)


def test_media_settings_need_a_model_that_reads_media(byte_model_and_tokenizer):
    model, tokenizer = byte_model_and_tokenizer
    with pytest.raises(TypeError, match="multimodal=True"):
        cg.load(model, tokenizer=tokenizer, processor_kwargs={"size": 1})
