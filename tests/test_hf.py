from __future__ import annotations

import copy

import pytest
import torch

from cognarium.loading import _resolve_revision, load
from cognarium.responses import Query
from cognarium.stimuli import Stimulus, Trial
from cognarium.subjects.exemplar import ExemplarLearner
from cognarium.subjects.hf import HFSubject


def query(text="Object: small", options=("A", "B")):
    return Query(trial=Trial(Stimulus(text=text), options=options))


def brute_force(
    model,
    tokenizer,
    text,
    option
):
    ids = tokenizer(text, return_tensors="pt")["input_ids"]
    option_ids = tokenizer(" " + option, add_special_tokens=False)["input_ids"]
    total = 0.0
    with torch.inference_mode():
        for token in option_ids:
            total += model(input_ids=ids).logits[0, -1].log_softmax(-1)[token].item()
            ids = torch.cat([ids, torch.tensor([[token]])], dim=1)
    return total


def test_single_token_scores_match_brute_force(tiny_model_and_tokenizer):
    model, tok = tiny_model_and_tokenizer
    scores = HFSubject(model, tok, device="cpu").respond([query()]).scores
    assert scores[0, 0].item() == pytest.approx(brute_force(
        model,
        tok,
        query().prompt(),
        "A"
    ), abs=1e-5)


def test_multi_token_option_sums_its_tokens(tiny_model_and_tokenizer):
    model, tok = tiny_model_and_tokenizer
    scores = HFSubject(model, tok, device="cpu").respond([query(options=("A", "two tokens"))]).scores
    assert scores[0, 1].item() == pytest.approx(brute_force(
        model,
        tok,
        query().prompt(),
        "two tokens"
    ), abs=1e-5)


def test_batch_size_and_padding_do_not_change_scores(tiny_model_and_tokenizer):
    model, tok = tiny_model_and_tokenizer
    queries = [query("Object: small red circle Category:"), query("Object:")]
    together = HFSubject(
        model,
        tok,
        device="cpu",
        batch_size=2
    ).respond(queries).scores
    apart = HFSubject(
        model,
        tok,
        device="cpu",
        batch_size=1
    ).respond(queries).scores
    assert torch.allclose(together, apart, atol=1e-5)


def test_capture_logits_returns_the_full_vocabulary(tiny_model_and_tokenizer):
    model, tok = tiny_model_and_tokenizer
    logits = HFSubject(model, tok, device="cpu").respond([query()], capture_logits=True).logits
    assert logits.shape == (1, len(tok))


@pytest.mark.parametrize(
    ("options", "message"),
    [(("A", "unknownword"), "unknownword"), (("A", "A "), "same tokens")]
)
def test_unscoreable_options_raise(tiny_model_and_tokenizer, options, message):
    model, tok = tiny_model_and_tokenizer
    with pytest.raises(ValueError, match=message):
        HFSubject(model, tok, device="cpu").respond([query(options=options)])


def test_prompt_over_the_position_limit_raises(tiny_model_and_tokenizer):
    model, tok = tiny_model_and_tokenizer
    with pytest.raises(ValueError, match="64"):
        HFSubject(model, tok, device="cpu").respond([query("small " * 70)])


def test_scores_agree_across_devices(tiny_model_and_tokenizer, device):
    model, tok = tiny_model_and_tokenizer
    expected = HFSubject(copy.deepcopy(model), tok, device="cpu").respond([query()]).scores
    actual = HFSubject(copy.deepcopy(model), tok, device=device).respond([query()]).scores
    assert actual.device.type == "cpu"
    assert torch.allclose(expected, actual, atol=1e-3)


def test_config_changes_with_the_weights(tiny_model_and_tokenizer):
    model, tok = tiny_model_and_tokenizer
    before = HFSubject(model, tok, device="cpu").config()
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    assert HFSubject(model, tok, device="cpu").config() != before


def test_tokenizer_without_pad_or_eos_raises(tiny_model_and_tokenizer):
    model, tok = tiny_model_and_tokenizer
    tok.pad_token = None
    tok.eos_token = None
    with pytest.raises(ValueError, match="pad"):
        HFSubject(model, tok, device="cpu")


def test_load_reads_a_saved_model(tiny_model_and_tokenizer, tmp_path):
    model, tok = tiny_model_and_tokenizer
    model.save_pretrained(tmp_path)
    tok.save_pretrained(tmp_path)
    subject = load(str(tmp_path), device="cpu")
    assert isinstance(subject, HFSubject)
    assert subject.respond([query()]).scores.shape == (1, 2)


def test_load_wraps_a_model_object(tiny_model_and_tokenizer):
    model, tok = tiny_model_and_tokenizer
    assert isinstance(load(model, tokenizer=tok, device="cpu"), HFSubject)


def test_load_returns_subjects_unchanged_and_rejects_others():
    subject = ExemplarLearner()
    assert load(subject) is subject
    with pytest.raises(TypeError):
        load(3)


def fake_hub_cache(root, sha="abc123"):
    repo = root / "models--org--tiny"
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text(sha)
    (repo / "snapshots" / sha).mkdir(parents=True)
    (repo / "snapshots" / sha / "config.json").write_text("{}")


def test_revision_comes_from_the_cached_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
    fake_hub_cache(tmp_path)
    assert _resolve_revision("org/tiny", None) == "abc123"
    assert _resolve_revision("org/tiny", "abc123") == "abc123"


def test_revision_falls_back_to_the_one_requested(tmp_path, monkeypatch):
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
    assert _resolve_revision("org/absent", "v1") == "v1"
    assert _resolve_revision("org/absent", None) is None


def test_a_local_folder_has_no_revision(tmp_path):
    assert _resolve_revision(str(tmp_path), None) is None


def test_load_records_the_revision(tiny_model_and_tokenizer, tmp_path, monkeypatch):
    model, tok = tiny_model_and_tokenizer
    model.save_pretrained(tmp_path / "model")
    tok.save_pretrained(tmp_path / "model")
    monkeypatch.setattr("cognarium.loading._resolve_revision", lambda name, revision: "abc123")
    assert load(str(tmp_path / "model"), device="cpu").config()["revision"] == "abc123"


def test_weights_still_count_when_there_is_a_revision(tiny_model_and_tokenizer):
    model, tok = tiny_model_and_tokenizer
    subject = HFSubject(
        model,
        tok,
        device="cpu",
        revision="abc123"
    )
    before = subject.config()
    assert before["revision"] == "abc123"
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    assert subject.config()["weights"] != before["weights"]
