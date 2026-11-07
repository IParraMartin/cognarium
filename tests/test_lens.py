from __future__ import annotations

import copy

import pandas as pd
import pytest
import torch
from helpers import TINY_ARCHITECTURES, tiny_architecture

from cognarium.errors import MissingAbilityError
from cognarium.models.lstm import LSTMLanguageModel, SimpleTokenizer
from cognarium.paradigms.learning import IteratedLearning
from cognarium.recording.layers import LayerMap
from cognarium.results import Results
from cognarium.runner import run
from cognarium.subjects.bayesian import BayesianLearner
from cognarium.subjects.custom import LSTM, CustomSubject
from cognarium.subjects.exemplar import ExemplarLearner
from cognarium.subjects.hf import HFSubject
from cognarium.tasks.shepard import ShepardCategories


def paradigm():
    return IteratedLearning(ShepardCategories(type="I"), generations=3, chains=2)


def recorded(
    subject,
    tmp_path,
    *,
    name="run",
    **kwargs
):
    return run(
        paradigm(),
        subject,
        record=["logits", "hidden_states"],
        out=tmp_path / name,
        progress=False,
        **kwargs
    )


@pytest.mark.parametrize("name", sorted(TINY_ARCHITECTURES))
def test_final_layer_reproduces_the_models_own_logits(name, tiny_tokenizer, tmp_path):
    model = tiny_architecture(name, tiny_tokenizer)
    subject = HFSubject(model, tiny_tokenizer, device="cpu")
    results = recorded(subject, tmp_path, state_dtype="float32")
    states = results.hidden.to_tensor("layer_01")
    logits = torch.cat([results.logits(step) for step in range(3)])
    with torch.inference_mode():
        read = subject.readout().head(states.to(subject.device))
    assert torch.allclose(read, logits, atol=1e-3)
    table = results.lens()
    final = table[table["layer"] == "layer_01"]
    assert final["top_token"].tolist() == [tiny_tokenizer.decode([i]) for i in logits.argmax(dim=-1).tolist()]


def test_final_layer_reproduces_soft_capped_logits(tiny_tokenizer, tmp_path):
    model = tiny_architecture("gemma2", tiny_tokenizer)
    model.config.final_logit_softcapping = 30.0
    # random weights keep logits far below the cap, where it changes nothing
    with torch.no_grad():
        model.lm_head.weight.mul_(400)
    subject = HFSubject(model, tiny_tokenizer, device="cpu")
    results = recorded(subject, tmp_path, state_dtype="float32")
    logits = torch.cat([results.logits(step) for step in range(3)])
    assert logits.abs().max() > 20
    with torch.inference_mode():
        read = subject.readout().head(results.hidden.to_tensor("layer_01").to(subject.device))
    assert torch.allclose(read, logits, atol=1e-3)


def test_readout_refuses_a_model_that_changes_logits_after_its_head(tiny_tokenizer):
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.for_model(
        "cohere",
        vocab_size=len(tiny_tokenizer),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2
    )
    subject = HFSubject(AutoModelForCausalLM.from_config(config), tiny_tokenizer, device="cpu")
    with pytest.raises(MissingAbilityError, match="logits"):
        subject.readout()


def test_readout_refuses_a_head_that_does_not_fit_the_blocks(tiny_tokenizer):
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.for_model(
        "opt",
        vocab_size=len(tiny_tokenizer),
        hidden_size=16,
        word_embed_proj_dim=8,
        ffn_dim=32,
        num_hidden_layers=2,
        num_attention_heads=2
    )
    subject = HFSubject(AutoModelForCausalLM.from_config(config), tiny_tokenizer, device="cpu")
    with pytest.raises(MissingAbilityError, match="fit"):
        subject.readout()


def test_readout_refuses_a_layer_map_that_misses_the_norm(tiny_model_and_tokenizer):
    model, tok = tiny_model_and_tokenizer
    wrong = LayerMap(layers="transformer.h", norm=None, head="lm_head")
    subject = HFSubject(
        model,
        tok,
        device="cpu",
        layer_map=wrong
    )
    with pytest.raises(MissingAbilityError, match="logits"):
        subject.readout()


def test_lens_table_shape_and_values(tiny_model_and_tokenizer, tmp_path):
    model, tok = tiny_model_and_tokenizer
    results = recorded(HFSubject(model, tok, device="cpu"), tmp_path)
    table = results.lens()
    assert len(table) == 48 * 2
    assert list(table.columns) == [
        "step",
        "query",
        "item",
        "generation",
        "layer",
        "top_token",
        "top_log_prob",
        "entropy",
        "option_log_probs",
        "logit_margin",
        "target_rank",
        "logit_difference"
    ]
    assert table["query"].max() == 15
    assert table[table["step"] == 1]["query"].tolist()[:16] == list(range(16))
    totals = table["option_log_probs"].map(lambda row: torch.tensor(row).exp().sum().item())
    assert totals.to_numpy() == pytest.approx([1.0] * len(table), abs=1e-5)
    assert table["entropy"].between(0, 20).all()
    assert (table["top_log_prob"] <= 0).all()
    assert (table["target_rank"] >= 1).all()
    assert (table["logit_margin"] >= 0).all()


def test_lens_rows_are_labelled_like_the_table(tiny_model_and_tokenizer, tmp_path):
    model, tok = tiny_model_and_tokenizer
    results = recorded(HFSubject(model, tok, device="cpu"), tmp_path)
    frame = results.to_pandas()
    table = results.lens(layers=["layer_00"])
    assert table["step"].tolist() == frame["step"].tolist()
    assert table["item"].tolist() == frame["item"].tolist()
    assert table["generation"].tolist() == frame["generation"].tolist()


def test_final_layer_matches_the_recorded_logit_columns(tiny_model_and_tokenizer, tmp_path):
    model, tok = tiny_model_and_tokenizer
    results = recorded(HFSubject(model, tok, device="cpu"), tmp_path, state_dtype="float32")
    frame = results.to_pandas()
    final = results.lens(layers=["layer_01"])
    assert final["target_rank"].tolist() == frame["target_rank"].tolist()
    assert final["logit_margin"].to_numpy() == pytest.approx(frame["logit_margin"].to_numpy(), abs=1e-3)
    assert final["entropy"].to_numpy() == pytest.approx(frame["vocab_entropy"].to_numpy(), abs=1e-3)


def test_lens_on_chosen_layers(tiny_model_and_tokenizer, tmp_path):
    model, tok = tiny_model_and_tokenizer
    results = recorded(HFSubject(model, tok, device="cpu"), tmp_path)
    assert set(results.lens(layers=["layer_00"])["layer"]) == {"layer_00"}
    with pytest.raises(ValueError, match="layer_00"):
        results.lens(layers=["nope"])


def test_batch_size_does_not_change_the_lens(tiny_model_and_tokenizer, tmp_path):
    model, tok = tiny_model_and_tokenizer
    results = recorded(HFSubject(model, tok, device="cpu"), tmp_path)
    one, many = results.lens(batch_size=1), results.lens(batch_size=7)
    assert one["top_token"].tolist() == many["top_token"].tolist()
    assert one["entropy"].to_numpy() == pytest.approx(many["entropy"].to_numpy(), abs=1e-3)


def test_lens_after_load_needs_the_subject(tiny_model_and_tokenizer, tmp_path):
    model, tok = tiny_model_and_tokenizer
    subject = HFSubject(model, tok, device="cpu")
    original = recorded(subject, tmp_path).lens()
    with pytest.raises(MissingAbilityError, match="subject"):
        Results.load(tmp_path / "run").lens()
    reloaded = Results.load(tmp_path / "run").lens(subject)
    pd.testing.assert_frame_equal(reloaded, original)


def test_lens_refuses_a_different_model(tiny_model_and_tokenizer, tmp_path):
    model, tok = tiny_model_and_tokenizer
    results = recorded(HFSubject(model, tok, device="cpu"), tmp_path)
    changed = copy.deepcopy(model)
    with torch.no_grad():
        changed.lm_head.weight.add_(1.0)
    with pytest.raises(ValueError, match="weights"):
        results.lens(HFSubject(changed, tok, device="cpu"))


def test_lens_refuses_untrained_weights_it_cannot_check(tmp_path):
    tok = SimpleTokenizer.from_texts(paradigm().texts())
    subject = CustomSubject(
        LSTMLanguageModel(len(tok), hidden_dim=8),
        tok,
        pretrained=False,
        layers={"lstm": "lstm"},
        readout="readout",
        device="cpu"
    )
    with pytest.warns(UserWarning):
        results = recorded(subject, tmp_path, learning="in_context")
    assert len(results.lens()) == 48
    other = CustomSubject(
        LSTMLanguageModel(len(tok), hidden_dim=8),
        tok,
        pretrained=False,
        layers={"lstm": "lstm"},
        readout="readout",
        device="cpu"
    )
    with pytest.raises(ValueError, match="untrained"):
        results.lens(other)


def test_lens_on_a_mean_pooled_run_raises(tiny_model_and_tokenizer, tmp_path):
    model, tok = tiny_model_and_tokenizer
    results = recorded(HFSubject(model, tok, device="cpu"), tmp_path, pool="mean")
    with pytest.raises(ValueError, match="last"):
        results.lens()


def test_lens_without_hidden_states_raises(tiny_model_and_tokenizer, tmp_path):
    model, tok = tiny_model_and_tokenizer
    results = run(
        paradigm(),
        HFSubject(model, tok, device="cpu"),
        record=["logits"],
        out=tmp_path / "run",
        progress=False
    )
    with pytest.raises(MissingAbilityError, match="hidden_states"):
        results.lens()


@pytest.mark.parametrize("subject", [ExemplarLearner(guess=0.05), BayesianLearner()])
def test_lens_on_non_neural_subjects_raises(subject, tmp_path):
    results = run(
        paradigm(),
        subject,
        record=["hidden_states"],
        out=tmp_path / "run",
        progress=False
    )
    with pytest.raises(MissingAbilityError, match=type(subject).__name__):
        results.lens()


def test_lens_on_a_weights_run_explains(tmp_path):
    results = recorded(LSTM(hidden_dim=8, train_epochs=1, device="cpu"), tmp_path)
    assert results.manifest["learning"] == "weights"
    with pytest.raises(MissingAbilityError, match="weights"):
        results.lens()


def test_lens_on_a_custom_module_in_context(tmp_path):
    tok = SimpleTokenizer.from_texts(paradigm().texts())
    subject = CustomSubject(
        LSTMLanguageModel(len(tok), hidden_dim=8),
        tok,
        layers={"lstm": "lstm"},
        readout="readout",
        device="cpu"
    )
    table = recorded(subject, tmp_path, learning="in_context").lens()
    assert len(table) == 48
    assert table["top_token"].map(lambda token: isinstance(token, str)).all()


def test_lens_runs_on_every_device(tiny_model_and_tokenizer, tmp_path, device):
    model, tok = tiny_model_and_tokenizer
    table = recorded(HFSubject(model, tok, device=device), tmp_path).lens()
    assert len(table) == 96
    assert table["entropy"].notna().all()


def test_lens_works_on_float16_states(tiny_model_and_tokenizer, tmp_path):
    model, tok = tiny_model_and_tokenizer
    subject = HFSubject(model, tok, device="cpu")
    rounded = recorded(subject, tmp_path, name="rounded").lens(layers=["layer_01"])
    exact = recorded(
        subject,
        tmp_path,
        name="exact",
        state_dtype="float32"
    ).lens(layers=["layer_01"])
    matches = sum(a == b for a, b in zip(rounded["top_token"], exact["top_token"], strict=True))
    assert matches / len(exact) >= 0.95
