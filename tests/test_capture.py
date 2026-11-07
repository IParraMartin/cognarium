from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from helpers import RecordingCapture, ScriptedSubject, tiny_architecture

from cognarium.errors import MissingAbilityError
from cognarium.models.lstm import LSTMLanguageModel, SimpleTokenizer
from cognarium.paradigms.learning import IteratedLearning
from cognarium.recording.capture import StateCapture, pool_states
from cognarium.recording.layers import KNOWN_LAYER_MAPS, LayerMap, find_layer_map
from cognarium.responses import Query
from cognarium.stimuli import Observation, Stimulus, Trial
from cognarium.subjects.bayesian import BayesianLearner
from cognarium.subjects.custom import LSTM, CustomSubject
from cognarium.subjects.exemplar import ExemplarLearner
from cognarium.subjects.hf import HFSubject
from cognarium.tasks.shepard import ShepardCategories


def query(text="Object: small", options=("A", "B")):
    return Query(trial=Trial(Stimulus(text=text), options=options))


def blocks_without_hooks(model):
    return all(not block._forward_hooks for block in model.transformer.h)


@pytest.mark.parametrize("name", sorted(KNOWN_LAYER_MAPS))
def test_known_architectures_resolve(name, tiny_tokenizer):
    model = tiny_architecture(name, tiny_tokenizer)
    assert HFSubject(model, tiny_tokenizer, device="cpu").state_layers() == ("layer_00", "layer_01")
    layer_map = KNOWN_LAYER_MAPS[name]
    assert find_layer_map(model) == layer_map
    assert model.get_submodule(layer_map.head) is model.get_output_embeddings()


@pytest.mark.parametrize("name", sorted(KNOWN_LAYER_MAPS))
def test_known_architectures_record_a_state(name, tiny_tokenizer):
    model = tiny_architecture(name, tiny_tokenizer)
    capture = RecordingCapture()
    HFSubject(model, tiny_tokenizer, device="cpu").respond([query()], capture_states=capture)
    assert capture.stacked("layer_01").shape == (1, 16)


def test_layer_map_auto_detects_an_unknown_type(tiny_model_and_tokenizer):
    model, _ = tiny_model_and_tokenizer
    model.config.model_type = "something_new"
    assert find_layer_map(model) == LayerMap("transformer.h", "transformer.ln_f", "lm_head")


def test_a_known_map_whose_paths_moved_falls_back_to_detection(tiny_model_and_tokenizer, monkeypatch):
    model, _ = tiny_model_and_tokenizer
    monkeypatch.setitem(KNOWN_LAYER_MAPS, "gpt2", LayerMap("gone.h", "gone.ln_f", "gone"))
    assert find_layer_map(model) == LayerMap("transformer.h", "transformer.ln_f", "lm_head")


def test_undetectable_module_explains():
    with pytest.raises(MissingAbilityError, match="LayerMap"):
        find_layer_map(torch.nn.Linear(2, 2))


def test_a_head_without_a_list_of_blocks_explains():
    class HeadOnly(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(2, 2)

        def get_output_embeddings(self):
            return self.head

    with pytest.raises(MissingAbilityError, match="LayerMap"):
        find_layer_map(HeadOnly())


def test_a_passed_layer_map_wins(tiny_model_and_tokenizer):
    model, tok = tiny_model_and_tokenizer
    subject = HFSubject(
        model,
        tok,
        device="cpu",
        layer_map=LayerMap("transformer.h", None, "lm_head")
    )
    states = torch.zeros(1, 16)
    assert torch.equal(subject.readout().head(states), model.lm_head(states))


def test_last_pool_takes_the_last_real_token_under_padding(tiny_model_and_tokenizer):
    model, tok = tiny_model_and_tokenizer
    queries = [query("Object: small"), query("Object: small red circle")]
    batched, alone = RecordingCapture(pool="last"), RecordingCapture(pool="last")
    HFSubject(
        model,
        tok,
        device="cpu",
        batch_size=2
    ).respond(queries, capture_states=batched)
    HFSubject(
        model,
        tok,
        device="cpu",
        batch_size=1
    ).respond(queries, capture_states=alone)
    assert torch.allclose(batched.stacked("layer_00"), alone.stacked("layer_00"), atol=1e-4)


def test_first_block_matches_transformers_hidden_states(tiny_model_and_tokenizer):
    model, tok = tiny_model_and_tokenizer
    ids = tok(query().prompt(), return_tensors="pt")["input_ids"]
    with torch.inference_mode():
        expected = model(input_ids=ids, output_hidden_states=True).hidden_states[1][0, -1]
    capture = RecordingCapture()
    HFSubject(model, tok, device="cpu").respond([query()], capture_states=capture)
    assert torch.allclose(capture.stacked("layer_00")[0], expected, atol=1e-6)


def test_mean_pool_and_no_pool(tiny_model_and_tokenizer):
    model, tok = tiny_model_and_tokenizer
    queries = [query("Object: small"), query("Object: small red circle")]
    mean, full = RecordingCapture(pool="mean"), RecordingCapture(pool=None)
    subject = HFSubject(
        model,
        tok,
        device="cpu",
        batch_size=2
    )
    subject.respond(queries, capture_states=mean)
    subject.respond(queries, capture_states=full)
    states, _, lengths = full.written["layer_00"][0]
    assert lengths.tolist() == [len(tok(item.prompt())["input_ids"]) for item in queries]
    for row, length in enumerate(lengths.tolist()):
        expected = states[row, :length].mean(dim=0)
        assert torch.allclose(mean.stacked("layer_00")[row], expected, atol=1e-6)


def test_each_query_is_written_once_with_multi_token_options(tiny_model_and_tokenizer):
    model, tok = tiny_model_and_tokenizer
    subject = HFSubject(model, tok, device="cpu")
    multi = RecordingCapture()
    plain = RecordingCapture()
    prompts = ["Object: small", "Object: large"]
    subject.respond(
        [query(prompt, ("small red", "large blue")) for prompt in prompts],
        capture_states=multi
    )
    subject.respond([query(prompt) for prompt in prompts], capture_states=plain)
    assert multi.queries("layer_00") == [0, 1]
    assert torch.allclose(multi.stacked("layer_00"), plain.stacked("layer_00"), atol=1e-6)


def test_states_do_not_change_the_scores(tiny_model_and_tokenizer):
    model, tok = tiny_model_and_tokenizer
    subject = HFSubject(model, tok, device="cpu")
    queries = [query(), query("Object: large", ("small red", "large blue"))]
    plain = subject.respond(queries, capture_logits=True)
    recorded = subject.respond(queries, capture_logits=True, capture_states=RecordingCapture())
    assert torch.equal(plain.scores, recorded.scores)
    assert torch.equal(plain.logits, recorded.logits)


def test_only_requested_layers_are_hooked_and_hooks_come_off(tiny_model_and_tokenizer):
    model, tok = tiny_model_and_tokenizer
    capture = RecordingCapture(layers=("layer_01",))
    HFSubject(model, tok, device="cpu").respond([query()], capture_states=capture)
    assert set(capture.written) == {"layer_01"}
    assert blocks_without_hooks(model)


def test_hooks_come_off_after_a_failure(tiny_model_and_tokenizer):
    model, tok = tiny_model_and_tokenizer

    def fail(*args):
        raise RuntimeError("disk full")

    capture = StateCapture(("layer_00",), "last", fail)
    with pytest.raises(RuntimeError, match="disk full"):
        HFSubject(model, tok, device="cpu").respond([query()], capture_states=capture)
    assert blocks_without_hooks(model)


def test_unknown_layer_lists_the_known_ones(tiny_model_and_tokenizer):
    model, tok = tiny_model_and_tokenizer
    capture = RecordingCapture(layers=("nope",))
    with pytest.raises(ValueError, match="layer_00"):
        HFSubject(model, tok, device="cpu").respond([query()], capture_states=capture)


def test_a_layer_the_forward_call_skips_raises(tiny_model_and_tokenizer):
    model, tok = tiny_model_and_tokenizer
    subject = CustomSubject(
        model,
        tok,
        adapter=lambda module, ids, mask: torch.zeros(len(ids), ids.shape[1], len(tok)),
        layers={"unused": "transformer.h.0"},
        device="cpu",
        option_prefix=""
    )
    subject.tokenizer.encode = tok.encode
    with pytest.raises(ValueError, match="unused"):
        subject.respond([query()], capture_states=RecordingCapture(layers=("unused",)))


def test_pool_states_reads_each_output_form():
    output = torch.arange(12.0).reshape(2, 3, 2)
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    expected = torch.tensor([[2.0, 3.0], [10.0, 11.0]])
    for form in [output, (output, "extra"), SimpleNamespace(last_hidden_state=output)]:
        states, lengths = pool_states(form, mask, "last")
        assert torch.equal(states, expected)
        assert lengths is None


def test_pool_states_explains_bad_input():
    mask = torch.ones(2, 3, dtype=torch.long)
    with pytest.raises(ValueError, match="pool"):
        pool_states(torch.zeros(2, 3, 2), mask, "nope")
    with pytest.raises(ValueError, match="shape"):
        pool_states(torch.zeros(3, 2, 2), mask, "last")
    with pytest.raises(ValueError, match="tensor"):
        pool_states("not a tensor", mask, "last")


def test_states_reach_the_capture_on_every_device(tiny_model_and_tokenizer, device):
    model, tok = tiny_model_and_tokenizer
    capture = RecordingCapture()
    HFSubject(model, tok, device=device).respond([query()], capture_states=capture)
    states = capture.stacked("layer_01")
    assert states.shape == (1, 16)
    assert states.isfinite().all()


def test_custom_module_records_named_layers():
    tok = SimpleTokenizer.from_texts(["Object: small A B"])
    model = LSTMLanguageModel(len(tok), hidden_dim=8)
    subject = CustomSubject(
        model,
        tok,
        layers={"lstm": "lstm"},
        device="cpu"
    )
    capture = RecordingCapture(layers=("lstm",))
    subject.respond([query(), query()], capture_states=capture)
    assert subject.state_layers() == ("lstm",)
    assert capture.stacked("lstm").shape == (2, 8)


def test_custom_readout_and_first_tokens():
    tok = SimpleTokenizer.from_texts(["Object: small A B"])
    model = LSTMLanguageModel(len(tok), hidden_dim=8)
    subject = CustomSubject(
        model,
        tok,
        readout="readout",
        device="cpu"
    )
    readout = subject.readout()
    assert readout.head(torch.zeros(1, 8)).shape == (1, len(tok))
    assert readout.device == subject.device
    ids = subject.first_tokens(["A", "B"])
    assert ids == [tok.encode("A")[0], tok.encode("B")[0]]
    assert readout.decode(ids[0]) == "A"


def test_custom_subject_without_layers_cannot_record():
    tok = SimpleTokenizer.from_texts(["A B"])
    subject = CustomSubject(LSTMLanguageModel(len(tok)), tok, device="cpu")
    assert subject.state_layers() == ()
    assert subject.readout() is None


def test_bound_lstm_keeps_its_layers_through_fresh():
    paradigm = IteratedLearning(ShepardCategories(type="I"))
    subject = LSTM(device="cpu").bind(paradigm, "weights")
    fresh = subject.fresh((1,))
    assert fresh.state_layers() == ("lstm",)
    assert fresh.readout().head is fresh.model.readout


def test_hf_readout_turns_the_last_layer_into_the_logits(tiny_model_and_tokenizer):
    model, tok = tiny_model_and_tokenizer
    subject = HFSubject(model, tok, device="cpu")
    capture = RecordingCapture()
    response = subject.respond([query()], capture_logits=True, capture_states=capture)
    with torch.inference_mode():
        logits = subject.readout().head(capture.stacked("layer_01"))
    assert torch.allclose(logits, response.logits, atol=1e-5)


def test_hf_first_tokens_and_decode(tiny_model_and_tokenizer):
    model, tok = tiny_model_and_tokenizer
    subject = HFSubject(model, tok, device="cpu")
    ids = subject.first_tokens(["A", "small red"])
    assert ids == [tok.convert_tokens_to_ids("A"), tok.convert_tokens_to_ids("small")]
    assert subject.readout().decode(ids[0]) == "A"


def test_exemplar_writes_similarity_to_each_exemplar():
    context = (
        Observation(Stimulus(features=torch.tensor([0.0, 0.0])), "A"),
        Observation(Stimulus(features=torch.tensor([1.0, 0.0])), "B")
    )
    queries = [
        Query(trial=Trial(Stimulus(features=torch.tensor(point)), options=("A", "B")), context=context)
        for point in ([0.0, 0.0], [1.0, 1.0])
    ]
    capture = RecordingCapture(layers=("similarity",), pool=None)
    ExemplarLearner(c=2.0, device="cpu").respond(queries, capture_states=capture)
    expected = torch.tensor([[0.0, -1.0], [-2.0, -1.0]], dtype=torch.float64).exp()
    assert torch.allclose(capture.stacked("similarity").double(), expected)
    assert capture.written["similarity"][0][2] is None


def test_exemplar_without_context_writes_zero_width_states():
    queries = [Query(trial=Trial(Stimulus(features=torch.zeros(2)), options=("A", "B")))]
    capture = RecordingCapture(layers=("similarity",), pool=None)
    ExemplarLearner(device="cpu").respond(queries, capture_states=capture)
    assert capture.stacked("similarity").shape == (1, 0)


def test_bayesian_writes_a_normalised_posterior():
    task = ShepardCategories(type="I")
    subject = BayesianLearner(device="cpu").bind(IteratedLearning(task), "posterior")
    context = (Observation(task.stimuli[0], task.seed_labels[0]),)
    queries = [
        Query(trial=Trial(stimulus, options=task.options), context=context)
        for stimulus in task.stimuli[:3]
    ]
    capture = RecordingCapture(layers=("posterior",), pool=None)
    subject.respond(queries, capture_states=capture)
    states = capture.stacked("posterior")
    assert states.shape == (3, 256)
    assert states.sum(dim=1).double().numpy() == pytest.approx([1.0, 1.0, 1.0])
    assert torch.equal(states.double(), subject.log_posterior(context).exp().cpu().double().expand(3, -1))


def test_non_neural_subjects_record_in_float32():
    assert ExemplarLearner.state_dtype == "float32"
    assert BayesianLearner.state_dtype == "float32"
    assert ExemplarLearner().state_layers() == ("similarity",)
    assert BayesianLearner().state_layers() == ("posterior",)


def test_neural_subjects_record_in_float16():
    assert HFSubject.state_dtype == "float16"
    assert CustomSubject.state_dtype == "float16"


def test_subjects_without_states_say_so():
    assert ScriptedSubject().state_layers() == ()
    assert ScriptedSubject().readout() is None
    assert ScriptedSubject().first_tokens(["A"]) is None
