from torch.utils.data import DataLoader

from openrlhf.datasets.prompts_dataset import PromptPoolDataset
from openrlhf.trainer.ppo_utils.experience_maker import _collect_prompt_batch


def _build_samples():
    return [
        {
            "idx": 0,
            "prompt_ref": "dataset_a:0",
            "datasource": "task_a",
            "prompt": "prompt a0",
            "label": "(A)",
            "knn_pseudo_label": "A",
            "late_phase_prompt": False,
            "raw_record": {"id": "a0"},
        },
        {
            "idx": 1,
            "prompt_ref": "dataset_a:1",
            "datasource": "task_a",
            "prompt": "prompt a1",
            "label": "(B)",
            "knn_pseudo_label": "B",
            "late_phase_prompt": True,
            "raw_record": {"id": "a1"},
        },
    ]


def test_prompt_pool_dataset_exposes_prompt_refs_and_lookup():
    dataset = PromptPoolDataset(_build_samples(), dataset_key="mixed_recovery")

    assert len(dataset) == 2
    item = dataset[1]
    assert item[-1] == "dataset_a:1"

    sample = dataset.get_sample_by_prompt_ref("dataset_a:0")
    assert sample["prompt"] == "prompt a0"
    assert sample["datasource"] == "task_a"


def test_collect_prompt_batch_returns_prompt_refs_for_phase_datasets():
    dataset = PromptPoolDataset(_build_samples(), dataset_key="mixed_recovery")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)

    (
        indices,
        datasources,
        prompts,
        labels,
        knn_pseudo_labels,
        late_phase_prompt_flags,
        prompt_refs,
        exhausted,
    ) = _collect_prompt_batch(iter(dataloader), 2)

    assert exhausted is True
    assert indices == [0, 1]
    assert datasources == ["task_a", "task_a"]
    assert prompts == ["prompt a0", "prompt a1"]
    assert labels == ["(A)", "(B)"]
    assert knn_pseudo_labels == ["A", "B"]
    assert late_phase_prompt_flags == [False, True]
    assert prompt_refs == ["dataset_a:0", "dataset_a:1"]
