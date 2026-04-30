from types import SimpleNamespace

import torch

from openrlhf.trainer.ppo_utils.experience_maker import Experience, RemoteExperienceMaker


def _build_maker(n_samples_per_prompt: int) -> RemoteExperienceMaker:
    args = SimpleNamespace(
        advantage_estimator="reinforce_baseline",
        n_samples_per_prompt=n_samples_per_prompt,
        reward_clip_range=None,
        gamma=1.0,
        lambd=1.0,
        no_advantage_std_norm=True,
        loss_aggregation="sample",
    )
    strategy = SimpleNamespace(args=args)
    kl_ctl = SimpleNamespace(value=0.0)
    return RemoteExperienceMaker(None, None, None, None, kl_ctl, strategy, tokenizer=None)


def _build_sample(reward: float, prompt_group_size: int) -> Experience:
    return Experience(
        index=[0],
        rewards=torch.tensor([reward], dtype=torch.float32),
        scores=torch.tensor([reward], dtype=torch.float32),
        action_mask=torch.tensor([[True]]),
        kl=torch.zeros((1, 1), dtype=torch.float32),
        info={"prompt_group_size": torch.tensor([prompt_group_size])},
    )


def test_make_experience_batch_uses_retained_prompt_group_sizes():
    maker = _build_maker(n_samples_per_prompt=4)
    def _split_rollout_samples(rollout_samples):
        for idx, sample in enumerate(rollout_samples):
            sample.index = [idx]
        return rollout_samples

    maker.split_rollout_samples = _split_rollout_samples
    maker.make_experience = lambda samples_list: samples_list

    rollout_samples = [
        _build_sample(1.0, 3),
        _build_sample(2.0, 3),
        _build_sample(3.0, 3),
        _build_sample(4.0, 1),
    ]

    experiences = maker.make_experience_batch(rollout_samples)

    assert maker._current_step_group_sizes == [3, 1]
    returns = torch.cat([experience.returns.flatten() for experience in experiences]).tolist()
    assert returns == [-1.0, 0.0, 1.0, 0.0]
