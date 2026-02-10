Pipeline Trace: scripts/train_grpo_tdc.sh

  Phase 0: Shell Setup

  The script parses args (TASK_NAME, PRETRAIN_PATH, LEARNING_RATE, NUM_GPUS), sets environment variables (OPENRLHF_MODEL_PATH,
  OPENRLHF_PROMPT_CONSTRUCTION_MODE, OPENRLHF_MAX_STEPS), starts a Ray cluster (ray start --head), then invokes:

  python -m openrlhf.cli.train_ppo_ray  --agent_func_path ... --agent_max_steps 40 ...

  ---
  Phase 1: CLI Entry — openrlhf/cli/train_ppo_ray.py :: train(args)

  1. Arg pre-processing: When agent_func_path is set, args.remote_rm_url is forced to "agent", which disables creation of separate
  reward/critic/reference model actors.
  2. Ray init: Connects to the existing Ray cluster (RAY_ADDRESS=auto).
  3. Placement group: --colocate_all_models creates a single placement group for all GPU bundles.
  4. vLLM engines: Calls create_vllm_engines() → creates LLMRayActor Ray remote actors.
  5. Actor model: Creates RayActorGroup wrapping PolicyModelActor Ray actors (no ref/critic/reward models since init_kl_coef=0 and
  advantage_estimator=dr_grpo).
  6. Trainer: Creates PPOTrainer.remote(...) — a Ray remote actor.
  7. Kick off training: ray.get(ppo_trainer.fit.remote())

  ---
  Phase 2: vLLM Engine Init — LLMRayActor.__init__()

  File: openrlhf/trainer/ray/vllm_engine.py

  1. Sets agent env vars: OPENRLHF_MODEL_PATH, OPENRLHF_PROMPT_CONSTRUCTION_MODE, OPENRLHF_MAX_STEPS
  2. Calls _load_agent_executor(agent_func_path):
    - Dynamically imports tool_calling_agent.py
    - Finds the AgentExecutor class (subclass of MultiTurnAgentExecutor)
    - Returns AgentExecutor() instance
  3. Creates the AsyncLLMEngine (vLLM)

  ---
  Phase 3: PPOTrainer Init — PPOTrainer.__init__()

  File: openrlhf/trainer/ppo_trainer.py

  1. Loads tokenizer via get_tokenizer()
  2. prepare_datasets() → creates PromptDataset dataloader from the JSONL file
  3. Creates SamplesGenerator(vllm_engines, ...) — coordinates rollout generation
  4. super().__init__() creates RemoteExperienceMaker(actor_model_group, ...) and FixedKLController(0)

  ---
  Phase 4: Main Training Loop — PPOTrainer.fit()

  for episode in range(num_episodes):
      while not exhausted:
          [A] rollout_samples = self.samples_generator.generate_samples(...)
          [B] status = self.train_step(rollout_samples)
          [C] self.save_logs_and_checkpoints(...)

  ---
  Phase 5: Rollout Generation (Step [A])

  Classes: SamplesGenerator → LLMRayActor → MultiTurnAgentExecutor

  1. SamplesGenerator._generate_vllm() pulls a batch of prompts from the dataloader
  2. _dispatch_prompts_to_vllm() load-balances prompts across vLLM engines:
  llm_engine.generate_responses.remote(prompt, label, sampling_params, n_samples=8)
  3. LLMRayActor.generate_responses() spawns 8 async tasks per prompt:
  [self.executor.execute(prompt, label, ...) for _ in range(8)]

  ---
  Phase 6: Multi-Turn Agent Loop — MultiTurnAgentExecutor.execute()

  File: openrlhf/utils/agent.py

  This is the core multi-turn loop for each sample:

  agent = ToolCallAgent()                      # Fresh instance per sample
  observation = agent.reset(prompt, label)      # → AgentSession.initialize()
  current_tokens = tokenize(observation)
  action_ranges = []

  while steps < max_steps:
      # 1. vLLM generates action tokens
      action_tokens = vllm.generate(current_tokens)

      # 2. Record action range (LLM-generated tokens only)
      action_ranges.append((len(current_tokens), len(current_tokens) + len(action_tokens)))

      # 3. Agent processes action, executes tools
      result = agent.step(action_text, label)   # → AgentSession.step()

      # 4. Concatenate: current_tokens + action_tokens + env_feedback_tokens
      current_tokens = current_tokens + action_tokens + tokenize(env_feedback)

      # 5. Log probs: real for action tokens, 0.0 for env feedback tokens
      rollout_log_probs += [real logprobs] + [0.0 * len(env_tokens)]

      if result["done"]: break

  return {observation_tokens, action_ranges, reward, rollout_log_probs}

  ---
  Phase 7: Inside the Agent — ToolCallAgent → AgentSession → GLMFlashProtocol

  ToolCallAgent.__init__() (tool_calling_agent.py):
  - Loads tokenizer from OPENRLHF_MODEL_PATH
  - Creates GLMFlashProtocol(tokenizer) — handles XML tool format
  - Creates AgentSession(protocol, tools, system_prompt)

  AgentSession.step(action_text, label) (agent_session.py):
  1. GLMFlashProtocol.parse_assistant_text(action_text) — parses <tool_call>func<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>
  2. If tool call found: executes the tool function, renders feedback via protocol.render_messages(), returns {feedback, reward=0, done=False}
  3. If no tool call (final answer): computes reward, returns {feedback="", reward=score, done=True}

  ---
  Phase 8: Building Experience Objects

  SamplesGenerator._process_response_into_experience() (experience_maker.py):

  # Build action_mask from action_ranges
  action_mask = torch.zeros(len(trajectory_tokens))
  for start, end in action_ranges:
      action_mask[start:end] = 1    # Only LLM actions get mask=1

  # Return Experience object
  Experience(sequences, attention_mask, action_mask, rewards, rollout_log_probs, ...)

  Dynamic filtering: If avg_score across the 8 samples is outside [0.2, 0.8], drop the prompt and fetch a new one.

  ---
  Phase 9: Training Step (Step [B])

  BasePPOTrainer.train_step(rollout_samples):

  1. RemoteExperienceMaker.make_experience_batch():
    - Runs PolicyModelActor.forward() to get fresh action_log_probs (actor forward pass)
    - No critic/ref/reward forward passes needed
    - compute_advantages_and_returns(): For dr_grpo, advantages = rewards - mean(rewards) within each group of 8 samples (no std normalization)
  2. Push experiences to PolicyModelActor replay buffers
  3. ppo_train() → ActorPPOTrainer.ppo_train():
  for epoch in range(2):       # max_epochs=2
      for experience in dataloader:
          training_step(experience)

  ---
  Phase 10: Loss Computation with Token-Level Masking

  ActorPPOTrainer.training_step() (ppo_actor.py):

  action_log_probs = actor(sequences, action_mask, attention_mask)
  loss = PolicyLoss(action_log_probs, old_log_probs, advantages, action_mask)

  PolicyLoss.forward() (openrlhf/models/loss.py):

  ratio = exp(log_probs - old_log_probs)
  surr1 = ratio * advantages
  surr2 = clamp(ratio, 1-eps, 1+eps) * advantages
  loss = -min(surr1, surr2)

  # CRITICAL: Only LLM action tokens contribute to the gradient
  loss = masked_mean(loss, action_mask)
  #       = (loss * action_mask).sum() / action_mask.sum()

  After the optimizer step, updated weights are broadcast back to vLLM engines via broadcast_to_vllm().

  ---
  Visual Summary

  train_grpo_tdc.sh
    └─ python -m openrlhf.cli.train_ppo_ray
         ├─ LLMRayActor (vLLM + AgentExecutor)
         ├─ RayActorGroup(PolicyModelActor)
         └─ PPOTrainer.fit()
              │
              ├─ [ROLLOUT] SamplesGenerator
              │    └─ LLMRayActor.generate_responses()
              │         └─ MultiTurnAgentExecutor.execute()  ← multi-turn loop
              │              ├─ ToolCallAgent.reset()
              │              │    └─ AgentSession.initialize()
              │              │         └─ GLMFlashProtocol.render_messages()
              │              └─ loop:
              │                   ├─ vLLM generate → action_tokens
              │                   ├─ action_ranges.append((start, end))
              │                   ├─ ToolCallAgent.step()
              │                   │    └─ AgentSession.step()
              │                   │         ├─ GLMFlashProtocol.parse_assistant_text()
              │                   │         └─ execute tool / compute reward
              │                   └─ concat tokens, accumulate log probs
              │
              ├─ [EXPERIENCE] _process_response_into_experience()
              │    └─ action_mask[start:end] = 1 for each action_range
              │
              ├─ [ADVANTAGE] RemoteExperienceMaker.make_experience()
              │    ├─ PolicyModelActor.forward() → action_log_probs
              │    └─ dr_grpo: advantages = rewards - mean(rewards per group)
              │
              └─ [TRAIN] ActorPPOTrainer.training_step()
                   └─ PolicyLoss: masked_mean(-min(surr1, surr2), action_mask)
                        └─ Only LLM-generated tokens affect gradients