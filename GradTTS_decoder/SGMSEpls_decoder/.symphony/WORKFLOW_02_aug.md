---
tracker:
  kind: linear
  project_slug: "simplified-dm-f6102c55aad4"
  required_labels:
    - "dtdm-sgmse-metrics-20260802-sgmse"
  active_states:
    - Todo
    - In Progress
  terminal_states:
    - Done
    - Canceled
    - Duplicate
polling:
  interval_ms: 900000
workspace:
  root: /mnt/parscratch/users/acp23xt/private/codex_temp
agent:
  max_concurrent_agents: 1
  max_turns: 3
codex:
  command: >-
    bash -lc ". /opt/apps/testapps/common/software/staging/Anaconda3/2022.10/etc/profile.d/conda.sh &&
    conda activate /mnt/parscratch/users/acp23xt/private/conda/envs/symphony &&
    export LC_ALL=en_GB.utf8 LANG=en_GB.utf8 &&
    export MIX_REBAR3=/mnt/parscratch/users/acp23xt/private/conda/envs/symphony/bin/rebar3 &&
    exec codex --config shell_environment_policy.inherit=all --config model_reasoning_effort=xhigh app-server"
  approval_policy: never
  # The Symphony execution node cannot reliably create bubblewrap user
  # namespaces, so Codex turns run without the bwrap sandbox.
  thread_sandbox: danger-full-access
  turn_sandbox_policy:
    type: dangerFullAccess
---
No codex session should be executed in a login node.

Work autonomously on Linear issue `{{ issue.identifier }}` until it is genuinely complete.

The Codex thread and turn policies intentionally use `danger-full-access` /
`dangerFullAccess` because this cluster cannot reliably create bubblewrap user
namespaces. Do not change these policies back or repeatedly retry commands
through `bwrap`.

All file creation, modification, and deletion must remain within these three
approved directory trees:

- `/mnt/parscratch/users/acp23xt/private/DTDM_sgmse`
- `/mnt/parscratch/users/acp23xt/private/codex_utmosv2`
- `/mnt/parscratch/users/acp23xt/private/codex_whisper`

Paths outside those trees are read-only for this workflow. Slurm submissions,
scheduler/accounting queries, Linear updates, and other required non-filesystem
operations remain allowed.

The target repository is `/mnt/parscratch/users/acp23xt/private/DTDM_sgmse`; always run repository commands there. The Symphony-created workspace is only runtime scratch space. Preserve all pre-existing dirty-tree changes and never reset, revert, delete, or overwrite unrelated user work.

Use Slurm for every long-running computation and dependency-aware jobs for orchestration. Do not run training, inference, WER, UTMOSv2, or long monitoring loops on the node hosting Symphony. Read and follow the issue description exactly. Inspect job exit states and logs, retry only when safe, and do not treat failed/cancelled jobs as completed.

After submitting dependency-aware Slurm jobs, record their IDs and end the agent turn. Do not poll running jobs from Codex. Resume only after an external completion/failure event or at intervals of at least 15 minutes. A Linear tracker refresh is not a reason to invoke the model when no relevant state has changed.

After every submitted run.sh has finished its training phase, extract the training loss for every epoch from every run/sweep and save the combined, clearly labelled records to `/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/training_loss.log`. Preserve experiment, sweep/configuration, epoch, and loss fields, validate coverage against the completed training outputs, and perform any non-trivial extraction through a dependency-aware Slurm job.

Keep one Linear progress comment updated with job IDs, current stages, failures, and the final table. Move Todo to In Progress before work. When all requested folders and generated paths have been processed and the requested sample count/mean/std summary is posted in that comment and saved in the target repository, move the issue to Done. Do not stop while work remains unless a true external blocker prevents progress.
