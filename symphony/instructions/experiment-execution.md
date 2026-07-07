# Experiment Execution

Do not launch long-running GPU work unless the issue asks for a run.

Mimas/local execution is acceptable for short bounded setup checks and the
current MOSS 4B target-only smoke/evaluation path. Use one visible GPU for
MOSS 4B unless a human asks for another model or execution target.

For longer Mimas GPU experiments, use the cooperative GPU queue at:

```text
/store/store5/software/simple-gpu-schedule/with-gpu
```

Launch long jobs in durable detached `screen` sessions with log files. Do not
spend agent turns waiting for queued or long-running experiments to complete.

Every queued long experiment should have a completion callback or finalizer
path before it is queued. The callback/finalizer should record success or
failure evidence, log path, output path, and residual risk, then move the issue
back to `Todo` so Symphony can resume finalization.

Do not queue a long experiment if the launched code lacks this callback or
finalizer path. First add or fix the hook, then validate the actual wrapper or
finalizer with the smallest practical smoke test.

Use Stanage only when the issue or a later human Linear comment directly asks
for Stanage/HPC/Slurm. Do not infer Stanage from GPU size or queue length. If
Stanage is requested, use short bounded SSH commands from Mimas and submit
meaningful compute through Slurm; do not run meaningful compute on a Stanage
login node.

Keep stdout/stderr and generated artifacts under durable issue-specific paths
when a run is too large for Git. Record exact commands, input request paths,
model paths, output paths, logs, branch, commit, and completion checks in the
Linear queue or completion comment.

Never write scratch or generated artifacts under `/tmp` or `/home`.
