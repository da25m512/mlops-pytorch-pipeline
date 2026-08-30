# Reflection

## What was the most challenging part?

The hardest part was not writing the model or the manifests — it was the seam
between the training Job and the serving Deployment, and every mistake I made
lived in that seam.

The first version of `serve.py` loaded the checkpoint at import time and let a
missing file raise. Locally that looked fine, because I always ran training
first. On the cluster it was a disaster: the Deployment and the Job came up
together, no checkpoint existed, `/health` answered 503, and the liveness probe
read that as a dead container. Three failures at a 10s period later, both
replicas were in `CrashLoopBackOff` — 6 restarts in seven minutes — and each
restart reset the clock before the checkpoint could ever appear. The fix was to
stop conflating two different questions. `/health` now answers 200 whenever the
process is serving HTTP; `/ready` answers 503 until a checkpoint is loaded. The
readiness probe keeps an unready pod out of the Service's endpoints without
killing it. That single change turned a self-inflicted crash loop into an
ordinary startup delay.

It was not enough on its own, and the second failure was the more interesting
one. With liveness fixed, the pods stayed alive — and stayed `0/1 READY` even
after the Job completed and wrote the checkpoint, until the Deployment failed
its progress deadline. `load_checkpoint()` ran once, at startup, recorded the
`FileNotFoundError`, and never looked again. I had built a service that could
tell you it was not ready but could never become ready. Retrying the load inside
`/ready` fixed it: the call is a no-op once a model is held, so it costs one
filesystem stat per probe, and a replica joins the Service on the first probe
after the file lands. Together these two bugs taught me the real lesson of the
assignment — in Kubernetes, "broken", "not ready yet" and "will never be ready"
are three different states, and it is the application's job to distinguish them.

The second hard problem was the probe timings themselves. My first attempt used
the same values for liveness and readiness. Loading a ResNet-18 checkpoint takes
a few seconds, so the liveness probe fired before the model was in memory,
declared the container dead, and killed it — a self-inflicted crash loop with no
underlying bug. Separating the two, giving readiness a 15-second initial delay
and leaving liveness to catch genuine hangs, fixed it. The `maxUnavailable: 0`
rolling-update setting came from the same line of thinking: a rollout should
never be able to take the last healthy replica out of service.

Emitting metrics as one JSON object per line is the thing I would keep in any
future project. It costs nothing to write, and it means `kubectl logs` output
can be piped straight into `jq` without parsing prose.
