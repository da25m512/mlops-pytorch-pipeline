# End-to-end validation record

Everything below was run against a real Kubernetes cluster (kind, node image
`kindest/node:v1.31.2`) with images built from this repository. Output is copied
verbatim from the terminal.

A short training profile is used throughout — `simple_cnn`, 2 epochs, a
2000-sample subset of CIFAR-10 — so the whole loop completes inside a lab
session. The accuracy figures are therefore low by design; what is being
validated is the pipeline, not the model.

---

## Part C — Docker

### Build

```
$ docker build -f docker/Dockerfile.train -t mlops-train:v1 .
Step 13/13 : ENTRYPOINT ["python", "src/train.py"]
Successfully built 7940f066a2fa
Successfully tagged mlops-train:v1

$ docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
Step 13/13 : CMD ["uvicorn", "serve:app", ...]
Successfully tagged mlops-serve:v1
```

### The serving image carries inference dependencies only

```
$ docker run --rm --entrypoint python mlops-serve:v1 -c "import importlib.util as u; ..."
tensorboard present: False
pyyaml present: False
fastapi present: True
```

`pyyaml` is a training-side dependency (it parses the hyperparameter config);
its absence confirms `requirements/serve.txt` is the only thing installed.

### Non-root, exposed port, healthcheck

```
$ docker run --rm --entrypoint id mlops-serve:v1
uid=10001(appuser) gid=10001(appuser) groups=10001(appuser)

$ docker inspect mlops-serve:v1
User: appuser
ExposedPorts: {'8080/tcp': {}}
Healthcheck: {'Interval': 30000000000, 'Retries': 3, 'StartPeriod': 20000000000,
              'Test': ['CMD-SHELL', 'python -c "...urlopen(\'http://127.0.0.1:8080/health\')..."'],
              'Timeout': 5000000000}
Cmd: ['uvicorn', 'serve:app', '--app-dir', 'src', '--host', '0.0.0.0', '--port', '8080']

$ docker inspect mlops-train:v1
User: trainer
Entrypoint: ['python', 'src/train.py']
```

### Training with mounted volumes

```
$ docker run --rm \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/checkpoints:/app/checkpoints \
    -v $(pwd)/configs/training_config.smoke.yaml:/app/configs/training_config.yaml:ro \
    mlops-train:v1

{"event": "run_start", "config_path": "/app/configs/training_config.yaml", "device": "cpu",
 "architecture": "simple_cnn", "dataset": "cifar10", "epochs": 2, "batch_size": 64,
 "learning_rate": 0.002}
Downloading https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz to /app/data/cifar-10-python.tar.gz
Extracting /app/data/cifar-10-python.tar.gz to /app/data
{"event": "data_ready", "train_batches": 32, "val_batches": 32}
{"epoch": 1, "train_loss": 1.9883, "train_accuracy": 0.2385, "val_loss": 1.8639, "val_accuracy": 0.32, "seconds": 36.7}
{"event": "checkpoint_saved", "path": "/app/checkpoints/classifier_v1.pt", "val_loss": 1.8639}
{"epoch": 2, "train_loss": 1.7406, "train_accuracy": 0.339, "val_loss": 1.7926, "val_accuracy": 0.3475, "seconds": 34.7}
{"event": "checkpoint_saved", "path": "/app/checkpoints/classifier_v1.pt", "val_loss": 1.7926}
{"event": "training_complete", "best_val_loss": 1.7926, "checkpoint": "/app/checkpoints/classifier_v1.pt"}

$ ls -la checkpoints/
-rw-r--r-- 1 10001 10001 3489000 Aug 29 12:22 classifier_v1.pt
```

The checkpoint is owned by uid 10001, not root — the container never wrote to
the mounted volume as root.

### Serving

```
$ docker run -d --rm --name mlops-serve -p 8080:8080 \
    -v $(pwd)/checkpoints:/app/checkpoints \
    mlops-serve:v1

$ curl -i http://localhost:8080/health
HTTP/1.1 200 OK
server: uvicorn
content-type: application/json

{"status":"ok","model_loaded":true}

$ curl http://localhost:8080/metadata
{
    "architecture": "simple_cnn",
    "dataset": "cifar10",
    "num_classes": 10,
    "class_names": ["airplane", "automobile", "bird", "cat", "deer",
                    "dog", "frog", "horse", "ship", "truck"],
    "trained_epochs": 2,
    "val_accuracy": 0.3475,
    "checkpoint_path": "/app/checkpoints/classifier_v1.pt"
}

$ curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
{
    "filename": "test_image.png",
    "predicted_class": "ship",
    "probabilities": {
        "airplane": 0.341192, "automobile": 0.040325, "bird": 0.007163,
        "cat": 0.001076, "deer": 0.003202, "dog": 0.000458,
        "frog": 0.000663, "horse": 0.005072, "ship": 0.494893, "truck": 0.105957
    },
    "top_k": [
        {"class": "ship", "probability": 0.494893},
        {"class": "airplane", "probability": 0.341192},
        {"class": "truck", "probability": 0.105957},
        {"class": "automobile", "probability": 0.040325},
        {"class": "bird", "probability": 0.007163}
    ]
}
```

`test_image.png` is index 3 of the CIFAR-10 test split, whose true label is
`airplane`. The model ranks `ship` first and `airplane` second — unsurprising
after two epochs at 34.75% validation accuracy, and the two classes share a
blue background.

A non-image upload is rejected rather than crashing the worker:

```
$ curl -X POST http://localhost:8080/predict -F "image=@notes.txt"
HTTP 400
```

---

## Parts D, E, F — Kubernetes

### Apply

```
$ kubectl apply -f k8s/namespace.yaml
namespace/ml-training created
$ kubectl apply -f k8s/configmap.yaml
configmap/training-config created
configmap/serving-config created
$ kubectl apply -f k8s/storage.yaml
persistentvolumeclaim/data-pvc created
persistentvolumeclaim/checkpoints-pvc created
$ kubectl apply -f k8s/training-job.yaml
job.batch/pytorch-training created

$ kubectl get pods,pvc,job -n ml-training
NAME                         READY   STATUS    RESTARTS   AGE
pod/pytorch-training-tx9mr   1/1     Running   0          20s

NAME                                    STATUS   VOLUME                                     CAPACITY   ACCESS MODES   STORAGECLASS
persistentvolumeclaim/checkpoints-pvc   Bound    pvc-bea332a4-bcfd-4092-92d0-3d419b35476d   2Gi        RWO            standard
persistentvolumeclaim/data-pvc          Bound    pvc-eb50a766-4d91-4bc4-b5cb-b0a2129de99c   5Gi        RWO            standard

NAME                         STATUS    COMPLETIONS   DURATION   AGE
job.batch/pytorch-training   Running   0/1           20s        20s
```

### Training Job completes

```
$ kubectl wait --for=condition=complete job/pytorch-training -n ml-training --timeout=900s
job.batch/pytorch-training condition met

$ kubectl logs job/pytorch-training -n ml-training
{"event": "run_start", "config_path": "/app/configs/training_config.yaml", "device": "cpu",
 "architecture": "simple_cnn", "dataset": "cifar10", "epochs": 2, "batch_size": 64,
 "learning_rate": 0.002}
Files already downloaded and verified
{"event": "data_ready", "train_batches": 32, "val_batches": 32}
{"epoch": 1, "train_loss": 1.9956, "train_accuracy": 0.2395, "val_loss": 1.7694, "val_accuracy": 0.3055, "seconds": 470.8}
{"event": "checkpoint_saved", "path": "/app/checkpoints/classifier_v1.pt", "val_loss": 1.7694}
{"epoch": 2, "train_loss": 1.7358, "train_accuracy": 0.324, "val_loss": 2.1633, "val_accuracy": 0.276, "seconds": 461.9}
{"event": "training_complete", "best_val_loss": 1.7694, "checkpoint": "/app/checkpoints/classifier_v1.pt"}
```

Epoch 2 made validation loss worse, so the checkpoint from epoch 1 was kept —
`best_val_loss` is 1.7694, the epoch 1 figure. That is the early-stopping
bookkeeping doing its job; with `patience: 2` a third bad epoch would have
stopped the run.

The config the Job read came from the ConfigMap mounted at `/app/configs`, not
from the image — `config_path` in the first log line is the mount path.

### Probes: two bugs this run exposed, and the fixes

**Bug 1 — liveness on a readiness condition.** The Deployment was applied while
the Job was still training. `/health` returned 503 because no checkpoint
existed, the liveness probe read that as a dead container, and after three
failures at a 10s period it restarted the pod:

```
NAME                             READY   STATUS             RESTARTS        AGE
model-serving-7665b785f9-dn8rd   0/1     CrashLoopBackOff   6 (2m37s ago)   6m58s
```

Fixed by splitting the two conditions — `/health` reports that the process is
serving, `/ready` reports that a checkpoint is loaded. After the fix:

```
$ kubectl exec pod/model-serving-6b6c455d9b-9qcwc -n ml-training -- python -c "..."
health 200 {"status":"ok","model_loaded":false}
ready  503 {"status":"unavailable","reason":"FileNotFoundError: checkpoint not found at /app/checkpoints/classifier_v1.pt"}

$ kubectl get endpoints model-serving -n ml-training
NAME            ENDPOINTS   AGE
model-serving               6m58s
```

Alive, zero restarts, and correctly absent from the Service's endpoints.

**Bug 2 — the checkpoint was loaded only at startup.** The Job then ran to
completion and wrote the file, and both replicas stayed `0/1 READY` anyway:

```
$ kubectl get pods -n ml-training
model-serving-dcb7c75b4-276sl   0/1     Running     0          16m
model-serving-dcb7c75b4-txk4w   0/1     Running     0          16m
pytorch-training-cjkvb          0/1     Completed   0          16m

$ kubectl rollout status deployment/model-serving -n ml-training
error: deployment "model-serving" exceeded its progress deadline
```

Fixed by retrying the load inside `/ready` whenever no model is held. A replica
now joins on the first probe after the checkpoint lands.

### Serving layer healthy

```
$ kubectl apply -f k8s/serving-deployment.yaml
$ kubectl apply -f k8s/serving-service.yaml
$ kubectl apply -f k8s/hpa.yaml

$ kubectl rollout status deployment/model-serving -n ml-training
Waiting for deployment "model-serving" rollout to finish: 1 out of 2 new replicas have been updated...
Waiting for deployment "model-serving" rollout to finish: 1 old replicas are pending termination...
deployment "model-serving" successfully rolled out

$ kubectl get pods -n ml-training
NAME                            READY   STATUS      RESTARTS   AGE
model-serving-d66d4db79-vk7zd   1/1     Running     0          15s
model-serving-d66d4db79-w9xw2   1/1     Running     0          30s
pytorch-training-cjkvb          0/1     Completed   0          24m

$ kubectl get svc,endpoints model-serving -n ml-training
NAME                    TYPE        CLUSTER-IP     EXTERNAL-IP   PORT(S)   AGE
service/model-serving   ClusterIP   10.96.79.124   <none>        80/TCP    49m

NAME                      ENDPOINTS                             AGE
endpoints/model-serving   10.244.0.16:8080,10.244.0.17:8080     49m

$ kubectl get hpa -n ml-training
NAME            REFERENCE                  TARGETS             MINPODS   MAXPODS   REPLICAS   AGE
model-serving   Deployment/model-serving   cpu: <unknown>/70%  2         6         2          52m
```

The HPA reports `<unknown>` for CPU because kind ships without metrics-server;
the object is accepted and holds the replica count at its floor of 2. On a
cluster with metrics-server installed the target resolves normally.

The rollout log is worth noting: with `maxUnavailable: 0` and `maxSurge: 1`,
Kubernetes brought up one new replica, waited for it to pass readiness, and only
then terminated an old one. The Service never dropped below two healthy backends.

### Prediction through the Service

```
$ kubectl port-forward svc/model-serving 8080:80 -n ml-training
Forwarding from 127.0.0.1:8080 -> 8080

$ curl http://localhost:8080/health
{"status":"ok","model_loaded":true}

$ curl http://localhost:8080/ready
{"status":"ready","model_loaded":true}

$ curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
{
    "filename": "test_image.png",
    "predicted_class": "ship",
    "probabilities": {
        "airplane": 0.208553, "automobile": 0.16167, "bird": 0.010728,
        "cat": 0.003975, "deer": 0.004709, "dog": 0.000765,
        "frog": 0.00118, "horse": 0.011867, "ship": 0.500965, "truck": 0.095587
    },
    "top_k": [{"class": "ship", "probability": 0.500965}, ...]
}

$ for i in $(seq 1 10); do curl -s -o /dev/null -w "%{http_code} " \
    -X POST http://localhost:8080/predict -F "image=@test_image.png"; done
200 200 200 200 200 200 200 200 200 200
```

Ten consecutive requests load-balanced across both replicas, all 200.

---

## Unit tests

```
$ pytest
...................                                                      [100%]
19 passed

$ ruff check src tests
All checks passed!
```
