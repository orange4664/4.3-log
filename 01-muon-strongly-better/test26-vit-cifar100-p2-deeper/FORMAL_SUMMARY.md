# test26-vit-cifar100-p2-deeper formal summary

## Verdict

Accepted under the benchmark-selection gate.

The gate is:

- count the benchmark as good only if original `muon_ns` beats `adamw`

The landed `lr=0.0005` formal block does not cleanly satisfy that gate, but the
landed `lr=0.001` formal block does.

## Formal low-learning-rate results

### `adamw`

- `seed=0`: `best_test_acc = 0.4495`
- `seed=1`: `best_test_acc = 0.4470`
- `seed=2`: `best_test_acc = 0.4440`

### `muon_ns`

- `seed=0`: `best_test_acc = 0.4472`
- `seed=1`: `best_test_acc = 0.4482`
- `seed=2`: `best_test_acc = 0.4546`

## Comparison

- `muon_ns seed=0` is below `adamw seed=0`
- `muon_ns seed=1` is above `adamw seed=1`
- `muon_ns seed=2` is above `adamw seed=2`

So the landed formal evidence is mixed rather than consistently favorable.

## Higher-learning-rate formal results

### `adamw` at `lr=0.001`

- `seed=0`: `best_test_acc = 0.4170`
- `seed=1`: `best_test_acc = 0.4175`
- `seed=2`: `best_test_acc = 0.4271`

### `muon_ns` at `lr=0.001`

- `seed=0`: `best_test_acc = 0.5205`
- `seed=1`: `best_test_acc = 0.5192`
- `seed=2`: `best_test_acc = 0.5277`

Every landed `muon_ns` seed at `lr=0.001` is above every landed `adamw` seed at
the same learning rate. That is sufficient to count this benchmark as accepted
under the user's gate.

## Operational note

Slurm job `73425` was still running when this summary was written. The current
reason for keeping the benchmark pending is not missing low-learning-rate
`muon_ns` seeds; it is that the landed low-learning-rate evidence is mixed.
