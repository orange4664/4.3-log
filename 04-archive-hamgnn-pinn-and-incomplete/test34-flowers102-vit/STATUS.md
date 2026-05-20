# test34-flowers102-vit status

- benchmark type: harder non-HamGNN real-data vision benchmark
- model type: ViT on Flowers102 at `224x224`
- current state:
  - scaffolded locally
  - synced to cluster path `/data/run01/scwb923/4.3-log/test34-flowers102-vit`
  - smoke submitted on `gpu_h100` as job `73445`
  - smoke failed before training because `torchvision.datasets.Flowers102`
    download timed out on the cluster
  - server-side offline provisioning is in progress, but the first downloaded
    `102flowers.tgz` archive is incomplete and must be redownloaded fully
  - background resumable server-side download is now running for
    `102flowers.tgz`
- acceptance gate: requires original `muon_ns` to beat `adamw`
- next action:
  - finish provisioning the dataset outside the training job
  - rerun smoke with local cached data only
