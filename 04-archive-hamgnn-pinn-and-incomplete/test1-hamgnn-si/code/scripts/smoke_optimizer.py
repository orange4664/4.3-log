import os, torch
from hamgnn_rt.optimizers import HamGNNMuonFDTOptimizer

torch.manual_seed(0)
lin = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Linear(16, 4))
for mode in ["muon_stream", "rt_v43_stream", "muon_stream_fdt_metric", "rt_v6_fdt_metric"]:
    opt = HamGNNMuonFDTOptimizer(lin.parameters(), lr=1e-3, mode=mode, diag_dir="/tmp/hamgnn_rt_smoke", name=mode, log_interval=1)
    x = torch.randn(12, 8); y = torch.randn(12, 4)
    for _ in range(3):
        opt.zero_grad()
        loss = (lin(x) - y).pow(2).mean()
        loss.backward(); opt.step()
    print(mode, float(loss.detach()))
