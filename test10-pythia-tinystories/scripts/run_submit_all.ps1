$methods = @("adamw", "muon_ns", "rt_v43_stream", "rt_v43_ns", "rt_v6_fdt_metric")
foreach ($m in $methods) {
  ssh nmcc-n46h1 "cd /data/run01/scwb923/4.3-log/test10-pythia-tinystories && METHOD=$m ROOT=/data/run01/scwb923/4.3-log/test10-pythia-tinystories sbatch scripts/submit_pythia_test10_a800.sbatch"
}
