# test8-hamgnn-toy-muonns status

This folder adds a HamGNN toy Hamiltonian benchmark chosen specifically because
historical server results suggest original `muon_ns` can slightly beat `adamw`.

Acceptance rule:

- only accept this benchmark if original `muon_ns` beats `adamw`

This benchmark is separate from the PINN/PDE screening folders and returns to a
HamGNN task where matrix-parameter geometry is more central.
