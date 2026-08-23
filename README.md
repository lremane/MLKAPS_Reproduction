# Reproducing MLKAPS on ScaLAPACK PDGEQRF: An Autotuning Case Study

This repository holds the code and configuration used to reproduce the [MLKAPS](https://github.com/MLCGO/MLKAPS) autotuning pipeline on the open-source ScaLAPACK `PDGEQRF` kernel.

The code for running MLKAPS on the `PDGEQRF` kernel was missing from the original MLKAPS repository. This repository provides that missing piece: the driver code and configuration needed to run MLKAPS on `PDGEQRF`, including the reformulation required to handle `PDGEQRF`'s constrained input parameters.

## Running the experiments

1. Clone the original MLKAPS repository, using the `dev` branch:
   ```bash
   git clone --branch dev https://github.com/MLCGO/MLKAPS.git
   ```
2. Move the `Scalapack-PDGEQRF` directory from this repository into the `examples` directory of the cloned MLKAPS repository.
3. Follow the normal installation instructions from the MLKAPS repository.
4. Run the experiment. If running on JURECA (where this was tested), the test results can be reproduced by first allocating exactly one node with 128 cores, then running from the `examples` directory:
   ```bash
   mlkaps Scalapack-PDGEQRF/pdgeqrf_mlkaps.json
   ```
   An example of how to allocate resources and execute MLKAPS on JURECA can be found in [`run_mlkaps.sh`](run_mlkaps.sh).

## Validation

[`validate_vs_default.py`](validate_vs_default.py) compares the results produced by the MLKAPS pipeline against a fixed default configuration.

## Results

The `Results` directory holds all data measured for the report.
