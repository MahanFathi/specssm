
# Spectral State-Space Models

This is a reimplementation of [Spectral State-Space Models](https://arxiv.org/abs/2312.06837) in JAX/Flax. Here we only test Spectral SSMs on [Long-range Arena (LRA)](https://github.com/google-research/long-range-arena). 

* everything's configurable using `gin`
* you can turn off auto-regressive part of the model 
* multi-gpu -> data parallelization for bigger batches
* includes a stolen implementation of S5 as well
* monitor on `wandb`

## Run
The bash scripts inside the `./bin` directory are executable. Before everything download the datasets:
```bash
./bin/download_lra.sh
```

If you're at Mila, set up your virtualenv run:
```bash
sbatch launch.sh spec_listops
```

To run things inside a singularity container first pull the docker image from:
```bash
module load singularity
singularity pull docker://mahanfathi/specssm:v1.0
```
Set your `wandb` key inside `claunch.sh` and run:
```bash
# claunch is containerized via singularity
sbatch claunch.sh spec_listops
```