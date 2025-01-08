This project uses conda or its alternatives like mamba or micromamba to setup and manage your python environment.

To install the environment from scratch you can run `micromamba env create -f environment.yaml`,
which will install the dependencies defined in environment.yaml

Or, to install a completely reproducible environment with pinned versions, you can install from the _conda.lock_ file via
`micromamba create -adaptation_modelling_env -f conda-lock.yml`.
