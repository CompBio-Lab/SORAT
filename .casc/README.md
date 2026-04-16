# Local CASC User Config

Create a local, gitignored user config at:

.casc/user.config

Generate it interactively:

python3 bin/casc_setup.py

Then run CASC with the local override file:

nextflow run main.nf -c .casc/user.config -profile local
nextflow run main.nf -c .casc/user.config -profile slurm

If your data paths or allocation changes, rerun the setup wizard with --force.
