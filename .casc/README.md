# Local CASC User Config

Create a local, gitignored user config at:

.casc/user.config

Generate it interactively:

python3 bin/casc_setup.py

Then run CASC with the local override file:

nextflow run main.nf -profile local
nextflow run main.nf -profile slurm

`.casc/user.config` is auto-loaded when present, and can store generic defaults
such as `default_inputs.sax` and `default_inputs.atrial`.

If your data paths or allocation changes, rerun the setup wizard with --force.
