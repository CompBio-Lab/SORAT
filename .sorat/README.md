# Local SORAT User Config

Create a local, gitignored user config at:

.sorat/user.config

Generate it interactively:

python3 bin/sorat_setup.py

Then run SORAT with the local override file:

nextflow run main.nf -profile local
nextflow run main.nf -profile slurm

`.sorat/user.config` is auto-loaded when present, and can store generic defaults
such as `default_inputs.sax` and `default_inputs.atrial`.

If your data paths or allocation changes, rerun the setup wizard with --force.
