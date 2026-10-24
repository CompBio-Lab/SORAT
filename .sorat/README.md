# Local settings

`bin/sorat_setup.py` writes your data paths, SLURM account and cache locations to `.sorat/user.config` in this folder. The file is gitignored and loaded automatically by `nextflow.config`, so the same command works on every machine:

```bash
python3 bin/sorat_setup.py           # first time
python3 bin/sorat_setup.py --force   # after your paths or allocation change
```

It can also hold defaults such as `default_inputs.sax` and `default_inputs.atrial`.
