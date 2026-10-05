# Object-ball outgoing guideline research

This later pipeline detects the localized white guideline belonging to the target object ball after cue-ball contact. The cue-ball aiming line is negative material.

Read [the guide](../docs/OBJECT_BALL_GUIDELINE.md) and the preserved [canonical progress record](docs/experiments/supervised_target_line_progress.md).

The portable entry point is `python predict.py --input PATH --output DIRECTORY`. It loads `runs/supervised_best_manifest.json` and resolves its checkpoint dependencies against this package directory. Historical shell and PowerShell launchers retain workstation-specific paths; they are preserved as research evidence.

The old `pyproject.toml` referred to a removed `src.main` entry point and is archived under `docs/historical_pyproject.toml`. Run the current modules directly, as shown in the guide.
