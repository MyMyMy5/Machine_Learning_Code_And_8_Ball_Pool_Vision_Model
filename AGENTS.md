# Publication working agreements

This repository preserves two distinct pool-vision systems. Read README.md and the relevant guide under docs/ before making changes.

- Keep full-frame line-extension labels and localized object-ball outgoing labels distinct.
- Preserve checkpoint files, source provenance, dataset identities, and historical logs.
- Consult object_ball_guideline/AGENTS.md and its canonical experiment records before changing the later target, inference policy, training, or promoted state.
- Run tools/verify_artifacts.py after modifying manifests or model packaging. Update the model registry and checksums only from actual artifacts.
- Publish weights through Git LFS. Do not replace a promoted model merely because an experiment has a newer file timestamp.
- New validation belongs in docs/VALIDATION.md; historical metrics are not new test results.
