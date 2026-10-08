# Workshop 9 — defect lab

This lab does not have its own copy of the service. The service it breaks is
the finished one in `workshops/assistant/after`, because that is the tree the
Dockerfile builds.

## Do this

```bash
cd ../../after
mv defects/test_regressions.py defects/test_regressions.reference.py
cp ../phases/09-defect-lab/test_regressions.py defects/test_regressions.py
make defect-lab
```

The first run is not green. That is the start. Do not edit `variants.py`.
When you are stuck, diff your tests against `defects/test_regressions.reference.py`.
