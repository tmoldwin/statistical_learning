# Dale's law branch (2× hidden size)

Capacity on this branch is doubled vs unconstrained baselines:
- demo `eight_word_ate_at_demo_ns`: H=100 (was 50)
- mixed `mixeddfa_rXX_ns`: H=200 (was 100)

Train:
```powershell
python scripts/train_dale_paper.py --demo --force
python scripts/train_dale_paper.py --mixed --seeds 1 --force
```

Home resume checklist: see repo-root `DALE_RESUME.md`.
