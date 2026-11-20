# Rules for every agent

1. **Read `SPEC.md` first.** It is the contract: interfaces, file ownership, and experiment rules.
2. **Only edit files you own** (ownership table in `SPEC.md` §2). If you need something from a file you
   don't own, code against the spec'd interface and mention it in your report — don't touch the file.
3. **Never edit `SPEC.md`, `CLAUDE.md`, `requirements.txt`, `pytest.ini`, `.gitignore`.**
4. Don't change public signatures from `SPEC.md`. Adding optional keyword arguments is OK.
5. Fakes/mocks of other agents' modules live **only inside your own test files**.
6. Pure PyTorch + numpy + matplotlib (stdlib fine). No new dependencies, no HF libs, no `nn.Transformer`.
   CPU only, seed 1337, `torch.set_num_threads(4)`.
7. Run `python -m pytest` from the repo root before every push; it must pass on your branch.
8. Commit small and often with clear messages (`model: add causal self-attention`). Push to your branch.
9. At the end, report: what you built, how you tested it, anything left undone, and **any interface
   concerns** (ambiguities or changes you think the spec needs) for the planner/integrator.
