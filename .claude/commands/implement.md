Implement a component of SigmaTau. Follow this workflow:

1. Read the legacy code in `matlab/legacy/` for the component named: $ARGUMENTS
2. Understand the algorithm — what SP1065 equation it implements, what the inputs/outputs are
3. Write the refactored version following the architecture in CLAUDE.md and relevant skills
4. Write a test that loads the same input, runs both legacy and refactored, and asserts < 1e-12 relative error
5. Run the test and fix any failures
6. Commit with message: "refactor: [component] — [one-line description]"

IMPORTANT: Do NOT proceed to step 6 unless tests pass. If stuck after 2 attempts, stop and explain what's failing.
