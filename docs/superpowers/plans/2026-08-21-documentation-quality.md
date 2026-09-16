# SNOPT Documentation Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make SNOPT.jl and OptimizationSNOPT.jl clear to install, run, and maintain.

**Architecture:** Give both packages one task-oriented documentation path and one persistent quality standard. Preserve public APIs while strengthening public contracts, examples, and Julia-native tests where the audit finds gaps.

**Tech Stack:** Julia 1.10, Documenter.jl, Julia `Test`, Markdown, GitHub Actions

**Spec:** `docs/superpowers/specs/2026-08-21-documentation-quality-design.md`

## Global Constraints

- Use concise technical English, usually under 20 words per sentence.
- Give each sentence one main idea.
- Define necessary technical terms when first used.
- Preserve stable public interfaces unless a correctness defect requires a change.
- Use Julia-native validation and tests.
- Do not add Python-only contract, property-test, or proof tools.
- State commands, prerequisites, results, side effects, and invariants explicitly.

---

### Task 1: Persistent repository guidance

**Files:**

- Create: `AGENTS.md`
- Create: `CLAUDE.md`
- Create: `../SNOPT/AGENTS.md`
- Create: `../SNOPT/CLAUDE.md`

**Interfaces:**

- Consumes: The approved design specification.
- Produces: Matching instructions for future Codex and Claude work.

- [ ] **Step 1: Write the shared guidance**

Include exact rules for writing, contracts, tests, examples, and verification.
State that public terminology must match across both packages.

- [ ] **Step 2: Compare all four files**

Run:

```bash
diff -u AGENTS.md CLAUDE.md
diff -u AGENTS.md ../SNOPT/AGENTS.md
diff -u AGENTS.md ../SNOPT/CLAUDE.md
```

Expected: no output.

- [ ] **Step 3: Commit the guidance**

```bash
git add AGENTS.md CLAUDE.md
git commit -m "docs: store writing and quality standards"
git -C ../SNOPT add AGENTS.md CLAUDE.md
git -C ../SNOPT commit -m "docs: store writing and quality standards"
```

### Task 2: OptimizationSNOPT user path

**Files:**

- Modify: `README.md`
- Modify: `examples/hs71.jl`
- Create: `docs/Project.toml`
- Create: `docs/make.jl`
- Create: `docs/src/index.md`
- Create: `docs/src/installation.md`
- Create: `docs/src/quickstart.md`
- Create: `docs/src/configuration.md`
- Create: `docs/src/api.md`

**Interfaces:**

- Consumes: `OptimizationProblem`, `OptimizationFunction`, `solve`, and `SnoptOptimizer`.
- Produces: Copyable installation, verification, solve, trace, and test commands.

- [ ] **Step 1: Write the task-oriented README**

Lead with package selection and licensing. Show `Pkg.add("OptimizationSNOPT")`,
`SNOPT.has_snopt()`, one unconstrained solve, the full example command, and test
commands. Link to SNOPT.jl for low-level use.

- [ ] **Step 2: Clean the worked example**

Keep all three derivative approaches. Remove decorative separators and stale
numbered comments. Use consistent spacing and explain each derivative choice.

- [ ] **Step 3: Add the Documenter project and build script**

Use `Documenter.makedocs` with these pages: Home, Installation, Quick start,
Configuration, and API. Enable `checkdocs = :exports` and strict doctests only
when examples do not require a licensed solver.

- [ ] **Step 4: Write the guide pages**

Reuse tested source files with `@example` only for library-free checks.
Use plain `julia` fences for solver-backed examples. Explain every abbreviation
on first use.

- [ ] **Step 5: Build the site**

Run:

```bash
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

Expected: Documenter completes without warnings.

### Task 3: OptimizationSNOPT contracts and behavior tests

**Files:**

- Modify: `src/OptimizationSNOPT.jl`
- Modify: `src/cache.jl`
- Modify: `src/callback.jl`
- Modify: `test/runtests.jl`
- Modify: `test/additional_tests.jl`

**Interfaces:**

- Consumes: Existing public types and validation helpers.
- Produces: Clear public contracts and property-style validation coverage.

- [ ] **Step 1: Add failing contract tests for any discovered gap**

Use table-driven loops for option classes, trace frequencies, bound shapes, and
callback outcomes. Name test sets as observable scenarios.

- [ ] **Step 2: Run the focused tests and confirm the failure**

Run:

```bash
julia --project=. test/runtests.jl
```

Expected: new tests fail only for the identified contract gap.

- [ ] **Step 3: Apply the smallest code or documentation fix**

Add a precise check when behavior is unsafe. Otherwise, document the existing
precondition, result, side effect, or process-wide invariant.

- [ ] **Step 4: Run the package tests**

Run:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Expected: all library-free tests pass. Solver tests pass or report a clear skip.

### Task 4: SNOPT user path and public contracts

**Files:**

- Modify: `../SNOPT/README.md`
- Modify: `../SNOPT/docs/src/index.md`
- Modify: `../SNOPT/docs/src/installation.md`
- Modify: `../SNOPT/docs/src/interface.md`
- Modify: `../SNOPT/docs/src/examples.md`
- Modify: `../SNOPT/docs/src/lowlevel.md`
- Modify: `../SNOPT/docs/src/api.md`
- Modify: `../SNOPT/examples/hs71.jl`
- Modify: `../SNOPT/examples/unconstrained.jl`
- Modify: public docstrings under `../SNOPT/src/`
- Modify: `../SNOPT/test/snopt_tests.jl`

**Interfaces:**

- Consumes: `has_snopt`, `snopt`, `SnoptA`, `SnoptB`, `SnoptC`, callbacks,
  workspace functions, and option functions.
- Produces: One consistent high-level and low-level user path with explicit contracts.

- [ ] **Step 1: Audit the existing guides against the user path**

Remove repetition and inconsistent terms. Keep one canonical explanation for
library discovery, callback mutation, workspace ownership, and start modes.

- [ ] **Step 2: Rewrite the README and guide pages**

Show exact install, verify, example, and test commands. Explain when to choose
SNOPT.jl instead of OptimizationSNOPT.jl.

- [ ] **Step 3: Clean examples and public docstrings**

State array shapes, mutation requirements, return values, side effects, and
workspace lifecycle rules. Keep implementation details out of user guides.

- [ ] **Step 4: Add property-style tests for discovered validation gaps**

Use Julia loops and `@testset` cases. Do not change the foreign-function
interface unless a failing test proves a defect.

- [ ] **Step 5: Build docs and run tests**

Run:

```bash
julia --project=../SNOPT/docs -e 'using Pkg; Pkg.develop(path="../SNOPT"); Pkg.instantiate()'
julia --project=../SNOPT/docs ../SNOPT/docs/make.jl
julia --project=../SNOPT -e 'using Pkg; Pkg.test()'
```

Expected: docs build cleanly. Tests pass or clearly skip licensed solver cases.

### Task 5: Cross-package verification and review

**Files:**

- Modify only files needed to correct findings from verification.

**Interfaces:**

- Consumes: Both completed documentation and code audits.
- Produces: Evidence that commands, terms, links, and APIs agree.

- [ ] **Step 1: Run mechanical documentation checks**

Search for placeholders, undefined abbreviations, stale commands, inconsistent
option names, and unusually long prose sentences.

- [ ] **Step 2: Check both diffs**

Run:

```bash
git diff --check
git -C ../SNOPT diff --check
git diff --stat
git -C ../SNOPT diff --stat
```

Expected: no whitespace errors and no unrelated changes.

- [ ] **Step 3: Run final package and documentation verification**

Repeat both test suites and both documentation builds after corrections.

- [ ] **Step 4: Commit the completed audits**

```bash
git add README.md examples src test docs AGENTS.md CLAUDE.md
git commit -m "docs: clarify setup and solver workflow"
git -C ../SNOPT add README.md docs examples src test AGENTS.md CLAUDE.md
git -C ../SNOPT commit -m "docs: clarify setup and public contracts"
```
