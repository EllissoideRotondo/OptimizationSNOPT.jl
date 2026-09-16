# SNOPT Documentation and Quality Design

## Goal

Make SNOPT.jl and OptimizationSNOPT.jl easy to install, run, and maintain.
Apply one clear writing and software-quality standard across both packages.

## Scope

This work covers both package repositories:

- `OptimizationSNOPT.jl` provides the Optimization.jl integration.
- `SNOPT.jl` provides the direct Julia interface to SNOPT.

The audit covers guides, examples, public docstrings, code comments, user-facing
errors, public input validation, and tests. Stable public interfaces remain
unchanged unless the audit finds a correctness defect.

## Writing standard

Documentation uses concise technical English. Sentences usually contain fewer
than 20 words. Each sentence has one main idea. Necessary technical terms are
defined when they first appear. Names and definitions stay consistent across
both packages.

Instructions lead with the result, then show exact commands. Tables are used
only when they make repeated mappings or results easier to compare. Examples
are copyable and state their working directory and prerequisites.

## Documentation structure

Each package README provides this path:

1. Decide which package to use.
2. Obtain a licensed SNOPT library.
3. Install the Julia package.
4. Configure library discovery.
5. Verify that Julia finds the library.
6. Run the smallest useful example.
7. Find advanced guides, examples, and tests.

OptimizationSNOPT.jl gains a small Documenter site. It covers installation,
quick start, constrained problems, solver configuration, tracing, concurrency,
and the public API. SNOPT.jl's existing site is edited to match the same terms
and task order.

## Code contracts and tests

Public boundaries state relevant preconditions, results, side effects, and
invariants. Runtime checks remain plain Julia checks with specific errors.
Internal comments explain constraints and reasons, not visible syntax.

Tests use Julia's `Test` standard library. Repeated input classes provide
property-style coverage where useful. Test-set names describe behavior as
observable scenarios. Python-only tools are not added. Formal proof tools are
not added because they do not support this Julia codebase.

The following invariants receive special attention:

- SNOPT owns one active Fortran workspace per Julia process.
- Both packages serialize workspace creation and solves.
- User callbacks must obey their documented array shapes and mutation rules.
- Option keys and values are normalized before they reach SNOPT.
- Bounds, constraint arrays, and derivative arrays must have compatible sizes.

## Persistent guidance

Both repositories receive matching `AGENTS.md` and `CLAUDE.md` files. These
files store the approved writing, contract, testing, and verification rules.
They apply to future code and documentation work in either repository.

## Verification

Verification includes:

- Both Julia test suites without a solver library.
- Solver-backed tests when `SNOPT.has_snopt()` is true.
- Both Documenter builds.
- Example syntax and documented command checks.
- Link, placeholder, terminology, and sentence-length audits.
- A final diff review for accidental API or behavior changes.

Checks that require a licensed library are reported separately when the library
is unavailable.
