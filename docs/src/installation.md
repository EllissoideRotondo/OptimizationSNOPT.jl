```@meta
CurrentModule = OptimizationSNOPT
```

# Installation

OptimizationSNOPT.jl needs two parts: the Julia package, installed with the package
manager, and the SNOPT shared library, available from the SNOPT distributors. The
library setup is the same as for SNOPT.jl, which OptimizationSNOPT.jl uses to call
SNOPT.

## Requirements

- Julia 1.11 or later.
- A licensed SNOPT 7.7 shared library, `libsnopt7`, that includes the C interface
  from [snopt-interface](https://github.com/snopt/snopt-interface).

## Install the package

```julia
using Pkg
Pkg.add("OptimizationSNOPT")
```

This also installs SNOPT.jl as a dependency.

## Make the library available

Set the environment variable `SNOPTDIR` to the directory that contains `libsnopt7`,
before starting Julia.

Linux and macOS (bash or zsh):

```bash
export SNOPTDIR=/path/to/snopt/lib
```

Windows (PowerShell):

```powershell
$env:SNOPTDIR = "C:\path\to\snopt\lib"
```

## Check the setup

This command prints the path of the library that SNOPT.jl found. For a package
installed in a project environment, add `--project=/path/to/project` after `julia`.

```bash
julia -e 'using OptimizationSNOPT; S = OptimizationSNOPT.SNOPT; @assert S.has_snopt(); println(S.libsnopt7)'
```

In PowerShell:

```powershell
julia -e "using OptimizationSNOPT; S = OptimizationSNOPT.SNOPT; @assert S.has_snopt(); println(S.libsnopt7)"
```

If the check fails, the
[SNOPT.jl installation guide](https://EllissoideRotondo.github.io/SNOPT.jl/dev/installation/)
gives the library search order, the supported platforms and the troubleshooting
steps.
