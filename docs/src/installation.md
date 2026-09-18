# Installation

## Requirements

Install Julia 1.10 or later. Obtain a licensed SNOPT 7 shared library that
includes the C API provided by
[snopt-interface](https://github.com/snopt/snopt-interface).

You must obtain the SNOPT library and license separately.

## Add the package

For a registry installation:

```julia
import Pkg
Pkg.add("OptimizationSNOPT")
```

For source development, clone both repositories into one directory:

```bash
git clone https://github.com/EllissoideRotondo/SNOPT.jl SNOPT
git clone https://github.com/EllissoideRotondo/OptimizationSNOPT.jl OptimizationSNOPT
julia --project=OptimizationSNOPT -e 'using Pkg; Pkg.develop(path="SNOPT"); Pkg.instantiate()'
```

## Configure the library

Set `SNOPTDIR` to the directory containing `libsnopt7`.

Linux and macOS:

```bash
export SNOPTDIR=/path/to/snopt/lib
```

Windows PowerShell:

```powershell
$env:SNOPTDIR = "C:\path\to\snopt\lib"
```

Some SNOPT distributions also require `SNOPT_LICENSE`. Follow the vendor's
instructions for your license type.

## Verify the setup

From the OptimizationSNOPT.jl repository, verify library discovery:

```bash
julia --project=. -e 'using SNOPT; @assert SNOPT.has_snopt(); println("SNOPT is ready")'
```

`SNOPT.has_snopt()` returns `true` when Julia finds a compatible library.
The [SNOPT.jl installation guide](https://EllissoideRotondo.github.io/SNOPT.jl/dev/installation/)
lists filenames, fallback search paths, and platform limits.

## Test the package

Run all tests from the repository root:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

The full solver suite runs when the shared library is available.
