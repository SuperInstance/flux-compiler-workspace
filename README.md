# flux-compiler-workspace

FLUX constraint compiler toolchain — parse constraint expressions, lower to IR, optimize, verify, and emit code for native, AVX-512, CUDA, WASM, eBPF, and RISC-V targets.

## What This Gives You

- **GUARD constraint parser** — Parse range, domain, and exact constraints with AND/OR/NOT combinators (Pest PEG grammar)
- **FLUX IR** — Intermediate representation with verification, halt reasons, and constraint modules
- **Multi-target codegen** — Emit for native, AVX-512, CUDA, WASM, eBPF, and RISC-V
- **Optimization passes** — IR-level constraint optimization
- **Verification** — Prove translation correctness from source to compiled output
- **CLI** — `fluxc` command-line compiler with compile, bench, show, and verify subcommands

## Quick Start

```bash
# Compile a constraint file
cargo run -p fluxc-cli -- compile -i constraints.guard -t native -o output.bin

# Show generated IR
cargo run -p fluxc-cli -- show -i constraints.guard

# Benchmark compiled constraints
cargo run -p fluxc-cli -- bench -i constraints.guard --iterations 10000

# Verify translation correctness
cargo run -p fluxc-cli -- verify -i constraints.guard --compiled output.bin
```

## Workspace Crates

| Crate | Description |
|-------|-------------|
| `fluxc-parser` | GUARD constraint parser (Pest PEG grammar) |
| `fluxc-ast` | Abstract syntax tree for FLUX constraints |
| `fluxc-ir` | Intermediate representation with verification |
| `fluxc-optimize` | IR optimization passes |
| `fluxc-codegen` | Multi-target code generation |
| `fluxc-verify` | Translation correctness verification |
| `fluxc-cli` | Command-line interface |

## API Reference

### Parser

```rust
use fluxc_parser::parse;

let expr = parse("x RANGE [0, 100] AND NOT (y DOMAIN 0xFF)")?;
```

### IR

```rust
use fluxc_ir::IrModule;

let module = IrModule::from_ast(ast);
module.verify()?; // Validate IR invariants
```

### Codegen

```rust
use fluxc_codegen::{codegen, Target};

let output = codegen(&module, Target::Native)?;
```

## How It Fits

Part of the [SuperInstance](https://github.com/SuperInstance) constraint theory ecosystem. The FLUX compiler sits between high-level constraint specifications and executable constraint checks:

- **Input**: Constraint DSL or GUARD expressions
- **Output**: Optimized constraint checks for any target platform
- **Related**: [constraint-dsl](https://github.com/SuperInstance/constraint-dsl), [flux-verify-api](https://github.com/SuperInstance/flux-verify-api), [constraint-theory-core](https://github.com/SuperInstance/constraint-theory-core)

## Testing

```bash
cargo test --workspace
```

## Installation

```bash
# Build from source
git clone https://github.com/SuperInstance/flux-compiler-workspace
cd flux-compiler-workspace
cargo build --release

# The CLI binary
./target/release/fluxc --help
```

Requires Rust ≥ 1.82.

## License

Apache-2.0
