# Phoenix Language

<p align="center">
  🔥 A modern, statically-typed programming language with LLVM-powered compilation
</p>

## Overview

Phoenix is a programming language designed for clarity, performance, and ease of use. It compiles directly to optimized machine code using LLVM, combining the speed of low-level languages with a clean, modern syntax.

## Example Code

```
fun fibonacci(n: int) : int {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}

fun main() {
    println("Hello from Phoenix!");

    // Vector operations
    var numbers: vec<int> = [1, 2, 3];
    push(numbers, 4);
    numbers[0] = 10;

    println("The 10th Fibonacci number is:");
    println(fibonacci(10));

    // Control flow
    for (var i = 0; i < 5; i++) {
        if (i > 2) {
            println("Bigger than 2");
        } else {
            println("Smaller than or equal to 2");
        }
    }
}

main();
```

## Key Features

- **Strong Static Typing**: Catch errors at compile time
- **First-class Functions**: Functions as values
- **Vector Support**: Built-in arrays with dynamic operations
- **Rich Control Flow**: If/else, while loops, for loops
- **Multiple Data Types**: Integer, float, boolean, string
- **LLVM Optimization**: Compiled to efficient machine code
- **Precise Error Reporting**: Error messages with file and line information

## Getting Started

```bash
# Build the compiler
cargo build -p phoenix_runtime -r; 
cargo build -r

# Compile a Phoenix file
./target/release/compiler --input program.px

# Run the compiled program
./output/output
```
(Currently only tested on mac)

## Project Status

Phoenix is under active development. Recent additions include:
- Vector support with push and indexing
- Span-based error reporting
- Return statements
- Multiple print function variants

## Roadmap

- [x] String as a first-class type
- [ ] Multi-file support
- [ ] Structs and classes
- [ ] Logical operators (`&&` and `||`)
  - a && b: If a evaluates to false, b is not evaluated, and the result is false.
  - a || b: If a evaluates to true, b is not evaluated, and the result is true.
- [ ] Functions as variables
- [ ] Package management
- [ ] REPL

---

Built with ❤️ and Rust