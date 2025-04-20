use std::path::Path;
use std::process::Command;

/// Links the generated object file using an external linker (e.g., clang).
/// Assumes a 'main' function compatible with C linking exists in the object file.
/// NOTE: Our current 'main' returns f64, which isn't the standard C main signature (int()).
/// Linking might succeed, but running it might crash or behave unexpectedly
/// without a proper C entry point or runtime setup.
/// For now, we just demonstrate the linking command.
pub fn link_object_file(obj_path: &Path, executable_name: &str) {
    println!("\n--- Linking ---");
    let linker = "clang"; // Using clang++ might also work and link C++ automatically
    let target_dir = std::env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| "target".to_string());
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    let lib_name = "libtoylang_compiler.a";
    let lib_path = Path::new(&target_dir).join(profile).join(lib_name);

    if !lib_path.exists() {
        eprintln!(
            "Linking Error: Static library not found at {}",
            lib_path.display()
        );
        eprintln!("Ensure you have run 'cargo build' first.");
        return;
    }

    println!(
        "Attempting to link {} with library {} using {}",
        obj_path.display(),
        lib_path.display(),
        linker
    );

    let zstd_prefix = "/opt/homebrew/opt/zstd"; // COMMON FOR APPLE SILICON
    // Use "/usr/local/opt/zstd" for Intel Macs if applicable
    let zstd_lib_path = format!("{}/lib", zstd_prefix);

    // --- Add necessary system libraries ---
    let status = Command::new(linker)
        .arg(obj_path) // ToyLang object file
        .arg(lib_path) // Static library with wrappers + LLVM bits
        .arg(format!("-L{}", zstd_lib_path)) // Add -L/opt/homebrew/opt/zstd/lib (or similar)
        .arg("-lc++") // C++ Standard Library
        .arg("-lz") // Zlib
        .arg("-lzstd") // Zstandard (Added)
        .arg("-lncurses") // Ncurses/Terminfo (Added)
        .arg("-lffi") // LibFFI (Added)
        .arg("-o")
        .arg(executable_name)
        .status(); // Execute the command
    match status {
        Ok(exit_status) if exit_status.success() => {
            println!("Successfully linked executable: {}", executable_name);
            println!("You should now be able to run './{}'", executable_name);
        }
        Ok(exit_status) => {
            eprintln!("Linking failed with status: {}", exit_status);
            eprintln!("Check if required libraries (libc++, zlib, potentially zstd) are installed and accessible.");
        }
        Err(e) => {
            eprintln!("Failed to execute linker '{}': {}", linker, e);
        }
    }
}
