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
    let linker = "clang";
    // Determine the path to the static library produced by cargo build
    // Usually in target/debug/ or target/release/
    // Use CARGO_TARGET_DIR if set, otherwise assume ./target/debug
    let target_dir = std::env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| "target".to_string());
    // Adjust debug/release based on your build profile
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    let lib_name = "libphoenix_runtime.a"; // Matches package name by default
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

    let status = Command::new(linker)
        .arg(obj_path) // Input object file from phoenix code
        .arg(lib_path) // Input static library containing the wrapper
        .arg("-o")
        .arg(executable_name)
        .status();

    match status {
        Ok(exit_status) if exit_status.success() => {
            println!("Successfully linked executable: {}", executable_name);
            println!("You should now be able to run './{}'", executable_name);
        }
        Ok(exit_status) => {
            eprintln!("Linking failed with status: {}", exit_status);
        }
        Err(e) => {
            eprintln!("Failed to execute linker '{}': {}", linker, e);
        }
    }
}