fn main() -> std::process::ExitCode {
    match wisp_cli::executable::main_entry() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            std::process::ExitCode::FAILURE
        }
    }
}
