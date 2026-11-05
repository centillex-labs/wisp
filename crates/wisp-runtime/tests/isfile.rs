use wisp_runtime::Interp;

fn run(src: &str) -> String {
    let program = wisp_syntax::parse(src).unwrap();
    let mut it = Interp::new("isfile.wisp", src);
    it.run(&program).unwrap();
    it.out
}

#[test]
fn isfile_reports_regular_files_directories_and_missing_paths() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("data.safetensors");
    std::fs::write(&file, b"wisp").unwrap();
    let nested = dir.path().join("nested");
    std::fs::create_dir(&nested).unwrap();

    let out = run(&format!(
        "disp(isfile('{}')); disp(isfile('{}')); disp(isfile('{}')); disp(isfile(''));",
        file.display(),
        nested.display(),
        dir.path().join("absent.safetensors").display()
    ));
    assert_eq!(out, "1\n0\n0\n0\n");
}

#[test]
fn isfile_requires_one_string_argument() {
    for src in ["isfile()", "isfile('a', 'b')", "isfile(1)"] {
        let error = wisp_syntax::parse(src)
            .and_then(|p| {
                let mut it = Interp::new("isfile.wisp", src);
                it.run(&p).map(|_| ())
            })
            .unwrap_err();
        assert!(matches!(error.code, "E-ARGS" | "E-TYPE"), "{src}");
    }
}

#[test]
fn isfile_is_listed_as_a_builtin() {
    assert!(wisp_runtime::builtins::exists("isfile"));
}
