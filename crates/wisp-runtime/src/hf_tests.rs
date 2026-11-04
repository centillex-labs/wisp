use super::*;
use std::{net::TcpListener, thread, time::Instant};

const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";
const TOKEN: &str = "hf_test_secret_not_a_real_credential";
fn options(bytes: &[u8]) -> DownloadOptions {
    DownloadOptions {
        revision: REVISION.into(),
        sha256: Some(digest(bytes)),
        ..DownloadOptions::default()
    }
}
fn server(responses: Vec<String>) -> (String, thread::JoinHandle<Vec<String>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let endpoint = format!("http://{}/", listener.local_addr().unwrap());
    let worker = thread::spawn(move || {
        let mut requests = Vec::new();
        for response in responses {
            let start = Instant::now();
            let (mut stream, _) = loop {
                match listener.accept() {
                    Ok(pair) => break pair,
                    Err(e)
                        if e.kind() == std::io::ErrorKind::WouldBlock
                            && start.elapsed() < Duration::from_secs(10) =>
                    {
                        thread::sleep(Duration::from_millis(5))
                    }
                    Err(e) => panic!("mock accept: {e}"),
                }
            };
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut bytes = Vec::new();
            while !bytes.ends_with(b"\r\n\r\n") {
                let mut byte = [0];
                stream.read_exact(&mut byte).unwrap();
                bytes.push(byte[0]);
                assert!(bytes.len() < 16384);
            }
            requests.push(String::from_utf8(bytes).unwrap());
            stream.write_all(response.as_bytes()).unwrap();
        }
        requests
    });
    (endpoint, worker)
}
fn ok(body: &str) -> String {
    format!(
        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
}
fn call(
    root: &Path,
    endpoint: &str,
    opts: &DownloadOptions,
    offline: bool,
) -> Result<PathBuf, String> {
    fetch(
        "owner/private-model",
        "weights/model.safetensors",
        opts,
        &root.join("cache"),
        Some(TOKEN),
        offline,
        endpoint,
        true,
    )
}

#[test]
fn authenticated_download_is_pinned_private_cached_and_token_free() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("cache");
    let (endpoint, worker) = server(vec![ok("weights")]);
    let opts = options(b"weights");
    let path = call(&root, &endpoint, &opts, false).unwrap();
    let requests = worker.join().unwrap();
    assert!(requests[0].contains(&format!(
        "/owner/private-model/resolve/{REVISION}/weights/model.safetensors"
    )));
    assert!(requests[0]
        .to_lowercase()
        .contains(&format!("authorization: bearer {TOKEN}")));
    assert_eq!(fs::read(&path).unwrap(), b"weights");
    let receipt = fs::read_to_string(path.parent().unwrap().join("receipt.json")).unwrap();
    assert!(!receipt.contains(TOKEN));
    assert_eq!(call(&root, "not-a-url", &opts, true).unwrap(), path);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(fs::metadata(&path).unwrap().permissions().mode() & 0o077, 0);
        assert_eq!(fs::metadata(&root).unwrap().permissions().mode() & 0o077, 0);
    }
    fs::write(path, b"tampered").unwrap();
    assert!(call(&root, "not-a-url", &opts, true)
        .unwrap_err()
        .contains("integrity"));
}

#[test]
fn credentials_are_stripped_on_cross_origin_redirects() {
    let dir = tempfile::tempdir().unwrap();
    let (cdn, cdn_worker) = server(vec![ok("weights")]);
    let (endpoint, worker) = server(vec![format!("HTTP/1.1 302 Found\r\nLocation: {cdn}signed-file?signature=test\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")]);
    call(dir.path(), &endpoint, &options(b"weights"), false).unwrap();
    assert!(worker.join().unwrap()[0].contains(TOKEN));
    let request = &cdn_worker.join().unwrap()[0];
    assert!(!request.to_lowercase().contains("authorization:"));
    assert!(!request.contains(TOKEN));
}

#[test]
fn same_origin_private_redirects_remain_authenticated() {
    let dir = tempfile::tempdir().unwrap();
    let (endpoint, worker) = server(vec![
        "HTTP/1.1 307 Temporary Redirect\r\nLocation: /private-resolved\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".into(), ok("weights")]);
    call(dir.path(), &endpoint, &options(b"weights"), false).unwrap();
    for request in worker.join().unwrap() {
        assert!(request.contains(TOKEN));
    }
}

#[test]
fn http_errors_do_not_echo_response_bodies_or_tokens() {
    let dir = tempfile::tempdir().unwrap();
    let response = format!(
        "HTTP/1.1 401 Unauthorized\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{TOKEN}",
        TOKEN.len()
    );
    let (endpoint, worker) = server(vec![response]);
    let error = call(dir.path(), &endpoint, &options(b"weights"), false).unwrap_err();
    worker.join().unwrap();
    assert!(error.contains("access denied"));
    assert!(!error.contains(TOKEN));
}

#[test]
fn checksum_size_and_truncation_fail_without_publishing_receipts() {
    for (body, opts) in [
        (ok("wrong"), options(b"weights")),
        (
            ok("weights"),
            DownloadOptions {
                max_bytes: 2,
                ..options(b"weights")
            },
        ),
        (
            "HTTP/1.1 200 OK\r\nContent-Length: 100\r\nConnection: close\r\n\r\nshort".into(),
            options(b"weights"),
        ),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let (endpoint, worker) = server(vec![body]);
        assert!(call(dir.path(), &endpoint, &opts, false).is_err());
        worker.join().unwrap();
        // An interrupted or invalid online attempt cannot become an offline cache hit.
        assert!(call(dir.path(), "not-a-url", &options(b"weights"), true)
            .unwrap_err()
            .contains("no verified"));
    }
}

#[test]
fn malformed_identifiers_revisions_and_checksums_are_rejected_before_io() {
    let opts = options(b"weights");
    for (repo, file) in [
        ("../private", "x"),
        ("owner/private", "../x"),
        ("owner/private", "/x"),
        ("owner/private", "folder/../x"),
        ("owner/private", "x?token=secret"),
        ("owner/private", "x\\y"),
    ] {
        assert!(validate(repo, file, &opts).is_err());
    }
    assert!(validate(
        "owner/private",
        "x",
        &DownloadOptions {
            revision: "main".into(),
            ..options(b"x")
        }
    )
    .is_err());
    assert!(validate(
        "owner/private",
        "x",
        &DownloadOptions {
            sha256: Some("bad".into()),
            ..options(b"x")
        }
    )
    .is_err());
    assert!(environment("", "fallback").is_err());
    assert!(environment("A=B", "fallback").is_err());
    assert!(environment("A\0B", "fallback").is_err());
}

#[test]
fn concurrent_downloads_share_a_locked_verified_artifact() {
    let dir = tempfile::tempdir().unwrap();
    let (endpoint, worker) = server(vec![ok("weights")]);
    thread::scope(|scope| {
        let a = scope.spawn(|| call(dir.path(), &endpoint, &options(b"weights"), false).unwrap());
        let b = scope.spawn(|| call(dir.path(), &endpoint, &options(b"weights"), false).unwrap());
        assert_eq!(a.join().unwrap(), b.join().unwrap());
    });
    assert_eq!(worker.join().unwrap().len(), 1);
}

#[cfg(unix)]
#[test]
fn cached_symlinks_are_rejected_without_reading_their_targets() {
    let dir = tempfile::tempdir().unwrap();
    let (endpoint, worker) = server(vec![ok("weights")]);
    let path = call(dir.path(), &endpoint, &options(b"weights"), false).unwrap();
    worker.join().unwrap();
    fs::remove_file(&path).unwrap();
    std::os::unix::fs::symlink("/dev/zero", &path).unwrap();
    assert!(call(dir.path(), "not-a-url", &options(b"weights"), true)
        .unwrap_err()
        .contains("invalid file type"));
}

#[test]
fn production_transport_refuses_http_before_sending_credentials() {
    let dir = tempfile::tempdir().unwrap();
    let error = fetch(
        "owner/private",
        "model.safetensors",
        &options(b"weights"),
        &dir.path().join("cache"),
        Some(TOKEN),
        false,
        "http://127.0.0.1:1/",
        false,
    )
    .unwrap_err();
    assert!(error.contains("HTTPS"));
    assert!(!error.contains(TOKEN));
}
