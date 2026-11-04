//! Pinned, authenticated Hugging Face file downloads. No Python or token persistence.
use reqwest::{
    blocking::Client,
    header::{HeaderValue, AUTHORIZATION, LOCATION},
    Url,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    time::Duration,
};

pub struct DownloadOptions {
    pub revision: String,
    pub sha256: Option<String>,
    pub offline: bool,
    pub max_bytes: u64,
}

impl Default for DownloadOptions {
    fn default() -> Self {
        Self {
            revision: String::new(),
            sha256: None,
            offline: false,
            max_bytes: 20 * 1024 * 1024 * 1024,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct Receipt {
    repo: String,
    file: String,
    revision: String,
    sha256: String,
    bytes: u64,
}

pub fn environment(name: &str, fallback: &str) -> Result<String, String> {
    if name.is_empty() || name.contains(['=', '\0']) {
        return Err("invalid environment variable name".into());
    }
    match std::env::var(name) {
        Ok(value) => Ok(value),
        Err(std::env::VarError::NotPresent) => Ok(fallback.into()),
        Err(std::env::VarError::NotUnicode(_)) => Err("environment value is not UTF-8".into()),
    }
}

fn env_flag(name: &str) -> Result<bool, String> {
    match environment(name, "")?.trim().to_ascii_lowercase().as_str() {
        "" | "0" | "false" | "no" => Ok(false),
        "1" | "true" | "yes" => Ok(true),
        _ => Err(format!("{name} must be a boolean")),
    }
}

fn cache_root() -> Result<PathBuf, String> {
    for (name, suffix) in [
        ("WISP_MODEL_CACHE", "huggingface"),
        ("HF_HUB_CACHE", "wisp"),
        ("HF_HOME", "wisp"),
        ("XDG_CACHE_HOME", "wisp/huggingface"),
    ] {
        if let Some(value) = std::env::var_os(name).filter(|v| !v.is_empty()) {
            return Ok(PathBuf::from(value).join(suffix));
        }
    }
    let home =
        std::env::var_os("HOME").ok_or("set WISP_MODEL_CACHE or HOME for the model cache")?;
    Ok(PathBuf::from(home).join(".cache/wisp/huggingface"))
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn file_digest(path: &Path) -> Result<(String, u64), String> {
    let mut file = fs::File::open(path).map_err(|_| "cannot read cached artifact")?;
    let mut hash = Sha256::new();
    let mut length = 0;
    let mut buffer = [0u8; 65536];
    loop {
        let n = file
            .read(&mut buffer)
            .map_err(|_| "cannot hash cached artifact")?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
        length += n as u64;
    }
    Ok((format!("{:x}", hash.finalize()), length))
}

fn component(s: &str) -> bool {
    !s.is_empty()
        && s != "."
        && s != ".."
        && s.bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"._-".contains(&c))
}
fn validate(repo: &str, file: &str, options: &DownloadOptions) -> Result<(), String> {
    if !(1..=2).contains(&repo.split('/').count()) || !repo.split('/').all(component) {
        return Err("HF repository must be a model name or owner/model".into());
    }
    if !file.split('/').all(component) {
        return Err("HF filename must be a relative path without traversal".into());
    }
    if options.revision.len() != 40 || !options.revision.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err("HF revision must be a pinned 40-character commit hash".into());
    }
    if options
        .sha256
        .as_ref()
        .is_some_and(|s| s.len() != 64 || !s.bytes().all(|c| c.is_ascii_hexdigit()))
    {
        return Err("expected SHA-256 must contain 64 hex characters".into());
    }
    if options.max_bytes == 0 {
        return Err("download size limit must be positive".into());
    }
    Ok(())
}

fn private_directory(path: &Path) -> Result<(), String> {
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder
        .create(path)
        .map_err(|_| "cannot create private HF cache directory")?;
    let metadata = fs::symlink_metadata(path).map_err(|_| "cannot inspect HF cache directory")?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err("HF cache must be a directory, not a symlink".into());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if metadata.permissions().mode() & 0o077 != 0 {
            return Err("HF cache directory must have private permissions (0700)".into());
        }
    }
    Ok(())
}

/// Credentials are read by the host, not passed as Wisp arguments or cached.
pub fn download(repo: &str, file: &str, options: &DownloadOptions) -> Result<PathBuf, String> {
    validate(repo, file, options)?;
    let token = if env_flag("HF_HUB_DISABLE_IMPLICIT_TOKEN")? {
        String::new()
    } else {
        environment("HF_TOKEN", "")?
    };
    fetch(
        repo,
        file,
        options,
        &cache_root()?,
        (!token.is_empty()).then_some(token.as_str()),
        options.offline || env_flag("HF_HUB_OFFLINE")?,
        "https://huggingface.co/",
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn fetch(
    repo: &str,
    file: &str,
    options: &DownloadOptions,
    root: &Path,
    token: Option<&str>,
    offline: bool,
    endpoint: &str,
    allow_http: bool,
) -> Result<PathBuf, String> {
    validate(repo, file, options)?;
    private_directory(root)?;
    let mut directory = root.to_path_buf();
    for segment in [
        digest(repo.as_bytes()),
        options.revision.to_ascii_lowercase(),
        digest(file.as_bytes()),
    ] {
        directory.push(segment);
        private_directory(&directory)?;
    }
    let extension = Path::new(file)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("bin");
    let artifact = directory.join(format!("artifact.{extension}"));
    let receipt_path = directory.join("receipt.json");
    let mut lock_options = fs::OpenOptions::new();
    lock_options
        .create(true)
        .truncate(false)
        .read(true)
        .write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        lock_options.mode(0o600);
    }
    let lock = lock_options
        .open(directory.join("download.lock"))
        .map_err(|_| "cannot lock HF cache")?;
    lock.lock().map_err(|_| "cannot lock HF cache")?;
    if receipt_path.exists() {
        let receipt_info =
            fs::symlink_metadata(&receipt_path).map_err(|_| "cannot inspect HF receipt")?;
        let artifact_info =
            fs::symlink_metadata(&artifact).map_err(|_| "cannot inspect cached artifact")?;
        if !receipt_info.is_file()
            || receipt_info.len() > 65536
            || !artifact_info.is_file()
            || artifact_info.len() > options.max_bytes
        {
            return Err("HF cache integrity check failed: invalid file type or size".into());
        }
        let metadata: Receipt = serde_json::from_reader(
            fs::File::open(&receipt_path).map_err(|_| "cannot read HF receipt")?,
        )
        .map_err(|_| "invalid HF cache receipt; remove this cache entry before retrying")?;
        let (hash, length) = file_digest(&artifact)?;
        if metadata.repo != repo
            || metadata.file != file
            || !metadata.revision.eq_ignore_ascii_case(&options.revision)
            || metadata.sha256 != hash
            || metadata.bytes != length
            || length > options.max_bytes
            || options
                .sha256
                .as_ref()
                .is_some_and(|s| !s.eq_ignore_ascii_case(&hash))
        {
            return Err(
                "HF cache integrity check failed; remove this cache entry before retrying".into(),
            );
        }
        return Ok(artifact);
    }
    if offline {
        return Err("HF offline mode: no verified cached artifact".into());
    }
    let mut builder = Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(Duration::from_secs(15))
        .timeout(Duration::from_secs(3600));
    if allow_http {
        builder = builder.no_proxy();
    } // isolated loopback tests only
    let client = builder
        .build()
        .map_err(|_| "cannot initialize HF HTTPS client")?;
    let mut url = Url::parse(endpoint).map_err(|_| "invalid HF endpoint")?;
    {
        let mut path = url.path_segments_mut().map_err(|_| "invalid HF endpoint")?;
        path.pop_if_empty();
        for segment in repo.split('/') {
            path.push(segment);
        }
        path.push("resolve")
            .push(&options.revision.to_ascii_lowercase());
        for segment in file.split('/') {
            path.push(segment);
        }
    }
    let mut authorization = match token {
        Some(token) => {
            let mut header = HeaderValue::from_str(&format!("Bearer {token}"))
                .map_err(|_| "HF_TOKEN contains invalid header characters")?;
            header.set_sensitive(true);
            Some(header)
        }
        None => None,
    };
    let original_origin = url.origin();
    let mut response = None;
    for _ in 0..6 {
        if (url.scheme() != "https" && !(allow_http && url.scheme() == "http"))
            || !url.username().is_empty()
            || url.password().is_some()
        {
            return Err("HF redirects must use HTTPS without URL credentials".into());
        }
        let mut request = client.get(url.clone());
        if let Some(header) = &authorization {
            request = request.header(AUTHORIZATION, header.clone());
        }
        let result = request
            .send()
            .map_err(|_| "HF request failed (connection, TLS, or timeout)")?;
        if result.status().is_redirection() {
            let location = result
                .headers()
                .get(LOCATION)
                .and_then(|v| v.to_str().ok())
                .ok_or("HF redirect has no valid Location")?;
            url = url.join(location).map_err(|_| "invalid HF redirect URL")?;
            // Authenticate same-origin HF routes, but never leak credentials to a CDN.
            if url.origin() != original_origin {
                authorization = None;
            }
            continue;
        }
        if result.status() != reqwest::StatusCode::OK {
            return Err(match result.status().as_u16() {
                401 | 403 => "HF access denied: check HF_TOKEN, repository permissions, and gated-model approval".into(),
                404 => "HF artifact not found or not accessible with these credentials".into(),
                code => format!("HF server returned HTTP {code}"),
            });
        }
        response = Some(result);
        break;
    }
    let mut response = response.ok_or("HF redirect limit exceeded")?;
    let expected_length = response.content_length();
    if expected_length.is_some_and(|n| n > options.max_bytes) {
        return Err("HF artifact exceeds the download size limit".into());
    }
    let mut temporary = tempfile::NamedTempFile::new_in(&directory)
        .map_err(|_| "cannot create temporary download")?;
    let mut hash = Sha256::new();
    let mut length = 0u64;
    let mut buffer = [0u8; 65536];
    loop {
        let n = response
            .read(&mut buffer)
            .map_err(|_| "HF download interrupted; incomplete artifact was not published")?;
        if n == 0 {
            break;
        }
        length = length
            .checked_add(n as u64)
            .ok_or("download length overflow")?;
        if length > options.max_bytes {
            return Err("HF artifact exceeds the download size limit".into());
        }
        hash.update(&buffer[..n]);
        temporary
            .write_all(&buffer[..n])
            .map_err(|_| "cannot write HF cache")?;
    }
    let hash = format!("{:x}", hash.finalize());
    if expected_length.is_some_and(|n| n != length)
        || options
            .sha256
            .as_ref()
            .is_some_and(|s| !s.eq_ignore_ascii_case(&hash))
    {
        return Err("HF download integrity check failed; artifact was not published".into());
    }
    temporary
        .as_file()
        .sync_all()
        .map_err(|_| "cannot sync HF cache")?;
    temporary
        .persist(&artifact)
        .map_err(|_| "cannot publish HF artifact")?;
    let receipt = Receipt {
        repo: repo.into(),
        file: file.into(),
        revision: options.revision.clone(),
        sha256: hash,
        bytes: length,
    };
    let mut temporary =
        tempfile::NamedTempFile::new_in(&directory).map_err(|_| "cannot create HF receipt")?;
    serde_json::to_writer(&mut temporary, &receipt).map_err(|_| "cannot write HF receipt")?;
    temporary
        .as_file()
        .sync_all()
        .map_err(|_| "cannot sync HF receipt")?;
    temporary
        .persist(receipt_path)
        .map_err(|_| "cannot publish HF receipt")?;
    Ok(artifact)
}

#[cfg(test)]
#[path = "hf_tests.rs"]
mod tests;
