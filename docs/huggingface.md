# Environment variables and Hugging Face

Native Rust implementation; no Python, shell downloader, or HF CLI required.
For loading and running Bonsai directly, see [`hf_model` and the strawberry program](strawberry.md).

```wisp
repo = getenv('WISP_HF_REPO');
revision = getenv('WISP_HF_REVISION');
file = getenv('WISP_HF_FILE', 'model.safetensors');

path = hf_download(repo, file, 'revision', revision);
net = wload(path, 'parts', 'model.layers.0.mlp.down_proj.weight');
```

Set `HF_TOKEN` through your shell or secret manager for private/gated repositories.
The token needs repository read access; gated models also require account approval.
Do not put the token in Wisp source or command-line arguments.

## `getenv(name [, default])`

Returns the inherited environment value as a string. An unset variable returns
`default`, or `''` when omitted. An explicitly empty value remains empty.
Invalid names and non-UTF-8 values are errors. Wisp does not automatically read
`.env` files or modify the process environment.

## `hf_download(repo, file, ...)`

Downloads one model-repository file and returns its local cache path.

| Option | Meaning |
| --- | --- |
| `'revision', hash` | Required full 40-character commit hash; branches such as `main` are refused. |
| `'sha256', hash` | Optional expected 64-character SHA-256, checked on downloads and cache hits. |
| `'offline', true` | Require an existing verified cache entry; never contact HF. |
| `'max_bytes', n` | Positive integer size limit; default 20 GiB. |

Both new downloads and cache hits are size-checked. Every completed artifact has a
local SHA-256 receipt; supply `sha256` to additionally verify against a known digest.
A corrupt cache entry is refused, not silently used. Downloads stream to temporary
files; a receipt commits successful completion. Concurrent requests share a file
lock. Incomplete downloads cannot become offline cache hits.

## Environment controls

| Variable | Purpose |
| --- | --- |
| `HF_TOKEN` | Bearer credential, read internally by the downloader. |
| `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` | Do not send `HF_TOKEN`. |
| `HF_HUB_OFFLINE=1` | Forbid network access even if the program sets `offline` to false. |
| `WISP_MODEL_CACHE` | Cache base; Wisp uses its `huggingface/` subdirectory. |
| `HF_HUB_CACHE` | Fallback cache base; Wisp uses its `wisp/` subdirectory. |
| `HF_HOME` | Next fallback; Wisp uses its `wisp/` subdirectory. |
| `XDG_CACHE_HOME` | Next fallback; Wisp uses `wisp/huggingface/`. |

Otherwise the cache is `$HOME/.cache/wisp/huggingface`. Wisp maintains its own cache
format; it does not import or modify the Hugging Face client's blob/snapshot cache.
`HF_ENDPOINT` is not supported: authenticated requests start at `https://huggingface.co`.

The downloader requires HTTPS, strips credentials when redirects leave the original
origin, and does not echo server error bodies or signed URLs. Tokens are not written
to download receipts or Wisp observation logs. On Unix, cache directories must be
private (0700); downloaded files are created with 0600 permissions.

**Only run trusted Wisp programs with credentials in their environment.** `getenv`
returns ordinary strings: a program can explicitly print them. This is not a
secret-taint system or a sandbox against malicious scripts.

## Example

Provide `HF_TOKEN` securely, then configure non-secret model identifiers:

```bash
export WISP_HF_REPO='your-organization/your-model'
export WISP_HF_REVISION='<full-40-character-commit-hash>'
export WISP_HF_FILE='model.safetensors'
export WISP_HF_PART='model.layers.0.mlp.down_proj.weight'
cargo build --release
./target/release/wisp examples/huggingface.wisp --budget 0
```

Use the actual tensor name and, for sharded checkpoints, the shard containing it.
Downloads and weight reads use no model-observation passes; they still consume
network, disk, and memory resources. `hf_download` retrieves one file at a time;
select the required shard explicitly. Selective `wload` decodes the chosen tensors
in memory, so choose parts that fit available RAM. Architecture-specific inference
uses an external host; see [Bonsai behavioral repair](bonsai-behavior-repair.md).

Tests use isolated loopback servers and fake credentials, never private tokens.
