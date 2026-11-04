// Set HF_TOKEN through your shell or secret manager; never put it in this file.
// WISP_HF_REVISION must be a full commit hash, not a moving branch name.
repo = getenv('WISP_HF_REPO');
revision = getenv('WISP_HF_REVISION');
file = getenv('WISP_HF_FILE', 'model.safetensors');
part = getenv('WISP_HF_PART');
assert(length(repo) > 0);
assert(length(revision) == 40);
assert(length(part) > 0); // select a manageable tensor rather than expanding a large model

path = hf_download(repo, file, 'revision', revision);
net = wload(path, 'parts', part);
disp(names(net));
assert(budget() == 0);
