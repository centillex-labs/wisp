# GGUF files

Wisp reads GGUF containers and rewrites them one tensor at a time. A rewrite copies every
tensor it was not asked to change byte for byte, so a file can hold tensor types Wisp does not
understand (K-quants, NVFP4 and so on) and still be edited.

```wisp
names = gguf_names('model.gguf');
w = gguf_read('model.gguf', 'blk.0.attn_output.weight');

r = gguf_rewrite('model.gguf', 'edited.gguf', {'blk.0.attn_output.weight'}, 'Q8_0', ...
                 @(name) single(gguf_read('model.gguf', name) * 0.5), ...
                 'meta', {'general.name', 'Edited'});
fprintf('%s\n', r.sha256);
```

None of these functions use observation passes. They do not load a net: `wload`, `wsave`,
patches and the changed-weight budget do not apply to GGUF files.

## Reading

| Call | Returns |
| --- | --- |
| `gguf_names(file)` | Cell of tensor names, in file order. |
| `gguf_type(file, name)` | Storage type as a string: `'F32'`, `'BF16'`, `'Q8_0'`, `'Q4_K'`, ... Types Wisp has no name for print as `'T<number>'`, for example `'T42'`. |
| `gguf_size(file, name)` | Shape as a row vector, outermost dimension first. |
| `gguf_meta(file, key [, default])` | A metadata value: number, logical, string, numeric row, or cell of strings. A missing key is an error unless `default` is given. |
| `gguf_read(file, name)` | The tensor as a float64 array. |

`gguf_read` decodes F32, F64, F16, BF16, Q8_0 and the integer types. Any other type fails with
`E-IO`: Wisp can copy it but not decode it.

Shapes are row-major with the outermost dimension first, the reverse of the order stored in the
file. This matches numpy's view of the same tensor. A one-dimensional tensor is an n-by-1 column.

Every call parses the file header again. That is cheap next to reading a tensor.

## Rewriting

```wisp
r = gguf_rewrite(src, out, names, types, edit, 'drop', {...}, 'meta', {key, value, ...});
```

| Argument | Meaning |
| --- | --- |
| `src`, `out` | Source and output paths. They must differ. |
| `names` | Cell of tensor names to replace. |
| `types` | One storage type for all of them, or a cell with one type per name. Wisp writes `F32`, `F64`, `F16`, `BF16` and `Q8_0`. |
| `edit` | Function of the tensor name that returns the new values. |
| `'drop'` | Cell of tensor names to leave out of the output. |
| `'meta'` | Cell of key, value string pairs. An existing string key is replaced in place; a new key is appended. |

Wisp walks the tensors in file order. For each planned tensor it calls `edit(name)`, encodes the
result and writes it; only one edited tensor is in memory at a time. Everything else is copied
from `src` without decoding. Shapes, tensor order, the remaining metadata and the alignment are
kept. The result must have the tensor's shape and only finite values.

The output is written to `out.partial` and renamed when complete. A failure removes the partial
file and leaves any existing `out` untouched.

`r` is a struct with `edited`, `dropped`, `copied`, `bytes` and `sha256` (of the whole output).

### Q8_0

Encoding follows llama.cpp's reference quantizer in float32: for each block of 32 values, the
scale is `max|x| / 127` stored as float16, and values round half away from zero. The tensor's
row length must be a multiple of 32.

## Matching float32 tools

Wisp arrays are float64. To reproduce a float32 pipeline bit for bit, round after each
operation:

```wisp
x = bf16(single(w + single(alpha * d)));
```

`single(x)` rounds to float32 and `bf16(x)` to bfloat16. One float64 operation followed by
`single` gives the same result as the float32 operation.

## Limits

- Little-endian GGUF versions 2 and 3 only.
- Metadata edits are limited to string values. `general.alignment` cannot be changed.
- For types Wisp cannot decode, a tensor's bytes are taken to run up to the next tensor, so its
  alignment padding is copied with it.
- Arrays cost 8 bytes per weight and expressions make temporary copies. A 340-million-weight
  tensor needs several times its 2.7 GB while it is being edited.

## Example: Inkling-2-Mini-Cybersecurity

[inkling_cybersecurity.wisp](../examples/inkling_cybersecurity.wisp) merges a rank-1 LoRA into
49 tensors of a 17.6 GB file in Unsloth's dynamic layout and copies the other 675, including
40 NVFP4 tensors. It needs the three input files named at the top of the script and about
150 GB of disk.
