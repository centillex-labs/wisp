use wisp_runtime::gguf::{encode, f32_to_f16_exact, Gguf, T_Q8_0};
use wisp_runtime::Interp;

fn run(src: &str) -> Result<(), wisp_syntax::Diag> {
    let program = wisp_syntax::parse(src)?;
    let mut interp = Interp::new("gguf.wisp", src);
    interp.echo = false;
    interp.run(&program).map(|_| ())
}

fn s(out: &mut Vec<u8>, v: &str) {
    out.extend_from_slice(&(v.len() as u64).to_le_bytes());
    out.extend_from_slice(v.as_bytes());
}

/// (name, dims, type, bytes)
type T<'a> = (&'a str, Vec<u64>, u32, Vec<u8>);

fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn write(path: &std::path::Path, name: &str, tensors: &[T]) {
    let mut b = b"GGUF".to_vec();
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    b.extend_from_slice(&4u64.to_le_bytes());
    s(&mut b, "general.architecture");
    b.extend_from_slice(&8u32.to_le_bytes());
    s(&mut b, "toy");
    s(&mut b, "general.name");
    b.extend_from_slice(&8u32.to_le_bytes());
    s(&mut b, name);
    s(&mut b, "adapter.lora_alpha");
    b.extend_from_slice(&6u32.to_le_bytes());
    b.extend_from_slice(&2f32.to_le_bytes());
    s(&mut b, "tokens");
    b.extend_from_slice(&9u32.to_le_bytes());
    b.extend_from_slice(&8u32.to_le_bytes());
    b.extend_from_slice(&2u64.to_le_bytes());
    s(&mut b, "a");
    s(&mut b, "bc");
    let mut offset = 0u64;
    for (name, dims, ty, bytes) in tensors {
        s(&mut b, name);
        b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for d in dims {
            b.extend_from_slice(&d.to_le_bytes());
        }
        b.extend_from_slice(&ty.to_le_bytes());
        b.extend_from_slice(&offset.to_le_bytes());
        offset = (offset + bytes.len() as u64).div_ceil(32) * 32;
    }
    b.resize(b.len().div_ceil(32) * 32, 0);
    for (_, _, _, bytes) in tensors {
        b.extend_from_slice(bytes);
        b.resize(b.len().div_ceil(32) * 32, 0);
    }
    std::fs::write(path, b).unwrap();
}

#[test]
fn float16_conversion_rounds_to_nearest_even() {
    assert_eq!(f32_to_f16_exact(1.0), 0x3c00);
    assert_eq!(f32_to_f16_exact(-2.0), 0xc000);
    assert_eq!(f32_to_f16_exact(65520.0), 0x7c00);
    assert_eq!(f32_to_f16_exact(1.0 + 1.0 / 2048.0), 0x3c00); // tie, even stays
    assert_eq!(f32_to_f16_exact(1.0 + 3.0 / 2048.0), 0x3c02); // tie, odd rounds up
    assert_eq!(f32_to_f16_exact(2f32.powi(-24)), 0x0001);
    assert_eq!(f32_to_f16_exact(2f32.powi(-25)), 0x0000); // tie to even (zero)
    assert_eq!(f32_to_f16_exact(1.5 * 2f32.powi(-25)), 0x0001);
}

#[test]
fn q8_0_matches_the_reference_quantizer() {
    let mut x = vec![0.0f64; 32];
    x[0] = 127.0;
    x[1] = -63.5; // half rounds away from zero
    x[2] = 0.4;
    x[3] = -0.5;
    let b = encode(T_Q8_0, &x).unwrap();
    assert_eq!(&b[..2], &0x3c00u16.to_le_bytes());
    assert_eq!(&b[2..6], &[127u8, (-64i8) as u8, 0, (-1i8) as u8]);
    assert_eq!(encode(T_Q8_0, &vec![0.0; 32]).unwrap(), vec![0u8; 34]);
    assert!(encode(T_Q8_0, &vec![0.0; 31]).is_none());
}

#[test]
fn rewrite_edits_planned_tensors_and_copies_the_rest_verbatim() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("src.gguf");
    let lora = dir.path().join("lora.gguf");
    let out = dir.path().join("out.gguf");
    let opaque: Vec<u8> = (0..37u8).map(|i| i.wrapping_mul(7) | 1).collect();
    let w: Vec<f32> = (0..64).map(|i| i as f32 * 0.25 - 4.0).collect();
    write(
        &src,
        "Toy",
        &[
            ("keep.weight", vec![3], 0, f32s(&[1.5, -2.5, 3.25])),
            ("blk.0.w.weight", vec![32, 2], 0, f32s(&w)),
            ("blk.0.x.scale", vec![2], 0, f32s(&[9.0, 8.0])),
            ("blk.0.x.weight", vec![32, 2], 42, opaque.clone()),
            ("tail.weight", vec![5], 42, opaque.clone()),
        ],
    );
    let a: Vec<f32> = (0..32).map(|i| i as f32).collect();
    write(
        &lora,
        "LoRA",
        &[
            ("blk.0.w.weight.lora_a", vec![32, 1], 0, f32s(&a)),
            ("blk.0.w.weight.lora_b", vec![1, 2], 0, f32s(&[0.5, -1.0])),
        ],
    );
    let (src_s, lora_s, out_s) = (
        src.to_str().unwrap(),
        lora.to_str().unwrap(),
        out.to_str().unwrap(),
    );
    run(&format!(
        r#"
src = '{src_s}'; lora = '{lora_s}'; out = '{out_s}';
assert(isequal(gguf_names(lora), {{'blk.0.w.weight.lora_a', 'blk.0.w.weight.lora_b'}}));
assert(isequal(gguf_type(src, 'blk.0.x.weight'), 'T42'));
assert(isequal(gguf_size(src, 'blk.0.w.weight'), [2 32]));
assert(isequal(gguf_size(src, 'keep.weight'), [3 1]));
assert(gguf_meta(lora, 'adapter.lora_alpha') == 2);
assert(isequal(gguf_meta(src, 'general.name'), 'Toy'));
assert(isequal(gguf_meta(src, 'tokens'), {{'a', 'bc'}}));
assert(gguf_meta(src, 'missing', 7) == 7);
assert(isequal(gguf_read(src, 'keep.weight'), [1.5; -2.5; 3.25]));
assert(endsWith('a.lora_a', '.lora_a') && !endsWith('a.lora_b', '.lora_a'));
assert(isequal(strrep('blk.0.x.weight', '.weight', '.scale'), 'blk.0.x.scale'));
assert(single(0.1) != 0.1 && single(0.5) == 0.5);
assert(bf16(1 + 1/1024) == 1);

alpha = gguf_meta(lora, 'adapter.lora_alpha');
edit = @(name) single(gguf_read(src, name) + single(alpha * (gguf_read(lora, strcat(name, '.lora_b')) .* gguf_read(lora, strcat(name, '.lora_a')))));
r = gguf_rewrite(src, out, {{'blk.0.w.weight', 'blk.0.x.weight'}}, {{'Q8_0', 'F32'}}, ...
    @(name) wrap(name, edit), 'drop', {{'blk.0.x.scale'}}, 'meta', {{'general.name', 'Toy-Edited', 'splice.src', 'src.gguf'}});
assert(r.edited == 2 && r.dropped == 1 && r.copied == 2);
assert(length(r.sha256) == 64);

assert(isequal(gguf_names(out), {{'keep.weight', 'blk.0.w.weight', 'blk.0.x.weight', 'tail.weight'}}));
assert(isequal(gguf_meta(out, 'general.name'), 'Toy-Edited'));
assert(isequal(gguf_meta(out, 'splice.src'), 'src.gguf'));
assert(isequal(gguf_meta(out, 'tokens'), {{'a', 'bc'}}));
assert(isequal(gguf_type(out, 'blk.0.w.weight'), 'Q8_0'));
assert(isequal(gguf_read(out, 'blk.0.x.weight'), ones(2, 32)));
got = gguf_read(out, 'blk.0.w.weight');
exact = edit('blk.0.w.weight');
assert(max(abs(got[:] - exact[:])) <= max(abs(exact[:])) / 127);

function x = wrap(name, edit)
    if isequal(name, 'blk.0.x.weight')
        x = ones(2, 32);
    else
        x = edit(name);
    end
end
"#
    ))
    .unwrap();

    let o = Gguf::open(out_s).unwrap();
    let bytes = std::fs::read(&out).unwrap();
    let raw = |n: &str| {
        let t = o.tensor(n).unwrap();
        let at = (o.data_start + t.offset) as usize;
        bytes[at..at + t.nbytes as usize].to_vec()
    };
    assert_eq!(raw("keep.weight"), f32s(&[1.5, -2.5, 3.25]));
    assert_eq!(&raw("tail.weight")[..37], &opaque[..]);
    assert_eq!(bytes.len() % 32, 0);
    assert!(!dir.path().join("out.gguf.partial").exists());

    // Errors leave no output behind.
    let bad = dir.path().join("bad.gguf");
    let bad_s = bad.to_str().unwrap();
    let e = run(&format!(
        "gguf_rewrite('{src_s}', '{bad_s}', {{'keep.weight'}}, 'F32', @(n) zeros(4, 1));"
    ))
    .unwrap_err();
    assert!(format!("{e:?}").contains("returned 4 values"));
    assert!(!bad.exists() && !dir.path().join("bad.gguf.partial").exists());
    let e = run(&format!("gguf_read('{src_s}', 'tail.weight');")).unwrap_err();
    assert!(format!("{e:?}").contains("copy but not decode"));
    let e = run(&format!(
        "gguf_rewrite('{src_s}', '{src_s}', {{'keep.weight'}}, 'F32', @(n) zeros(3, 1));"
    ))
    .unwrap_err();
    assert!(format!("{e:?}").contains("must not be the source"));
}
