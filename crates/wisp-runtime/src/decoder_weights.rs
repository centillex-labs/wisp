//! Bounded, validated tensor reads without mapping a mutable checkpoint into memory.
use candle_core::{DType, Device, Shape, Tensor};
use candle_nn::var_builder::SimpleBackend;
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
    sync::Mutex,
};

pub(crate) struct Weights {
    file: Mutex<File>,
    metadata: safetensors::tensor::Metadata,
    data_start: u64,
}
impl Weights {
    pub fn open(path: &Path) -> Result<Self, String> {
        let mut file = File::open(path).map_err(|e| e.to_string())?;
        let (metadata, data_start) =
            crate::io::safetensors_header(&mut file, &path.display().to_string())?;
        Ok(Self {
            file: Mutex::new(file),
            metadata,
            data_start,
        })
    }
    pub fn dtype(&self, name: &str) -> Result<crate::net::Dtype, String> {
        use crate::net::Dtype;
        match self.metadata.info(name).map(|i| i.dtype) {
            Some(safetensors::Dtype::F16) => Ok(Dtype::F16),
            Some(safetensors::Dtype::BF16) => Ok(Dtype::BF16),
            Some(safetensors::Dtype::F32) => Ok(Dtype::F32),
            _ => Err(format!(
                "native Qwen3 requires F16, BF16, or F32 weights: {name}"
            )),
        }
    }
}
impl SimpleBackend for Weights {
    fn get(
        &self,
        shape: Shape,
        name: &str,
        _: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        let info = self
            .metadata
            .info(name)
            .ok_or_else(|| candle_core::Error::Msg(format!("missing tensor {name}")))?;
        if info.shape != shape.dims() {
            candle_core::bail!(
                "invalid shape for {name}: {:?}, expected {:?}",
                info.shape,
                shape.dims()
            );
        }
        self.get_unchecked(name, dtype, dev)
    }
    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        let info = self
            .metadata
            .info(name)
            .ok_or_else(|| candle_core::Error::Msg(format!("missing tensor {name}")))?;
        let stored = match self.dtype(name).map_err(candle_core::Error::Msg)? {
            crate::net::Dtype::F16 => DType::F16,
            crate::net::Dtype::BF16 => DType::BF16,
            crate::net::Dtype::F32 => DType::F32,
            _ => unreachable!(),
        };
        let mut bytes = vec![0; info.data_offsets.1 - info.data_offsets.0];
        {
            let mut file = self
                .file
                .lock()
                .map_err(|_| candle_core::Error::Msg("weight reader lock poisoned".into()))?;
            file.seek(SeekFrom::Start(
                self.data_start + info.data_offsets.0 as u64,
            ))?;
            file.read_exact(&mut bytes)?;
        }
        Tensor::from_raw_buffer(&bytes, stored, &info.shape, dev)?.to_dtype(dtype)
    }
    fn contains_tensor(&self, name: &str) -> bool {
        self.metadata.info(name).is_some()
    }
}
