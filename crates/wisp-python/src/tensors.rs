//! Bounded pure operations. Never dispatch through the general Wisp builtins.
use wisp_runtime::tensor::{Tag, Tensor};

pub(crate) const FUNCTIONS: &[&str] = &[
    "wisp_tensor",
    "wisp_zeros",
    "wisp_transpose",
    "wisp_matmul",
    "wisp_add",
    "wisp_shape",
    "wisp_values",
];
pub(crate) const MAX_ELEMENTS: usize = 65_536;
const MAX_STORED_ELEMENTS: usize = 262_144;
const MAX_TENSORS: usize = 256;
const MAX_WORK: usize = 10_000_000;
const MAX_CALL_WORK: usize = 1_000_000;

pub(crate) enum Argument {
    Integer(usize),
    Data(Vec<f64>),
}
pub(crate) enum Output {
    Handle(usize),
    Shape(Vec<usize>),
    Data(Vec<f64>),
}
#[derive(Default)]
pub(crate) struct Tensors {
    values: Vec<Tensor>,
    elements: usize,
    work: usize,
}
impl Tensors {
    fn reserve(&mut self, rows: usize, cols: usize, work: usize) -> Result<(), String> {
        let n = rows.checked_mul(cols).ok_or("tensor shape overflow")?;
        if rows > MAX_ELEMENTS || cols > MAX_ELEMENTS || n > MAX_ELEMENTS {
            return Err("tensor size limit exceeded".into());
        }
        if self.values.len() >= MAX_TENSORS || n > MAX_STORED_ELEMENTS - self.elements {
            return Err("session tensor storage limit exceeded".into());
        }
        if work > MAX_CALL_WORK || work > MAX_WORK - self.work {
            return Err("session tensor work limit exceeded".into());
        }
        self.work += work;
        Ok(())
    }
    fn store(&mut self, tensor: Tensor) -> Output {
        let id = self.values.len();
        self.elements += tensor.numel();
        self.values.push(tensor);
        Output::Handle(id)
    }
    fn get(&self, id: usize) -> Result<&Tensor, String> {
        self.values
            .get(id)
            .ok_or_else(|| "unknown tensor handle".into())
    }
    pub(crate) fn call(&mut self, name: &str, args: &[Argument]) -> Result<Output, String> {
        let count = match name {
            "wisp_tensor" => 3,
            "wisp_zeros" | "wisp_add" | "wisp_matmul" => 2,
            "wisp_transpose" | "wisp_shape" | "wisp_values" => 1,
            _ => return Err("host capability denied".into()),
        };
        if args.len() != count {
            return Err(format!("{name} expects {count} positional arguments"));
        }
        let integer = |i: usize| match &args[i] {
            Argument::Integer(n) => Ok(*n),
            _ => Err("expected a non-negative integer"),
        };
        if name == "wisp_tensor" || name == "wisp_zeros" {
            let offset = usize::from(name == "wisp_tensor");
            let rows = integer(offset)?;
            let cols = integer(offset + 1)?;
            let n = rows.checked_mul(cols).ok_or("tensor shape overflow")?;
            self.reserve(rows, cols, n)?;
            let data = if name == "wisp_tensor" {
                let Argument::Data(data) = &args[0] else {
                    return Err("expected a flat numeric sequence".into());
                };
                if data.len() != n {
                    return Err("data length does not match shape".into());
                }
                if !data.iter().all(|x| x.is_finite()) {
                    return Err("tensor inputs must be finite".into());
                }
                data.clone()
            } else {
                vec![0.0; n]
            };
            return Ok(self.store(Tensor::new(vec![rows, cols], data, Tag::Lit)));
        }
        let a = integer(0)?;
        let first = self.get(a)?;
        match name {
            "wisp_shape" => return Ok(Output::Shape(first.shape.clone())),
            "wisp_values" => return Ok(Output::Data(first.data.clone())),
            _ => {}
        }
        let (rows, cols) = (first.rows(), first.cols());
        let result = if name == "wisp_transpose" {
            self.reserve(cols, rows, rows * cols)?;
            self.get(a)?.transpose()?
        } else {
            let b = integer(1)?;
            let other = self.get(b)?;
            if name == "wisp_add" {
                if first.shape != other.shape {
                    return Err("addition requires equal shapes".into());
                }
                self.reserve(rows, cols, rows * cols)?;
                self.get(a)?.zip(self.get(b)?, |a, b| a + b)?
            } else {
                if cols != other.rows() {
                    return Err("matrix inner dimensions differ".into());
                }
                let out_cols = other.cols();
                let work = rows
                    .checked_mul(cols)
                    .and_then(|n| n.checked_mul(out_cols))
                    .ok_or("matrix work overflow")?;
                self.reserve(rows, out_cols, work)?;
                self.get(a)?.matmul(self.get(b)?)?
            }
        };
        Ok(self.store(result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn ints(v: &[usize]) -> Vec<Argument> {
        v.iter().copied().map(Argument::Integer).collect()
    }
    #[test]
    fn deny_unknown_capabilities_and_invalid_handles() {
        let mut t = Tensors::default();
        assert!(t.call("wload", &[]).is_err());
        assert!(t.call("wisp_values", &ints(&[0])).is_err());
    }
    #[test]
    fn allocations_and_work_checked_before_engine_call() {
        let mut t = Tensors::default();
        assert!(t
            .call("wisp_zeros", &ints(&[usize::MAX, usize::MAX]))
            .is_err());
        assert!(t.call("wisp_zeros", &ints(&[0, usize::MAX])).is_err());
        t.call("wisp_zeros", &ints(&[256, 256])).unwrap();
        assert!(t
            .call("wisp_matmul", &ints(&[0, 0]))
            .err()
            .unwrap()
            .contains("work limit"));
    }
    #[test]
    fn storage_cannot_be_reset_by_script() {
        let mut t = Tensors::default();
        for _ in 0..4 {
            t.call("wisp_zeros", &ints(&[256, 256])).unwrap();
        }
        assert!(t
            .call("wisp_zeros", &ints(&[1, 1]))
            .err()
            .unwrap()
            .contains("storage limit"));
        assert!(t.call("budget", &ints(&[999999])).is_err());
    }
}
