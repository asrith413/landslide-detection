use std::path::Path;
use std::sync::Arc;

use tract_onnx::prelude::tract_ndarray::Array4;
use tract_onnx::prelude::*;

use crate::CoreError;

pub struct LandslideModel {
    model: Arc<RunnableModel<TypedFact, Box<dyn TypedOp>>>,
    input_size: usize,
}

impl LandslideModel {
    pub fn load(path: impl AsRef<Path>, input_size: usize) -> Result<Self, CoreError> {
        if input_size == 0 {
            return Err(CoreError::Geometry("input size must be positive".into()));
        }
        let model = tract_onnx::onnx()
            .model_for_path(path.as_ref())
            .map_err(|e| CoreError::Model(e.to_string()))?
            .into_optimized()
            .map_err(|e| CoreError::Model(e.to_string()))?
            .into_runnable()
            .map_err(|e| CoreError::Model(e.to_string()))?;
        Ok(Self { model, input_size })
    }

    pub fn input_size(&self) -> usize {
        self.input_size
    }

    pub fn predict(&self, input: &[f32]) -> Result<Vec<f32>, CoreError> {
        let n = self.input_size;
        if input.len() != 3 * n * n {
            return Err(CoreError::Buffer(format!(
                "expected {} input values, got {}",
                3 * n * n,
                input.len()
            )));
        }
        let tensor = Array4::from_shape_fn((1, 3, n, n), |(_, c, y, x)| input[c * n * n + y * n + x]);
        let result = self
            .model
            .run(tvec!(tensor.into_tensor().into()))
            .map_err(|e| CoreError::Model(e.to_string()))?;
        let view = result[0]
            .to_plain_array_view::<f32>()
            .map_err(|e| CoreError::Model(e.to_string()))?;
        let values: Vec<f32> = view.iter().copied().collect();
        if values.len() != n * n {
            return Err(CoreError::Model(format!(
                "expected {} output values, got {}",
                n * n,
                values.len()
            )));
        }
        Ok(values)
    }
}
