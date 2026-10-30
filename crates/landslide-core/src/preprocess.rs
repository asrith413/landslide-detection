use crate::CoreError;

pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

fn percentile(sorted: &[u8], p: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (sorted.len() - 1) as f32 * p.clamp(0.0, 100.0) / 100.0;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    let frac = rank - lo as f32;
    sorted[lo] as f32 * (1.0 - frac) + sorted[hi] as f32 * frac
}

pub fn percentile_stretch(rgb: &[u8], low: f32, high: f32) -> Vec<u8> {
    let pixels = rgb.len() / 3;
    let mut out = vec![0u8; rgb.len()];
    for c in 0..3 {
        let mut ch: Vec<u8> = (0..pixels).map(|i| rgb[i * 3 + c]).collect();
        ch.sort_unstable();
        let mut lo = percentile(&ch, low);
        let mut hi = percentile(&ch, high);
        if hi - lo < 1e-6 {
            lo = *ch.first().unwrap_or(&0) as f32;
            hi = *ch.last().unwrap_or(&0) as f32;
        }
        for i in 0..pixels {
            let v = if hi - lo < 1e-6 {
                0.0
            } else {
                (rgb[i * 3 + c] as f32 - lo) / (hi - lo) * 255.0
            };
            out[i * 3 + c] = v.clamp(0.0, 255.0) as u8;
        }
    }
    out
}

pub fn preprocess_rgb(
    rgb: &[u8],
    width: usize,
    height: usize,
    mean: [f32; 3],
    std: [f32; 3],
) -> Result<Vec<f32>, CoreError> {
    if width == 0 || height == 0 {
        return Err(CoreError::Geometry("image must be non-empty".into()));
    }
    if rgb.len() != width * height * 3 {
        return Err(CoreError::Buffer(format!(
            "expected {} RGB bytes, got {}",
            width * height * 3,
            rgb.len()
        )));
    }
    let stretched = percentile_stretch(rgb, 2.0, 98.0);
    let mut out = vec![0.0f32; 3 * width * height];
    for c in 0..3 {
        for y in 0..height {
            for x in 0..width {
                let v = stretched[(y * width + x) * 3 + c] as f32 / 255.0;
                out[c * width * height + y * width + x] = (v - mean[c]) / std[c];
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_image_maps_to_zero() {
        let rgb = vec![128u8; 8 * 8 * 3];
        let out = percentile_stretch(&rgb, 2.0, 98.0);
        assert!(out.iter().all(|v| *v == 0));
    }

    #[test]
    fn preprocess_shape_and_range() {
        let mut rgb = vec![0u8; 16 * 16 * 3];
        for (i, v) in rgb.iter_mut().enumerate() {
            *v = (i % 255) as u8;
        }
        let out = preprocess_rgb(&rgb, 16, 16, IMAGENET_MEAN, IMAGENET_STD).unwrap();
        assert_eq!(out.len(), 3 * 16 * 16);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn rejects_wrong_buffer() {
        assert!(preprocess_rgb(&[0u8; 10], 4, 4, IMAGENET_MEAN, IMAGENET_STD).is_err());
    }
}
