use crate::CoreError;

pub fn overlay_rgba(
    rgb: &[u8],
    width: usize,
    height: usize,
    probability: &[f32],
    threshold: f32,
    alpha: f32,
) -> Result<Vec<u8>, CoreError> {
    if rgb.len() != width * height * 3 {
        return Err(CoreError::Buffer("RGB buffer size mismatch".into()));
    }
    if probability.len() != width * height {
        return Err(CoreError::Buffer("probability buffer size mismatch".into()));
    }
    let a = alpha.clamp(0.0, 1.0);
    let color = [220.0f32, 40.0, 40.0];
    let mut out = vec![0u8; width * height * 4];
    for i in 0..width * height {
        let mask = probability[i] >= threshold;
        for c in 0..3 {
            let base = rgb[i * 3 + c] as f32;
            let v = if mask {
                base * (1.0 - a) + color[c] * a
            } else {
                base
            };
            out[i * 4 + c] = v.clamp(0.0, 255.0) as u8;
        }
        out[i * 4 + 3] = 255;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlay_size_and_alpha() {
        let rgb = vec![100u8; 4 * 4 * 3];
        let prob = vec![0.9f32; 16];
        let out = overlay_rgba(&rgb, 4, 4, &prob, 0.5, 0.5).unwrap();
        assert_eq!(out.len(), 4 * 4 * 4);
        assert!(out[0] > 100);
        assert_eq!(out[3], 255);
    }

    #[test]
    fn rejects_mismatch() {
        assert!(overlay_rgba(&[0u8; 3], 4, 4, &[0.0; 16], 0.5, 0.5).is_err());
    }
}
