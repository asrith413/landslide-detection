use std::io::Cursor;

use base64::{engine::general_purpose::STANDARD, Engine};
use image::{DynamicImage, GrayImage, ImageFormat, RgbaImage};
use landslide_core::{overlay_rgba, SegmentOptions, Segmenter};
use serde::Serialize;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SegmentResponse {
    width: usize,
    height: usize,
    coverage: f32,
    latency_ms: f64,
    tiles: usize,
    original_png: String,
    mask_png: String,
    overlay_png: String,
}

fn encode_png(image: DynamicImage) -> Result<String, String> {
    let mut buffer = Cursor::new(Vec::new());
    image
        .write_to(&mut buffer, ImageFormat::Png)
        .map_err(|e| e.to_string())?;
    Ok(format!(
        "data:image/png;base64,{}",
        STANDARD.encode(buffer.into_inner())
    ))
}

#[tauri::command]
fn segment_image(
    model_path: String,
    image_path: String,
    threshold: f32,
) -> Result<SegmentResponse, String> {
    let options = SegmentOptions {
        tile_size: 128,
        overlap: 32,
        threshold,
        ..SegmentOptions::default()
    };
    let segmenter = Segmenter::new(&model_path, options).map_err(|e| e.to_string())?;
    let result = segmenter
        .segment_path(&image_path)
        .map_err(|e| e.to_string())?;
    let rgb = image::open(&image_path)
        .map_err(|e| e.to_string())?
        .to_rgb8();
    let (width, height) = rgb.dimensions();
    let original = encode_png(DynamicImage::ImageRgb8(rgb))?;
    let mask_bytes: Vec<u8> = result
        .mask
        .iter()
        .map(|v| if *v > 0 { 255 } else { 0 })
        .collect();
    let mask_image =
        GrayImage::from_raw(width, height, mask_bytes).ok_or("mask buffer size mismatch")?;
    let mask = encode_png(DynamicImage::ImageLuma8(mask_image))?;
    let overlay_bytes = overlay_rgba(
        image::open(&image_path)
            .map_err(|e| e.to_string())?
            .to_rgb8()
            .as_raw(),
        result.width,
        result.height,
        &result.probability,
        threshold,
        0.45,
    )
    .map_err(|e| e.to_string())?;
    let overlay_image =
        RgbaImage::from_raw(width, height, overlay_bytes).ok_or("overlay buffer size mismatch")?;
    let overlay = encode_png(DynamicImage::ImageRgba8(overlay_image))?;
    Ok(SegmentResponse {
        width: result.width,
        height: result.height,
        coverage: result.coverage,
        latency_ms: result.latency_ms,
        tiles: result.tiles,
        original_png: original,
        mask_png: mask,
        overlay_png: overlay,
    })
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![segment_image])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
