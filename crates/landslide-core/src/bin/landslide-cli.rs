use std::path::PathBuf;

use landslide_core::{overlay_rgba, SegmentOptions, Segmenter};

fn usage() -> ! {
    eprintln!(
        "usage: landslide-cli --model MODEL.onnx --input IMAGE [--mask OUT.png] \
[--prob OUT.png] [--overlay OUT.png] [--threshold 0.5] [--tile 128] [--overlap 32]"
    );
    std::process::exit(2)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model: Option<PathBuf> = None;
    let mut input: Option<PathBuf> = None;
    let mut mask: Option<PathBuf> = None;
    let mut prob: Option<PathBuf> = None;
    let mut overlay: Option<PathBuf> = None;
    let mut threshold = 0.5f32;
    let mut tile = 128usize;
    let mut overlap = 32usize;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let value = || args.get(i + 1).cloned().unwrap_or_else(|| usage());
        match args[i].as_str() {
            "--model" => model = Some(PathBuf::from(value())),
            "--input" => input = Some(PathBuf::from(value())),
            "--mask" => mask = Some(PathBuf::from(value())),
            "--prob" => prob = Some(PathBuf::from(value())),
            "--overlay" => overlay = Some(PathBuf::from(value())),
            "--threshold" => threshold = value().parse()?,
            "--tile" => tile = value().parse()?,
            "--overlap" => overlap = value().parse()?,
            "-h" | "--help" => usage(),
            other => {
                eprintln!("unknown argument: {other}");
                usage();
            }
        }
        i += 2;
    }

    let model = model.unwrap_or_else(|| usage());
    let input = input.unwrap_or_else(|| usage());
    let options = SegmentOptions {
        tile_size: tile,
        overlap,
        threshold,
        ..SegmentOptions::default()
    };
    let segmenter = Segmenter::new(&model, options)?;
    let result = segmenter.segment_path(&input)?;

    if let Some(path) = mask {
        let bytes: Vec<u8> = result
            .mask
            .iter()
            .map(|v| if *v > 0 { 255 } else { 0 })
            .collect();
        image::save_buffer(
            path,
            &bytes,
            result.width as u32,
            result.height as u32,
            image::ColorType::L8,
        )?;
    }
    if let Some(path) = prob {
        let bytes: Vec<u8> = result
            .probability
            .iter()
            .map(|v| (v.clamp(0.0, 1.0) * 255.0) as u8)
            .collect();
        image::save_buffer(
            path,
            &bytes,
            result.width as u32,
            result.height as u32,
            image::ColorType::L8,
        )?;
    }
    if let Some(path) = overlay {
        let rgb = image::open(&input)?.to_rgb8();
        let rgba = overlay_rgba(
            rgb.as_raw(),
            result.width,
            result.height,
            &result.probability,
            threshold,
            0.45,
        )?;
        image::save_buffer(
            path,
            &rgba,
            result.width as u32,
            result.height as u32,
            image::ColorType::Rgba8,
        )?;
    }

    println!(
        "{{\"width\":{},\"height\":{},\"tiles\":{},\"coverage\":{:.6},\"latency_ms\":{:.3}}}",
        result.width, result.height, result.tiles, result.coverage, result.latency_ms
    );
    Ok(())
}
