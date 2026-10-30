use std::path::Path;
use std::time::Instant;

use crate::model::LandslideModel;
use crate::preprocess::{preprocess_rgb, IMAGENET_MEAN, IMAGENET_STD};
use crate::tiling::{stitch_tiles, tile_positions, Tile};
use crate::CoreError;

#[derive(Clone, Debug)]
pub struct SegmentOptions {
    pub tile_size: usize,
    pub overlap: usize,
    pub threshold: f32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Default for SegmentOptions {
    fn default() -> Self {
        Self {
            tile_size: 128,
            overlap: 32,
            threshold: 0.5,
            mean: IMAGENET_MEAN,
            std: IMAGENET_STD,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Segmentation {
    pub width: usize,
    pub height: usize,
    pub probability: Vec<f32>,
    pub mask: Vec<u8>,
    pub coverage: f32,
    pub tiles: usize,
    pub latency_ms: f64,
}

pub struct Segmenter {
    model: LandslideModel,
    options: SegmentOptions,
}

impl Segmenter {
    pub fn new(model_path: impl AsRef<Path>, options: SegmentOptions) -> Result<Self, CoreError> {
        if options.tile_size == 0 {
            return Err(CoreError::Geometry("tile size must be positive".into()));
        }
        if options.overlap >= options.tile_size {
            return Err(CoreError::Geometry(
                "overlap must be smaller than tile".into(),
            ));
        }
        let model = LandslideModel::load(model_path, options.tile_size)?;
        Ok(Self { model, options })
    }

    pub fn options(&self) -> &SegmentOptions {
        &self.options
    }

    pub fn segment_path(&self, path: impl AsRef<Path>) -> Result<Segmentation, CoreError> {
        let image = image::open(path.as_ref())?.to_rgb8();
        let (width, height) = image.dimensions();
        self.segment_rgb(image.as_raw(), width as usize, height as usize)
    }

    pub fn segment_rgb(
        &self,
        rgb: &[u8],
        width: usize,
        height: usize,
    ) -> Result<Segmentation, CoreError> {
        if width == 0 || height == 0 {
            return Err(CoreError::Geometry("image must be non-empty".into()));
        }
        if rgb.len() != width * height * 3 {
            return Err(CoreError::Buffer("RGB buffer size mismatch".into()));
        }
        let started = Instant::now();
        let tiles = tile_positions(height, width, self.options.tile_size, self.options.overlap)?;
        let mut probs: Vec<(Tile, Vec<f32>)> = Vec::with_capacity(tiles.len());
        for tile in &tiles {
            let padded = extract_replicated(rgb, width, height, tile, self.options.tile_size);
            let input = preprocess_rgb(
                &padded,
                self.options.tile_size,
                self.options.tile_size,
                self.options.mean,
                self.options.std,
            )?;
            let full = self.model.predict(&input)?;
            let mut cropped = Vec::with_capacity(tile.h * tile.w);
            for y in 0..tile.h {
                let start = y * self.options.tile_size;
                cropped.extend_from_slice(&full[start..start + tile.w]);
            }
            probs.push((*tile, cropped));
        }
        let probability = stitch_tiles(&probs, height, width, self.options.overlap)?;
        let mut mask = vec![0u8; width * height];
        let mut count = 0usize;
        for (i, p) in probability.iter().enumerate() {
            if *p >= self.options.threshold {
                mask[i] = 1;
                count += 1;
            }
        }
        let latency_ms = started.elapsed().as_secs_f64() * 1000.0;
        Ok(Segmentation {
            width,
            height,
            probability,
            mask,
            coverage: count as f32 / (width * height) as f32,
            tiles: tiles.len(),
            latency_ms,
        })
    }
}

fn extract_replicated(
    rgb: &[u8],
    width: usize,
    height: usize,
    tile: &Tile,
    size: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; size * size * 3];
    for y in 0..size {
        let sy = (tile.y + y).min(height - 1);
        for x in 0..size {
            let sx = (tile.x + x).min(width - 1);
            let src = (sy * width + sx) * 3;
            let dst = (y * size + x) * 3;
            out[dst..dst + 3].copy_from_slice(&rgb[src..src + 3]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replicated_tile_matches_edges() {
        let (w, h) = (4, 4);
        let rgb: Vec<u8> = (0..w * h * 3).map(|i| (i % 255) as u8).collect();
        let tile = Tile {
            y: 2,
            x: 2,
            h: 2,
            w: 2,
        };
        let out = extract_replicated(&rgb, w, h, &tile, 4);
        assert_eq!(out.len(), 4 * 4 * 3);
        let src = (2 * w + 2) * 3;
        assert_eq!(&out[0..3], &rgb[src..src + 3]);
    }
}
