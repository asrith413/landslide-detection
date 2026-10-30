pub mod history;
pub mod model;
pub mod overlay;
pub mod pipeline;
pub mod preprocess;
pub mod tiling;

pub use history::{HistoryEntry, HistoryStore};
pub use model::LandslideModel;
pub use overlay::overlay_rgba;
pub use pipeline::{SegmentOptions, Segmentation, Segmenter};
pub use preprocess::{percentile_stretch, preprocess_rgb, IMAGENET_MEAN, IMAGENET_STD};
pub use tiling::{blend_window, stitch_tiles, tile_positions, Tile};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("invalid geometry: {0}")]
    Geometry(String),
    #[error("invalid buffer: {0}")]
    Buffer(String),
    #[error("model error: {0}")]
    Model(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("image error: {0}")]
    Image(#[from] image::ImageError),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}
