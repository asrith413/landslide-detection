use crate::CoreError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tile {
    pub y: usize,
    pub x: usize,
    pub h: usize,
    pub w: usize,
}

fn starts(total: usize, tile: usize, stride: usize) -> Vec<usize> {
    if total <= tile {
        return vec![0];
    }
    let last = total - tile;
    let mut vals: Vec<usize> = (0..=last).step_by(stride).collect();
    if vals.last().copied() != Some(last) {
        vals.push(last);
    }
    vals
}

pub fn tile_positions(
    height: usize,
    width: usize,
    tile: usize,
    overlap: usize,
) -> Result<Vec<Tile>, CoreError> {
    if tile == 0 {
        return Err(CoreError::Geometry("tile must be positive".into()));
    }
    if overlap >= tile {
        return Err(CoreError::Geometry("overlap must be smaller than tile".into()));
    }
    if height == 0 || width == 0 {
        return Err(CoreError::Geometry("image must be non-empty".into()));
    }
    let stride = tile - overlap;
    let mut out = Vec::new();
    for y in starts(height, tile, stride) {
        for x in starts(width, tile, stride) {
            out.push(Tile {
                y,
                x,
                h: tile.min(height - y),
                w: tile.min(width - x),
            });
        }
    }
    Ok(out)
}

pub fn blend_window(h: usize, w: usize, overlap: usize) -> Vec<f32> {
    let ov_y = overlap.min(h.saturating_sub(1) / 2);
    let ov_x = overlap.min(w.saturating_sub(1) / 2);
    let mut wy = vec![1.0f32; h];
    let mut wx = vec![1.0f32; w];
    if ov_y > 0 {
        for i in 0..ov_y {
            let v = 1.0 / (ov_y as f32 + 1.0) + (1.0 - 1.0 / (ov_y as f32 + 1.0)) * i as f32
                / (ov_y as f32 - 1.0).max(1.0);
            wy[i] = v;
            wy[h - 1 - i] = v;
        }
    }
    if ov_x > 0 {
        for i in 0..ov_x {
            let v = 1.0 / (ov_x as f32 + 1.0) + (1.0 - 1.0 / (ov_x as f32 + 1.0)) * i as f32
                / (ov_x as f32 - 1.0).max(1.0);
            wx[i] = v;
            wx[w - 1 - i] = v;
        }
    }
    let mut out = vec![0.0f32; h * w];
    for y in 0..h {
        for x in 0..w {
            out[y * w + x] = wy[y] * wx[x];
        }
    }
    out
}

pub fn stitch_tiles(
    tiles: &[(Tile, Vec<f32>)],
    height: usize,
    width: usize,
    overlap: usize,
) -> Result<Vec<f32>, CoreError> {
    let mut acc = vec![0.0f32; height * width];
    let mut weight = vec![0.0f32; height * width];
    for (tile, prob) in tiles {
        if prob.len() != tile.h * tile.w {
            return Err(CoreError::Buffer(format!(
                "tile probability has {} values, expected {}",
                prob.len(),
                tile.h * tile.w
            )));
        }
        if tile.y + tile.h > height || tile.x + tile.w > width {
            return Err(CoreError::Geometry("tile exceeds image bounds".into()));
        }
        let win = blend_window(tile.h, tile.w, overlap);
        for y in 0..tile.h {
            for x in 0..tile.w {
                let gi = (tile.y + y) * width + tile.x + x;
                let wi = y * tile.w + x;
                acc[gi] += prob[wi] * win[wi];
                weight[gi] += win[wi];
            }
        }
    }
    for i in 0..acc.len() {
        acc[i] = (acc[i] / weight[i].max(1e-6)).clamp(0.0, 1.0);
    }
    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn covers_edges() {
        let tiles = tile_positions(300, 200, 128, 32).unwrap();
        let last = tiles.last().unwrap();
        assert_eq!(last.y + last.h, 300);
        assert_eq!(last.x + last.w, 200);
    }

    #[test]
    fn constant_map_is_preserved() {
        let (h, w, t, ov) = (256, 192, 128, 32);
        let tiles: Vec<(Tile, Vec<f32>)> = tile_positions(h, w, t, ov)
            .unwrap()
            .into_iter()
            .map(|tile| (tile, vec![0.7f32; tile.h * tile.w]))
            .collect();
        let out = stitch_tiles(&tiles, h, w, ov).unwrap();
        assert!(out.iter().all(|v| (v - 0.7).abs() < 1e-5));
    }

    #[test]
    fn small_image_is_preserved() {
        let (h, w, t, ov) = (40, 24, 128, 32);
        let tiles: Vec<(Tile, Vec<f32>)> = tile_positions(h, w, t, ov)
            .unwrap()
            .into_iter()
            .map(|tile| (tile, vec![0.4f32; tile.h * tile.w]))
            .collect();
        let out = stitch_tiles(&tiles, h, w, ov).unwrap();
        assert!(out.iter().all(|v| (v - 0.4).abs() < 1e-5));
    }

    #[test]
    fn rejects_bad_overlap() {
        assert!(tile_positions(100, 100, 128, 128).is_err());
    }
}
