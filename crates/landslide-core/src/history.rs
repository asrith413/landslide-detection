use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::CoreError;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub id: String,
    pub created_at_ms: u64,
    pub file_name: String,
    pub width: usize,
    pub height: usize,
    pub threshold: f32,
    pub coverage: f32,
    pub latency_ms: f64,
    pub model: String,
}

pub struct HistoryStore {
    path: PathBuf,
    limit: usize,
}

impl HistoryStore {
    pub fn new(path: impl AsRef<Path>, limit: usize) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            limit: limit.max(1),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn load(&self) -> Result<Vec<HistoryEntry>, CoreError> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }
        let text = std::fs::read_to_string(&self.path)?;
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }
        Ok(serde_json::from_str(&text)?)
    }

    pub fn append(&self, mut entry: HistoryEntry) -> Result<Vec<HistoryEntry>, CoreError> {
        if entry.created_at_ms == 0 {
            entry.created_at_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);
        }
        let mut items = self.load()?;
        items.insert(0, entry);
        items.truncate(self.limit);
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&self.path, serde_json::to_string_pretty(&items)?)?;
        Ok(items)
    }

    pub fn clear(&self) -> Result<(), CoreError> {
        if self.path.exists() {
            std::fs::remove_file(&self.path)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(name: &str) -> HistoryEntry {
        HistoryEntry {
            id: name.into(),
            created_at_ms: 1,
            file_name: format!("{name}.png"),
            width: 128,
            height: 128,
            threshold: 0.5,
            coverage: 0.1,
            latency_ms: 12.0,
            model: "test".into(),
        }
    }

    #[test]
    fn append_and_limit() {
        let dir = tempfile::tempdir().unwrap();
        let store = HistoryStore::new(dir.path().join("history.json"), 2);
        store.append(entry("a")).unwrap();
        store.append(entry("b")).unwrap();
        let items = store.append(entry("c")).unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].id, "c");
        assert_eq!(store.load().unwrap().len(), 2);
        store.clear().unwrap();
        assert!(store.load().unwrap().is_empty());
    }
}
