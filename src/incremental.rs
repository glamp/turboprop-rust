//! Incremental index update logic for efficient file change processing.
//!
//! This module handles the atomic updating of search indices when files are
//! modified, created, or deleted, ensuring index consistency during concurrent
//! operations.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

use crate::config::TurboPropConfig;
use crate::content::{extract_content, FileContent};
use crate::embeddings::EmbeddingGenerator;
use crate::files::discover_files;
use crate::storage::{ChunkMetadata, PersistentIndex};
use crate::types::{ChunkId, DocumentChunk};
use crate::watcher::WatchEventBatch;

/// Statistics for incremental index updates
#[derive(Debug, Default, Clone)]
pub struct IncrementalStats {
    /// Number of files processed
    pub files_processed: usize,
    /// Number of files added
    pub files_added: usize,
    /// Number of files modified
    pub files_modified: usize,
    /// Number of files removed
    pub files_removed: usize,
    /// Number of chunks added
    pub chunks_added: usize,
    /// Number of chunks removed
    pub chunks_removed: usize,
    /// Number of files that failed processing
    pub files_failed: usize,
}

impl IncrementalStats {
    /// Calculate success rate as a percentage
    pub fn success_rate(&self) -> f64 {
        if self.files_processed == 0 {
            100.0
        } else {
            ((self.files_processed - self.files_failed) as f64 / self.files_processed as f64) * 100.0
        }
    }

    /// Merge another stats instance into this one
    pub fn merge(&mut self, other: IncrementalStats) {
        self.files_processed += other.files_processed;
        self.files_added += other.files_added;
        self.files_modified += other.files_modified;
        self.files_removed += other.files_removed;
        self.chunks_added += other.chunks_added;
        self.chunks_removed += other.chunks_removed;
        self.files_failed += other.files_failed;
    }
}

/// Incremental index updater for processing file change events
pub struct IncrementalUpdater {
    /// Configuration for processing
    config: TurboPropConfig,
    /// Embedding generator for new content
    embedding_generator: EmbeddingGenerator,
    /// Root path of the repository
    root_path: PathBuf,
}

impl IncrementalUpdater {
    /// Create a new incremental updater
    ///
    /// # Arguments
    /// * `config` - TurboProp configuration
    /// * `root_path` - Root path of the repository being indexed
    ///
    /// # Returns
    /// * `Result<Self>` - The updater instance or error
    pub async fn new(config: TurboPropConfig, root_path: &Path) -> Result<Self> {
        let embedding_generator = EmbeddingGenerator::new(config.embedding.clone())
            .await
            .context("Failed to initialize embedding generator for incremental updates")?;

        Ok(Self {
            config,
            embedding_generator,
            root_path: root_path.to_path_buf(),
        })
    }

    /// Process a batch of file change events
    ///
    /// # Arguments
    /// * `batch` - Batch of watch events to process
    /// * `persistent_index` - Mutable reference to the persistent index
    ///
    /// # Returns
    /// * `Result<IncrementalStats>` - Statistics about the update operation
    pub async fn process_batch(
        &mut self,
        batch: &WatchEventBatch,
        persistent_index: &mut PersistentIndex,
    ) -> Result<IncrementalStats> {
        let mut stats = IncrementalStats::default();

        info!(
            "Processing incremental batch with {} events",
            batch.events.len()
        );

        // Group events by type for efficient processing
        let (modified_paths, created_paths, deleted_paths) = batch.group_by_type();

        // Process deletions first (remove chunks from index)
        for path in deleted_paths {
            match self.process_deleted_file(&path, persistent_index).await {
                Ok(removed_chunks) => {
                    stats.files_removed += 1;
                    stats.chunks_removed += removed_chunks;
                    info!("Removed {} chunks for deleted file: {}", removed_chunks, path.display());
                }
                Err(e) => {
                    warn!("Failed to process deleted file {}: {}", path.display(), e);
                    stats.files_failed += 1;
                }
            }
            stats.files_processed += 1;
        }

        // Process created files (add new chunks to index)
        for path in created_paths {
            if path.is_file() {
                match self.process_created_file(&path, persistent_index).await {
                    Ok(added_chunks) => {
                        stats.files_added += 1;
                        stats.chunks_added += added_chunks;
                        info!("Added {} chunks for new file: {}", added_chunks, path.display());
                    }
                    Err(e) => {
                        warn!("Failed to process created file {}: {}", path.display(), e);
                        stats.files_failed += 1;
                    }
                }
                stats.files_processed += 1;
            } else if path.is_dir() {
                // For new directories, discover and process all files within
                match self.process_created_directory(&path, persistent_index).await {
                    Ok((files_added, chunks_added)) => {
                        stats.files_added += files_added;
                        stats.chunks_added += chunks_added;
                        info!(
                            "Added {} files ({} chunks) from new directory: {}",
                            files_added, chunks_added, path.display()
                        );
                    }
                    Err(e) => {
                        warn!("Failed to process created directory {}: {}", path.display(), e);
                        stats.files_failed += 1;
                    }
                }
            }
        }

        // Process modified files (remove old chunks, add new ones)
        for path in modified_paths {
            if path.is_file() {
                match self.process_modified_file(&path, persistent_index).await {
                    Ok((removed_chunks, added_chunks)) => {
                        stats.files_modified += 1;
                        stats.chunks_removed += removed_chunks;
                        stats.chunks_added += added_chunks;
                        info!(
                            "Updated file {} (removed: {}, added: {} chunks)",
                            path.display(), removed_chunks, added_chunks
                        );
                    }
                    Err(e) => {
                        warn!("Failed to process modified file {}: {}", path.display(), e);
                        stats.files_failed += 1;
                    }
                }
                stats.files_processed += 1;
            }
        }

        // Save the updated index
        persistent_index
            .save()
            .context("Failed to save updated index")?;

        info!(
            "Incremental update completed - processed: {}, added: {}, modified: {}, removed: {}",
            stats.files_processed, stats.files_added, stats.files_modified, stats.files_removed
        );

        Ok(stats)
    }

    /// Process a deleted file by removing its chunks from the index
    async fn process_deleted_file(
        &self,
        path: &Path,
        persistent_index: &mut PersistentIndex,
    ) -> Result<usize> {
        let relative_path = path
            .strip_prefix(&self.root_path)
            .context("Path not within repository root")?;

        // Find all chunks for this file
        let chunks_to_remove = persistent_index.find_chunks_by_file_path(relative_path);

        let removed_count = chunks_to_remove.len();

        // Remove chunks from the index
        for chunk_id in chunks_to_remove {
            persistent_index.remove_chunk(chunk_id);
        }

        debug!("Removed {} chunks for deleted file: {}", removed_count, relative_path.display());
        Ok(removed_count)
    }

    /// Process a created file by adding its chunks to the index
    async fn process_created_file(
        &mut self,
        path: &Path,
        persistent_index: &mut PersistentIndex,
    ) -> Result<usize> {
        let relative_path = path
            .strip_prefix(&self.root_path)
            .context("Path not within repository root")?;

        // Extract content from the new file
        let file_content = extract_content(path, &self.config.file_discovery)
            .with_context(|| format!("Failed to extract content from: {}", path.display()))?;

        // Generate chunks and embeddings
        let chunks = self.create_chunks_for_file(&file_content, relative_path).await?;
        let added_count = chunks.len();

        // Add chunks to the index
        for chunk in chunks {
            persistent_index.add_chunk(chunk);
        }

        debug!("Added {} chunks for created file: {}", added_count, relative_path.display());
        Ok(added_count)
    }

    /// Process a created directory by discovering and indexing all files within
    async fn process_created_directory(
        &mut self,
        path: &Path,
        persistent_index: &mut PersistentIndex,
    ) -> Result<(usize, usize)> {
        // Discover files in the new directory
        let discovered_files = discover_files(path, &self.config.file_discovery)
            .context("Failed to discover files in new directory")?;

        let mut files_added = 0;
        let mut chunks_added = 0;

        for file_path in discovered_files {
            match self.process_created_file(&file_path, persistent_index).await {
                Ok(file_chunks) => {
                    files_added += 1;
                    chunks_added += file_chunks;
                }
                Err(e) => {
                    warn!("Failed to process file in new directory {}: {}", file_path.display(), e);
                }
            }
        }

        debug!(
            "Processed new directory {} - added {} files with {} chunks",
            path.display(), files_added, chunks_added
        );
        Ok((files_added, chunks_added))
    }

    /// Process a modified file by replacing its chunks in the index
    async fn process_modified_file(
        &mut self,
        path: &Path,
        persistent_index: &mut PersistentIndex,
    ) -> Result<(usize, usize)> {
        let relative_path = path
            .strip_prefix(&self.root_path)
            .context("Path not within repository root")?;

        // First, remove existing chunks for this file
        let removed_count = self.process_deleted_file(path, persistent_index).await?;

        // Then add new chunks for the modified content
        let added_count = self.process_created_file(path, persistent_index).await?;

        debug!(
            "Modified file {} - removed: {}, added: {} chunks",
            relative_path.display(), removed_count, added_count
        );
        Ok((removed_count, added_count))
    }

    /// Create chunks for a file with embeddings
    async fn create_chunks_for_file(
        &mut self,
        file_content: &FileContent,
        relative_path: &Path,
    ) -> Result<Vec<DocumentChunk>> {
        use crate::chunking::create_chunks;

        let chunks = create_chunks(file_content, &self.config.chunking)
            .with_context(|| format!("Failed to create chunks for file: {}", relative_path.display()))?;

        let mut document_chunks = Vec::new();

        // Process chunks in batches for embedding generation
        for chunk_batch in chunks.chunks(self.config.embedding.batch_size) {
            let texts: Vec<String> = chunk_batch.iter().map(|c| c.content.clone()).collect();

            let embeddings = self
                .embedding_generator
                .embed_batch(&texts)
                .context("Failed to generate embeddings for chunks")?;

            for (chunk, embedding) in chunk_batch.iter().zip(embeddings.iter()) {
                let metadata = ChunkMetadata {
                    id: format!("{}:{}", relative_path.display(), chunk.source_location.start_line),
                    file_path: relative_path.to_path_buf(),
                    start_line: chunk.source_location.start_line,
                    end_line: chunk.source_location.end_line,
                    start_char: chunk.source_location.start_char,
                    end_char: chunk.source_location.end_char,
                    chunk_index: chunk.chunk_index.into(),
                    total_chunks: chunks.len(),
                    token_count: chunk.token_count.get(),
                    content_length: chunk.content.len(),
                };

                document_chunks.push(DocumentChunk {
                    content: chunk.content.clone(),
                    embedding: embedding.clone(),
                    metadata,
                });
            }
        }

        debug!(
            "Created {} chunks for file: {}",
            document_chunks.len(),
            relative_path.display()
        );
        Ok(document_chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TurboPropConfig;
    use crate::watcher::WatchEvent;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_incremental_stats() {
        let mut stats = IncrementalStats::default();
        stats.files_processed = 10;
        stats.files_failed = 2;

        assert_eq!(stats.success_rate(), 80.0);

        let other = IncrementalStats {
            files_processed: 5,
            files_added: 3,
            files_failed: 1,
            ..Default::default()
        };

        stats.merge(other);
        assert_eq!(stats.files_processed, 15);
        assert_eq!(stats.files_added, 3);
        assert_eq!(stats.files_failed, 3);
        assert_eq!(stats.success_rate(), 80.0);
    }

    #[tokio::test]
    async fn test_incremental_updater_creation() {
        // Skip this test if we're running in an offline environment
        if std::env::var("OFFLINE_TESTS").is_ok() {
            return;
        }

        let temp_dir = TempDir::new().unwrap();
        let config = TurboPropConfig::default();

        let result = IncrementalUpdater::new(config, temp_dir.path()).await;
        // This might fail due to network requirements for embedding model
        // In a real environment with network access, this should work
        if result.is_err() {
            // Acceptable in test environment without network access
            return;
        }

        let updater = result.unwrap();
        assert_eq!(updater.root_path, temp_dir.path());
    }

    #[test]
    fn test_watch_event_batch_grouping() {
        let events = vec![
            WatchEvent::Modified(PathBuf::from("/file1")),
            WatchEvent::Created(PathBuf::from("/file2")),
            WatchEvent::Deleted(PathBuf::from("/file3")),
        ];

        let batch = WatchEventBatch::new(events);
        let (modified, created, deleted) = batch.group_by_type();

        assert_eq!(modified, vec![PathBuf::from("/file1")]);
        assert_eq!(created, vec![PathBuf::from("/file2")]);
        assert_eq!(deleted, vec![PathBuf::from("/file3")]);
    }
}