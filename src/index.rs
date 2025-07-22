//! Vector index management with persistence capabilities.
//!
//! This module extends the existing ChunkIndex with persistent storage,
//! incremental updates, and concurrent access safety.

use anyhow::Result;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

use crate::chunking::ChunkingStrategy;
use crate::config::TurboPropConfig;
use crate::embeddings::EmbeddingGenerator;
use crate::files::FileDiscovery;
use crate::storage::{IndexConfig, IndexStorage};
use crate::types::{ChunkIndex, FileMetadata, IndexedChunk};

/// Enhanced persistent vector index that extends ChunkIndex with storage capabilities
#[derive(Debug)]
pub struct PersistentChunkIndex {
    /// In-memory chunk index for fast queries
    chunk_index: ChunkIndex,
    /// Storage backend for persistence
    storage: IndexStorage,
    /// Configuration used to build this index
    config: IndexConfig,
    /// Base path that was indexed
    indexed_path: PathBuf,
    /// Storage version used for this index
    storage_version: String,
}

impl PersistentChunkIndex {
    /// Create a new persistent chunk index
    pub fn new<P: AsRef<Path>>(base_path: P) -> Result<Self> {
        let indexed_path = base_path.as_ref().to_path_buf();
        let storage = IndexStorage::new(&indexed_path)?;
        let chunk_index = ChunkIndex::new();
        let config = IndexConfig::default();

        Ok(Self {
            chunk_index,
            storage,
            config,
            indexed_path,
            storage_version: TurboPropConfig::default().general.storage_version,
        })
    }

    /// Load an existing index from disk
    pub fn load<P: AsRef<Path>>(base_path: P) -> Result<Self> {
        let indexed_path = base_path.as_ref().to_path_buf();
        let storage = IndexStorage::new(&indexed_path)?;

        if !storage.index_exists() {
            anyhow::bail!("No index found at {}", indexed_path.display());
        }

        // Use default config for loading, which will be replaced by the actual config from disk
        let default_config = TurboPropConfig::default();
        let (indexed_chunks, config) =
            storage.load_index(&default_config.general.storage_version)?;
        let mut chunk_index = ChunkIndex::new();

        // Populate the in-memory index
        for indexed_chunk in indexed_chunks {
            chunk_index.add_chunk(indexed_chunk.chunk, indexed_chunk.embedding);
        }

        info!("Loaded persistent index with {} chunks", chunk_index.len());

        Ok(Self {
            chunk_index,
            storage,
            config,
            indexed_path,
            storage_version: default_config.general.storage_version,
        })
    }

    /// Build a complete index from the specified path with full configuration
    pub async fn build<P: AsRef<Path>>(base_path: P, config: &TurboPropConfig) -> Result<Self> {
        let indexed_path = base_path.as_ref().to_path_buf();
        info!("Building persistent index for: {}", indexed_path.display());

        if !indexed_path.exists() {
            anyhow::bail!("Path does not exist: {}", indexed_path.display());
        }

        if !indexed_path.is_dir() {
            anyhow::bail!("Path is not a directory: {}", indexed_path.display());
        }

        let storage = IndexStorage::new(&indexed_path)?;

        // Initialize embedding generator
        let mut embedding_generator = EmbeddingGenerator::new(config.embedding.clone()).await?;

        // Discover files
        let discovery = FileDiscovery::new(config.file_discovery.clone());
        let files = discovery.discover_files(&indexed_path)?;
        info!("Discovered {} files for indexing", files.len());

        // Create index configuration
        let index_config = IndexConfig {
            model_name: config.embedding.model_name.clone(),
            embedding_dimensions: config.embedding.embedding_dimensions,
            batch_size: config.embedding.batch_size,
            respect_gitignore: config.file_discovery.respect_gitignore,
            include_untracked: config.file_discovery.include_untracked,
        };

        // Process files and build index
        let mut chunk_index = ChunkIndex::new();
        let mut all_indexed_chunks = Vec::new();
        let chunking_strategy = ChunkingStrategy::new(config.chunking.clone());

        for file in &files {
            info!("Processing: {}", file.path.display());

            // Generate chunks from the file
            let chunks = chunking_strategy.chunk_file(&file.path)?;

            if chunks.is_empty() {
                debug!("No chunks generated for: {}", file.path.display());
                continue;
            }

            // Prepare text for embedding
            let chunk_texts: Vec<String> =
                chunks.iter().map(|chunk| chunk.content.clone()).collect();

            // Generate embeddings
            debug!("Generating embeddings for {} chunks", chunks.len());
            let embeddings = embedding_generator.embed_batch(&chunk_texts)?;

            // Create indexed chunks efficiently without cloning
            let batch_indexed_chunks: Vec<IndexedChunk> = chunks
                .into_iter()
                .zip(embeddings.into_iter())
                .map(|(chunk, embedding)| IndexedChunk { chunk, embedding })
                .collect();

            // Add to chunk_index efficiently without cloning (takes ownership)
            let chunks_for_index = batch_indexed_chunks.clone(); // This clone is necessary due to dual usage
            chunk_index.add_indexed_chunks(chunks_for_index);

            // Move the original chunks to the final collection
            all_indexed_chunks.extend(batch_indexed_chunks);
        }

        info!("Generated {} indexed chunks", all_indexed_chunks.len());

        // Save to disk
        storage.save_index(
            &all_indexed_chunks,
            &index_config,
            &config.general.storage_version,
        )?;

        Ok(Self {
            chunk_index,
            storage,
            config: index_config,
            indexed_path,
            storage_version: config.general.storage_version.clone(),
        })
    }

    /// Update the index incrementally based on file changes
    pub async fn update_incremental(&mut self, config: &TurboPropConfig) -> Result<UpdateResult> {
        info!(
            "Performing incremental index update for: {}",
            self.indexed_path.display()
        );

        // Discover current files
        let discovery = FileDiscovery::new(config.file_discovery.clone());
        let current_files = discovery.discover_files(&self.indexed_path)?;

        // Load existing indexed files from storage if not already loaded
        let (existing_indexed_chunks, existing_metadata) = if self.chunk_index.is_empty() {
            if self.storage.index_exists() {
                let (chunks, _config) = self.storage.load_index(&self.storage_version)?;
                let metadata = self.storage.load_metadata()?;
                (chunks, Some(metadata))
            } else {
                (Vec::new(), None)
            }
        } else {
            // Use current in-memory data - no metadata available for timestamp comparison
            (self.chunk_index.get_chunks().to_vec(), None)
        };

        // Build a set of existing files for comparison
        let existing_files: HashSet<PathBuf> = existing_indexed_chunks
            .iter()
            .map(|chunk| chunk.chunk.source_location.file_path.clone())
            .collect();

        let current_file_paths: HashSet<PathBuf> =
            current_files.iter().map(|f| f.path.clone()).collect();

        // Identify files that need to be processed
        let mut files_to_add = Vec::new();
        let mut files_to_update = Vec::new();
        let files_to_remove: Vec<PathBuf> = existing_files
            .difference(&current_file_paths)
            .cloned()
            .collect();

        for file in &current_files {
            if !existing_files.contains(&file.path) {
                files_to_add.push(file.clone());
            } else {
                // Check if file was modified since last index using timestamp comparison
                let needs_update = if let Some(ref metadata) = existing_metadata {
                    // Compare current file timestamp with stored timestamp
                    match metadata.file_timestamps.get(&file.path) {
                        Some(stored_timestamp) => {
                            match file.last_modified.duration_since(*stored_timestamp) {
                                Ok(duration) => {
                                    // File was modified after indexing if duration > 0
                                    duration.as_secs() > 0 || duration.as_nanos() > 0
                                }
                                Err(_) => {
                                    // If timestamp comparison fails, err on the side of updating
                                    warn!(
                                        "Failed to compare timestamps for {}, updating file",
                                        file.path.display()
                                    );
                                    true
                                }
                            }
                        }
                        None => {
                            // No stored timestamp, assume file needs update
                            debug!(
                                "No stored timestamp for {}, updating file",
                                file.path.display()
                            );
                            true
                        }
                    }
                } else {
                    // No metadata available (using in-memory data), always update
                    debug!(
                        "No metadata available for timestamp comparison, updating file: {}",
                        file.path.display()
                    );
                    true
                };

                if needs_update {
                    files_to_update.push(file.clone());
                }
            }
        }

        let mut result = UpdateResult {
            added_files: files_to_add.len(),
            updated_files: files_to_update.len(),
            removed_files: files_to_remove.len(),
            total_chunks_before: existing_indexed_chunks.len(),
            total_chunks_after: 0,
        };

        // If no changes needed, return early
        if files_to_add.is_empty() && files_to_update.is_empty() && files_to_remove.is_empty() {
            info!("No file changes detected, index is up to date");
            result.total_chunks_after = existing_indexed_chunks.len();
            return Ok(result);
        }

        info!(
            "Incremental update: {} files to add, {} to update, {} to remove",
            files_to_add.len(),
            files_to_update.len(),
            files_to_remove.len()
        );

        // Initialize embedding generator
        let mut embedding_generator = EmbeddingGenerator::new(config.embedding.clone()).await?;

        // Start with existing chunks, but filter out removed and updated files
        let mut filtered_chunks: Vec<IndexedChunk> = existing_indexed_chunks
            .into_iter()
            .filter(|chunk| {
                let file_path = &chunk.chunk.source_location.file_path;
                !files_to_remove.contains(file_path)
                    && !files_to_update.iter().any(|f| f.path == *file_path)
            })
            .collect();

        // Process new and updated files
        let chunking_strategy = ChunkingStrategy::new(config.chunking.clone());
        let files_to_process: Vec<&FileMetadata> =
            files_to_add.iter().chain(files_to_update.iter()).collect();

        let mut failed_files = Vec::new();
        let mut processed_files = Vec::new();

        for file in files_to_process {
            info!("Processing: {}", file.path.display());

            // Handle chunking failures gracefully
            let chunks = match chunking_strategy.chunk_file(&file.path) {
                Ok(chunks) => chunks,
                Err(e) => {
                    warn!(
                        "Failed to chunk file {}: {}. Skipping file.",
                        file.path.display(),
                        e
                    );
                    failed_files.push(file.path.clone());
                    continue;
                }
            };

            if chunks.is_empty() {
                debug!("No chunks generated for: {}", file.path.display());
                continue;
            }

            let chunk_texts: Vec<String> =
                chunks.iter().map(|chunk| chunk.content.clone()).collect();

            // Handle embedding generation failures gracefully
            let embeddings = match embedding_generator.embed_batch(&chunk_texts) {
                Ok(embeddings) => embeddings,
                Err(e) => {
                    warn!(
                        "Failed to generate embeddings for {}: {}. Skipping file.",
                        file.path.display(),
                        e
                    );
                    failed_files.push(file.path.clone());
                    continue;
                }
            };

            // Process successful embeddings
            for (chunk, embedding) in chunks.into_iter().zip(embeddings.into_iter()) {
                filtered_chunks.push(IndexedChunk { chunk, embedding });
            }

            processed_files.push(file.path.clone());
            debug!("Successfully processed: {}", file.path.display());
        }

        // Report processing results
        if !failed_files.is_empty() {
            warn!(
                "Failed to process {} files: {:?}. Continuing with {} successfully processed files.", 
                failed_files.len(),
                failed_files,
                processed_files.len()
            );
        }

        info!(
            "Batch processing completed: {} files processed successfully, {} files failed",
            processed_files.len(),
            failed_files.len()
        );

        result.total_chunks_after = filtered_chunks.len();

        // Build new index completely before replacing existing one to avoid race conditions
        let mut new_chunk_index = ChunkIndex::new();
        // Use efficient bulk addition instead of individual clones
        new_chunk_index.add_indexed_chunks(filtered_chunks.clone());

        // Atomically replace the in-memory index only after it's fully built
        self.chunk_index = new_chunk_index;

        // Update configuration if needed
        self.config.model_name = config.embedding.model_name.clone();
        self.config.embedding_dimensions = config.embedding.embedding_dimensions;
        self.config.batch_size = config.embedding.batch_size;

        // Save updated index to disk
        self.storage.save_index(
            &filtered_chunks,
            &self.config,
            &config.general.storage_version,
        )?;

        info!(
            "Incremental update completed: {} chunks before, {} chunks after",
            result.total_chunks_before, result.total_chunks_after
        );

        Ok(result)
    }

    /// Perform a similarity search using the in-memory index
    pub fn similarity_search(
        &self,
        query_embedding: &[f32],
        limit: usize,
    ) -> Vec<(f32, &IndexedChunk)> {
        self.chunk_index.similarity_search(query_embedding, limit)
    }

    /// Search using a text query (requires embedding generation)
    pub async fn search_text(
        &mut self,
        query: &str,
        limit: usize,
        config: &TurboPropConfig,
    ) -> Result<Vec<(f32, &IndexedChunk)>> {
        // Generate embedding for the query
        let mut embedding_generator = EmbeddingGenerator::new(config.embedding.clone()).await?;
        let query_embedding = embedding_generator.embed_single(query)?;

        Ok(self.similarity_search(&query_embedding, limit))
    }

    /// Get the number of chunks in the index
    pub fn len(&self) -> usize {
        self.chunk_index.len()
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.chunk_index.is_empty()
    }

    /// Get all indexed chunks
    pub fn get_chunks(&self) -> &[IndexedChunk] {
        self.chunk_index.get_chunks()
    }

    /// Get the configuration used to build this index
    pub fn config(&self) -> &IndexConfig {
        &self.config
    }

    /// Get the indexed path
    pub fn indexed_path(&self) -> &Path {
        &self.indexed_path
    }

    /// Check if the persistent index exists on disk
    pub fn exists_on_disk(&self) -> bool {
        self.storage.index_exists()
    }

    /// Save current in-memory index to disk
    pub fn save(&self) -> Result<()> {
        let indexed_chunks = self.chunk_index.get_chunks();
        self.storage
            .save_index(indexed_chunks, &self.config, &self.storage_version)
    }

    /// Clear the index (both memory and disk)
    pub fn clear(&mut self) -> Result<()> {
        self.chunk_index = ChunkIndex::new();
        self.storage.clear_index()
    }

    /// Get storage directory path
    pub fn storage_path(&self) -> &Path {
        self.storage.index_dir()
    }
}

/// Result of an incremental update operation
#[derive(Debug, Clone)]
pub struct UpdateResult {
    /// Number of new files added to the index
    pub added_files: usize,
    /// Number of existing files that were updated
    pub updated_files: usize,
    /// Number of files removed from the index
    pub removed_files: usize,
    /// Total number of chunks before the update
    pub total_chunks_before: usize,
    /// Total number of chunks after the update
    pub total_chunks_after: usize,
}

impl UpdateResult {
    /// Check if any changes were made during the update
    pub fn has_changes(&self) -> bool {
        self.added_files > 0 || self.updated_files > 0 || self.removed_files > 0
    }

    /// Calculate the net change in chunk count
    pub fn chunk_delta(&self) -> i64 {
        self.total_chunks_after as i64 - self.total_chunks_before as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TurboPropConfig;
    use std::fs;
    use tempfile::TempDir;
    use tracing::warn;

    fn create_test_file(dir: &Path, name: &str, content: &str) -> PathBuf {
        let file_path = dir.join(name);
        fs::write(&file_path, content).unwrap();
        file_path
    }

    #[tokio::test]
    async fn test_persistent_index_creation() {
        let temp_dir = TempDir::new().unwrap();
        let index = PersistentChunkIndex::new(temp_dir.path()).unwrap();

        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
        assert!(!index.exists_on_disk());
    }

    #[tokio::test]
    async fn test_build_and_load_index() {
        let temp_dir = TempDir::new().unwrap();

        // Create some test files
        create_test_file(
            temp_dir.path(),
            "test1.txt",
            "Hello world this is a test file",
        );
        create_test_file(
            temp_dir.path(),
            "test2.txt",
            "Another test file with different content",
        );

        // Skip this test if offline
        if std::env::var("OFFLINE_TESTS").is_ok() {
            return;
        }

        // Build index
        let config = TurboPropConfig::default();
        let index = PersistentChunkIndex::build(temp_dir.path(), &config).await;

        // This test requires network access to download embedding model
        if index.is_err() {
            warn!("Skipping build test due to network/model requirements");
            return;
        }

        let index = index.unwrap();
        assert!(!index.is_empty());
        assert!(index.exists_on_disk());

        // Test loading
        let loaded_index = PersistentChunkIndex::load(temp_dir.path()).unwrap();
        assert_eq!(loaded_index.len(), index.len());
        assert!(loaded_index.exists_on_disk());
    }

    #[test]
    fn test_load_nonexistent_index() {
        let temp_dir = TempDir::new().unwrap();
        let result = PersistentChunkIndex::load(temp_dir.path());

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No index found"));
    }

    #[test]
    fn test_update_result() {
        let result = UpdateResult {
            added_files: 2,
            updated_files: 1,
            removed_files: 0,
            total_chunks_before: 10,
            total_chunks_after: 15,
        };

        assert!(result.has_changes());
        assert_eq!(result.chunk_delta(), 5);

        let no_change_result = UpdateResult {
            added_files: 0,
            updated_files: 0,
            removed_files: 0,
            total_chunks_before: 10,
            total_chunks_after: 10,
        };

        assert!(!no_change_result.has_changes());
        assert_eq!(no_change_result.chunk_delta(), 0);
    }
}

use crate::types::{ChunkId, DocumentChunk};
use std::collections::HashMap;

/// Simple search index for managing document chunks
#[derive(Debug)]
pub struct SearchIndex {
    chunks: HashMap<ChunkId, DocumentChunk>,
}

impl Default for SearchIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl SearchIndex {
    /// Create a new empty search index
    pub fn new() -> Self {
        Self {
            chunks: HashMap::new(),
        }
    }

    /// Add a chunk to the index
    pub fn add_chunk(&mut self, chunk: DocumentChunk) {
        let chunk_id = ChunkId::from(format!(
            "{}:{}",
            chunk.metadata.file_path.display(),
            chunk.metadata.start_line
        ));
        self.chunks.insert(chunk_id, chunk);
    }

    /// Remove a chunk from the index
    pub fn remove_chunk(&mut self, chunk_id: ChunkId) {
        self.chunks.remove(&chunk_id);
    }

    /// Get the number of chunks in the index
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Get all chunks in the index
    pub fn chunks(&self) -> impl Iterator<Item = (&ChunkId, &DocumentChunk)> {
        self.chunks.iter()
    }

    /// Find all chunk IDs for a given file path
    pub fn find_chunks_by_file_path(&self, file_path: &std::path::Path) -> Vec<ChunkId> {
        self.chunks
            .iter()
            .filter(|(_, chunk)| chunk.metadata.file_path == file_path)
            .map(|(chunk_id, _)| chunk_id.clone())
            .collect()
    }
}
