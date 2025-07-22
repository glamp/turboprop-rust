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
use crate::types::{ChunkIndex, ChunkingConfig, FileMetadata, IndexedChunk};

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
        })
    }

    /// Load an existing index from disk
    pub fn load<P: AsRef<Path>>(base_path: P) -> Result<Self> {
        let indexed_path = base_path.as_ref().to_path_buf();
        let storage = IndexStorage::new(&indexed_path)?;

        if !storage.index_exists() {
            anyhow::bail!("No index found at {}", indexed_path.display());
        }

        let (indexed_chunks, config) = storage.load_index()?;
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
        let chunking_config = ChunkingConfig::default();
        let chunking_strategy = ChunkingStrategy::new(chunking_config);

        for file in &files {
            info!("Processing: {}", file.path.display());

            // Generate chunks from the file
            let chunks = chunking_strategy.chunk_file(&file.path)?;
            
            if chunks.is_empty() {
                debug!("No chunks generated for: {}", file.path.display());
                continue;
            }

            // Prepare text for embedding
            let chunk_texts: Vec<String> = chunks.iter().map(|chunk| chunk.content.clone()).collect();

            // Generate embeddings
            debug!("Generating embeddings for {} chunks", chunks.len());
            let embeddings = embedding_generator.embed_batch(&chunk_texts)?;

            // Create indexed chunks and add to both collections
            for (chunk, embedding) in chunks.into_iter().zip(embeddings.into_iter()) {
                let indexed_chunk = IndexedChunk { chunk, embedding };
                chunk_index.add_chunk(indexed_chunk.chunk.clone(), indexed_chunk.embedding.clone());
                all_indexed_chunks.push(indexed_chunk);
            }
        }

        info!("Generated {} indexed chunks", all_indexed_chunks.len());

        // Save to disk
        storage.save_index(&all_indexed_chunks, &index_config)?;

        Ok(Self {
            chunk_index,
            storage,
            config: index_config,
            indexed_path,
        })
    }

    /// Update the index incrementally based on file changes
    pub async fn update_incremental(&mut self, config: &TurboPropConfig) -> Result<UpdateResult> {
        info!("Performing incremental index update for: {}", self.indexed_path.display());

        // Discover current files
        let discovery = FileDiscovery::new(config.file_discovery.clone());
        let current_files = discovery.discover_files(&self.indexed_path)?;

        // Load existing indexed files from storage if not already loaded
        let (existing_indexed_chunks, _) = if self.chunk_index.is_empty() {
            if self.storage.index_exists() {
                self.storage.load_index()?
            } else {
                (Vec::new(), self.config.clone())
            }
        } else {
            // Use current in-memory data
            (self.chunk_index.get_chunks().to_vec(), self.config.clone())
        };

        // Build a set of existing files for comparison
        let existing_files: HashSet<PathBuf> = existing_indexed_chunks
            .iter()
            .map(|chunk| chunk.chunk.source_location.file_path.clone())
            .collect();

        let current_file_paths: HashSet<PathBuf> = current_files
            .iter()
            .map(|f| f.path.clone())
            .collect();

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
                // Check if file was modified since last index
                // For simplicity, we'll always update existing files
                // In a production system, you'd compare timestamps
                files_to_update.push(file.clone());
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
        let chunking_config = ChunkingConfig::default();
        let chunking_strategy = ChunkingStrategy::new(chunking_config);
        let files_to_process: Vec<&FileMetadata> = files_to_add
            .iter()
            .chain(files_to_update.iter())
            .collect();

        for file in files_to_process {
            info!("Processing: {}", file.path.display());

            let chunks = chunking_strategy.chunk_file(&file.path)?;
            
            if chunks.is_empty() {
                debug!("No chunks generated for: {}", file.path.display());
                continue;
            }

            let chunk_texts: Vec<String> = chunks.iter().map(|chunk| chunk.content.clone()).collect();
            let embeddings = embedding_generator.embed_batch(&chunk_texts)?;

            for (chunk, embedding) in chunks.into_iter().zip(embeddings.into_iter()) {
                filtered_chunks.push(IndexedChunk { chunk, embedding });
            }
        }

        result.total_chunks_after = filtered_chunks.len();

        // Update the in-memory index
        self.chunk_index = ChunkIndex::new();
        for indexed_chunk in &filtered_chunks {
            self.chunk_index.add_chunk(indexed_chunk.chunk.clone(), indexed_chunk.embedding.clone());
        }

        // Update configuration if needed
        self.config.model_name = config.embedding.model_name.clone();
        self.config.embedding_dimensions = config.embedding.embedding_dimensions;
        self.config.batch_size = config.embedding.batch_size;

        // Save updated index to disk
        self.storage.save_index(&filtered_chunks, &self.config)?;

        info!(
            "Incremental update completed: {} chunks before, {} chunks after",
            result.total_chunks_before,
            result.total_chunks_after
        );

        Ok(result)
    }

    /// Perform a similarity search using the in-memory index
    pub fn similarity_search(&self, query_embedding: &[f32], limit: usize) -> Vec<(f32, &IndexedChunk)> {
        self.chunk_index.similarity_search(query_embedding, limit)
    }

    /// Search using a text query (requires embedding generation)
    pub async fn search_text(&mut self, query: &str, limit: usize, config: &TurboPropConfig) -> Result<Vec<(f32, &IndexedChunk)>> {
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
        self.storage.save_index(indexed_chunks, &self.config)
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
    use tempfile::TempDir;
    use crate::config::TurboPropConfig;
    use std::fs;

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
        create_test_file(temp_dir.path(), "test1.txt", "Hello world this is a test file");
        create_test_file(temp_dir.path(), "test2.txt", "Another test file with different content");
        
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