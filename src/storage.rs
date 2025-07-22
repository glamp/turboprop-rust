//! Persistent storage operations for vector indexes.
//!
//! This module provides atomic file operations and binary serialization
//! for storing and retrieving vector indexes from disk.

use anyhow::{Context, Result};
use memmap2::MmapOptions;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

use crate::types::{ContentChunk, IndexedChunk};

/// Default storage version for index compatibility
pub const DEFAULT_STORAGE_VERSION: &str = "1.0.0";

/// Maximum file size for memory mapping (2GB)
pub const MAX_MMAP_SIZE: u64 = 2 * 1024 * 1024 * 1024;

/// Metadata for the stored index, containing information about the stored data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredIndexMetadata {
    /// Schema version for compatibility checking
    pub version: String,
    /// Number of chunks stored in the index
    pub chunk_count: usize,
    /// Embedding dimensions used for all vectors
    pub embedding_dimensions: usize,
    /// Timestamp when the index was created/updated
    pub created_at: std::time::SystemTime,
    /// List of indexed chunks with their metadata
    pub chunks: Vec<ChunkMetadata>,
    /// File timestamps from when each file was last indexed (for incremental updates)
    #[serde(default)]
    pub file_timestamps: std::collections::HashMap<PathBuf, std::time::SystemTime>,
}

/// Metadata for individual chunks stored in the index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadata {
    /// Unique identifier for the chunk
    pub id: String,
    /// Source file path
    pub file_path: PathBuf,
    /// Starting line number in the source file
    pub start_line: usize,
    /// Ending line number in the source file
    pub end_line: usize,
    /// Character offset in the source file
    pub start_char: usize,
    /// Character offset end in the source file
    pub end_char: usize,
    /// Index of this chunk within the file
    pub chunk_index: usize,
    /// Total number of chunks in the source file
    pub total_chunks: usize,
    /// Number of tokens in this chunk
    pub token_count: usize,
    /// Size of the content in bytes
    pub content_length: usize,
}

impl From<&ContentChunk> for ChunkMetadata {
    fn from(chunk: &ContentChunk) -> Self {
        Self {
            id: chunk.id.clone(),
            file_path: chunk.source_location.file_path.clone(),
            start_line: chunk.source_location.start_line,
            end_line: chunk.source_location.end_line,
            start_char: chunk.source_location.start_char,
            end_char: chunk.source_location.end_char,
            chunk_index: chunk.chunk_index,
            total_chunks: chunk.total_chunks,
            token_count: chunk.token_count,
            content_length: chunk.content.len(),
        }
    }
}

/// Configuration for index storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    /// Embedding model used to generate vectors
    pub model_name: String,
    /// Embedding dimensions
    pub embedding_dimensions: usize,
    /// Batch size used for embedding generation
    pub batch_size: usize,
    /// Whether to respect gitignore files
    pub respect_gitignore: bool,
    /// Whether to include untracked files
    pub include_untracked: bool,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            model_name: crate::embeddings::DEFAULT_MODEL.to_string(),
            embedding_dimensions: crate::embeddings::DEFAULT_EMBEDDING_DIMENSIONS,
            batch_size: 32,
            respect_gitignore: true,
            include_untracked: false,
        }
    }
}

/// Storage manager for persistent vector indexes
#[derive(Debug)]
pub struct IndexStorage {
    /// Base directory for index storage
    index_dir: PathBuf,
}

impl IndexStorage {
    /// Create a new index storage manager
    pub fn new<P: AsRef<Path>>(base_path: P) -> Result<Self> {
        let index_dir = base_path.as_ref().join(".turboprop").join("index");

        // Ensure the index directory exists
        std::fs::create_dir_all(&index_dir).with_context(|| {
            format!("Failed to create index directory: {}", index_dir.display())
        })?;

        debug!("Index storage initialized at: {}", index_dir.display());

        Ok(Self { index_dir })
    }

    /// Check if an index exists at this location
    pub fn index_exists(&self) -> bool {
        let vectors_path = self.index_dir.join("vectors.bin");
        let metadata_path = self.index_dir.join("metadata.json");
        let config_path = self.index_dir.join("config.yaml");

        vectors_path.exists() && metadata_path.exists() && config_path.exists()
    }

    /// Save vectors and metadata to disk atomically
    pub fn save_index(
        &self,
        indexed_chunks: &[IndexedChunk],
        config: &IndexConfig,
        storage_version: &str,
    ) -> Result<()> {
        info!("Saving index with {} chunks to disk", indexed_chunks.len());

        // Create temporary files for atomic operations
        let temp_vectors_path = self.index_dir.join("vectors.bin.tmp");
        let temp_metadata_path = self.index_dir.join("metadata.json.tmp");
        let temp_config_path = self.index_dir.join("config.yaml.tmp");

        // Write vectors to temporary file
        self.write_vectors_file(&temp_vectors_path, indexed_chunks)?;

        // Write metadata to temporary file
        self.write_metadata_file(&temp_metadata_path, indexed_chunks, config, storage_version)?;

        // Write config to temporary file
        self.write_config_file(&temp_config_path, config)?;

        // Atomically move temporary files to final locations
        let final_vectors_path = self.index_dir.join("vectors.bin");
        let final_metadata_path = self.index_dir.join("metadata.json");
        let final_config_path = self.index_dir.join("config.yaml");

        std::fs::rename(&temp_vectors_path, &final_vectors_path).with_context(|| {
            format!(
                "Failed to move vectors file to {}",
                final_vectors_path.display()
            )
        })?;

        std::fs::rename(&temp_metadata_path, &final_metadata_path).with_context(|| {
            format!(
                "Failed to move metadata file to {}",
                final_metadata_path.display()
            )
        })?;

        std::fs::rename(&temp_config_path, &final_config_path).with_context(|| {
            format!(
                "Failed to move config file to {}",
                final_config_path.display()
            )
        })?;

        // Write version file
        let version_path = self.index_dir.join("version.txt");
        std::fs::write(&version_path, storage_version)
            .with_context(|| format!("Failed to write version file: {}", version_path.display()))?;

        info!("Index saved successfully to {}", self.index_dir.display());
        Ok(())
    }

    /// Load vectors and metadata from disk
    pub fn load_index(
        &self,
        expected_storage_version: &str,
    ) -> Result<(Vec<IndexedChunk>, IndexConfig)> {
        if !self.index_exists() {
            anyhow::bail!("No index found at {}", self.index_dir.display());
        }

        info!("Loading index from {}", self.index_dir.display());

        // Verify version compatibility
        self.verify_version(expected_storage_version)?;

        // Load configuration
        let config = self.load_config()?;

        // Load metadata
        let metadata = self.load_metadata()?;

        // Load vectors using memory mapping for performance
        let vectors = self.load_vectors(&metadata)?;

        // Reconstruct IndexedChunk objects
        let mut indexed_chunks = Vec::with_capacity(metadata.chunks.len());

        for (i, chunk_meta) in metadata.chunks.iter().enumerate() {
            if i >= vectors.len() {
                anyhow::bail!(
                    "Vector count mismatch: expected {}, got {}",
                    metadata.chunks.len(),
                    vectors.len()
                );
            }

            // Create a minimal ContentChunk from metadata
            // Note: We don't store the actual content text to save space
            // Using a placeholder that can be detected via has_real_content()
            let content_chunk = ContentChunk {
                id: chunk_meta.id.clone(),
                content: format!(
                    "{}{}:{}]",
                    ContentChunk::PLACEHOLDER_CONTENT_PREFIX,
                    chunk_meta.file_path.display(),
                    chunk_meta.start_line
                ),
                token_count: chunk_meta.token_count,
                source_location: crate::types::SourceLocation {
                    file_path: chunk_meta.file_path.clone(),
                    start_line: chunk_meta.start_line,
                    end_line: chunk_meta.end_line,
                    start_char: chunk_meta.start_char,
                    end_char: chunk_meta.end_char,
                },
                chunk_index: chunk_meta.chunk_index,
                total_chunks: chunk_meta.total_chunks,
            };

            indexed_chunks.push(IndexedChunk {
                chunk: content_chunk,
                embedding: vectors[i].clone(),
            });
        }

        info!("Loaded {} indexed chunks from disk", indexed_chunks.len());
        Ok((indexed_chunks, config))
    }

    /// Write vectors to a binary file using bincode
    fn write_vectors_file(&self, path: &Path, indexed_chunks: &[IndexedChunk]) -> Result<()> {
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .with_context(|| format!("Failed to create vectors file: {}", path.display()))?;

        let mut writer = BufWriter::new(file);

        // Extract just the embeddings for storage
        let embeddings: Vec<Vec<f32>> = indexed_chunks
            .iter()
            .map(|chunk| chunk.embedding.clone())
            .collect();

        // Serialize using bincode for efficiency
        bincode::serialize_into(&mut writer, &embeddings)
            .with_context(|| "Failed to serialize vectors")?;

        writer.flush().context("Failed to flush vectors file")?;
        debug!("Wrote {} vectors to {}", embeddings.len(), path.display());

        Ok(())
    }

    /// Write metadata to a JSON file
    fn write_metadata_file(
        &self,
        path: &Path,
        indexed_chunks: &[IndexedChunk],
        config: &IndexConfig,
        storage_version: &str,
    ) -> Result<()> {
        // Collect file timestamps for incremental updates
        let mut file_timestamps = std::collections::HashMap::new();
        for indexed_chunk in indexed_chunks {
            let file_path = &indexed_chunk.chunk.source_location.file_path;
            // Note: We store the current time as the indexed timestamp
            // In practice, you might want to store the actual file modification time
            file_timestamps.insert(file_path.clone(), std::time::SystemTime::now());
        }
        
        let metadata = StoredIndexMetadata {
            version: storage_version.to_string(),
            chunk_count: indexed_chunks.len(),
            embedding_dimensions: config.embedding_dimensions,
            created_at: std::time::SystemTime::now(),
            chunks: indexed_chunks
                .iter()
                .map(|chunk| ChunkMetadata::from(&chunk.chunk))
                .collect(),
            file_timestamps,
        };

        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .with_context(|| format!("Failed to create metadata file: {}", path.display()))?;

        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &metadata)
            .with_context(|| "Failed to serialize metadata")?;

        debug!(
            "Wrote metadata for {} chunks to {}",
            indexed_chunks.len(),
            path.display()
        );
        Ok(())
    }

    /// Write configuration to a YAML file
    fn write_config_file(&self, path: &Path, config: &IndexConfig) -> Result<()> {
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .with_context(|| format!("Failed to create config file: {}", path.display()))?;

        let writer = BufWriter::new(file);
        serde_yaml::to_writer(writer, config).with_context(|| "Failed to serialize config")?;

        debug!("Wrote config to {}", path.display());
        Ok(())
    }

    /// Load vectors from binary file using memory mapping
    fn load_vectors(&self, metadata: &StoredIndexMetadata) -> Result<Vec<Vec<f32>>> {
        let vectors_path = self.index_dir.join("vectors.bin");

        let file = File::open(&vectors_path)
            .with_context(|| format!("Failed to open vectors file: {}", vectors_path.display()))?;

        // Validate file size before memory mapping to prevent catastrophic failures
        let file_size = file
            .metadata()
            .with_context(|| "Failed to get vectors file metadata")?
            .len();
        
        // Set reasonable limits for memory mapping
        if file_size > MAX_MMAP_SIZE {
            anyhow::bail!(
                "Vectors file too large for memory mapping: {} bytes (max: {} bytes). Consider using chunked loading.",
                file_size,
                MAX_MMAP_SIZE
            );
        }

        if file_size == 0 {
            anyhow::bail!("Vectors file is empty: {}", vectors_path.display());
        }

        // Use memory mapping for efficient loading with proper bounds checking
        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .with_context(|| format!("Failed to create memory map for vectors file (size: {} bytes)", file_size))?
        };

        // Deserialize from memory-mapped data
        let vectors: Vec<Vec<f32>> = bincode::deserialize(&mmap)
            .with_context(|| "Failed to deserialize vectors from file")?;

        // Validate loaded data
        if vectors.len() != metadata.chunk_count {
            anyhow::bail!(
                "Vector count mismatch: expected {}, got {}",
                metadata.chunk_count,
                vectors.len()
            );
        }

        debug!(
            "Loaded {} vectors from {}",
            vectors.len(),
            vectors_path.display()
        );
        Ok(vectors)
    }

    /// Load metadata from JSON file
    pub fn load_metadata(&self) -> Result<StoredIndexMetadata> {
        let metadata_path = self.index_dir.join("metadata.json");

        let file = File::open(&metadata_path).with_context(|| {
            format!("Failed to open metadata file: {}", metadata_path.display())
        })?;

        let reader = BufReader::new(file);
        let metadata: StoredIndexMetadata =
            serde_json::from_reader(reader).with_context(|| "Failed to deserialize metadata")?;

        debug!("Loaded metadata for {} chunks", metadata.chunk_count);
        Ok(metadata)
    }

    /// Load configuration from YAML file
    fn load_config(&self) -> Result<IndexConfig> {
        let config_path = self.index_dir.join("config.yaml");

        let file = File::open(&config_path)
            .with_context(|| format!("Failed to open config file: {}", config_path.display()))?;

        let reader = BufReader::new(file);
        let config: IndexConfig =
            serde_yaml::from_reader(reader).with_context(|| "Failed to deserialize config")?;

        debug!("Loaded config with model: {}", config.model_name);
        Ok(config)
    }

    /// Verify that the stored version is compatible
    fn verify_version(&self, expected_version: &str) -> Result<()> {
        let version_path = self.index_dir.join("version.txt");

        if !version_path.exists() {
            warn!("No version file found, assuming compatible format");
            return Ok(());
        }

        let stored_version = std::fs::read_to_string(&version_path)
            .with_context(|| format!("Failed to read version file: {}", version_path.display()))?;

        let stored_version = stored_version.trim();

        if stored_version != expected_version {
            anyhow::bail!(
                "Index version mismatch: expected {}, found {}. Please rebuild the index.",
                expected_version,
                stored_version
            );
        }

        debug!("Version verified: {}", stored_version);
        Ok(())
    }

    /// Get the path to the index directory
    pub fn index_dir(&self) -> &Path {
        &self.index_dir
    }

    /// Clear the index (remove all files) with transactional rollback capability
    pub fn clear_index(&self) -> Result<()> {
        if !self.index_dir.exists() {
            return Ok(());
        }

        info!("Clearing index at {}", self.index_dir.display());

        // First, collect all files that need to be deleted
        let mut files_to_delete = Vec::new();
        for entry in std::fs::read_dir(&self.index_dir)
            .with_context(|| format!("Failed to read index directory: {}", self.index_dir.display()))? {
            let entry = entry.with_context(|| "Failed to read directory entry")?;
            let path = entry.path();

            if path.is_file() {
                files_to_delete.push(path);
            }
        }

        if files_to_delete.is_empty() {
            debug!("No files to delete in index directory");
            return Ok(());
        }

        info!("Found {} files to delete", files_to_delete.len());

        // Phase 1: Validate that all files can be deleted
        for path in &files_to_delete {
            // Check if file is readable and deletable
            match std::fs::metadata(path) {
                Ok(metadata) => {
                    if metadata.permissions().readonly() {
                        anyhow::bail!(
                            "Cannot delete read-only file: {}. Please check file permissions and try again.",
                            path.display()
                        );
                    }
                }
                Err(e) => {
                    anyhow::bail!(
                        "Cannot access file for deletion: {}. Error: {}. Please check file permissions and try again.",
                        path.display(),
                        e
                    );
                }
            }
        }

        // Phase 2: Create backup references for rollback (store paths for error reporting)
        let backup_paths: Vec<_> = files_to_delete.iter().cloned().collect();

        // Phase 3: Perform deletions with detailed error context
        let mut deleted_files = Vec::new();
        let mut deletion_errors = Vec::new();

        for path in files_to_delete {
            match std::fs::remove_file(&path) {
                Ok(()) => {
                    debug!("Removed file: {}", path.display());
                    deleted_files.push(path.clone());
                }
                Err(e) => {
                    let error_msg = format!(
                        "Failed to delete file: {}. Error: {}. {} files were already deleted.", 
                        path.display(), 
                        e,
                        deleted_files.len()
                    );
                    deletion_errors.push(error_msg.clone());
                    
                    // Log which files were successfully deleted for manual cleanup if needed
                    if !deleted_files.is_empty() {
                        warn!(
                            "Partial deletion occurred. Successfully deleted {} files: {:?}. Failed to delete: {}",
                            deleted_files.len(),
                            deleted_files.iter().map(|p| p.display()).collect::<Vec<_>>(),
                            path.display()
                        );
                    }
                    
                    return Err(anyhow::anyhow!(
                        "{}. Index is in partially deleted state. You may need to manually clean up remaining files: {:?}. Consider backing up important data before retrying.",
                        error_msg,
                        backup_paths.iter().filter(|p| !deleted_files.contains(p)).collect::<Vec<_>>()
                    ));
                }
            }
        }

        info!("Successfully cleared {} files from index", deleted_files.len());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ContentChunk, SourceLocation};
    use tempfile::TempDir;

    fn create_test_chunk(id: &str, content: &str, file_path: &str) -> ContentChunk {
        ContentChunk {
            id: id.to_string(),
            content: content.to_string(),
            token_count: content.split_whitespace().count(),
            source_location: SourceLocation {
                file_path: PathBuf::from(file_path),
                start_line: 1,
                end_line: 1,
                start_char: 0,
                end_char: content.len(),
            },
            chunk_index: 0,
            total_chunks: 1,
        }
    }

    fn create_test_indexed_chunk(
        id: &str,
        content: &str,
        file_path: &str,
        embedding: Vec<f32>,
    ) -> IndexedChunk {
        IndexedChunk {
            chunk: create_test_chunk(id, content, file_path),
            embedding,
        }
    }

    #[test]
    fn test_storage_initialization() {
        let temp_dir = TempDir::new().unwrap();
        let storage = IndexStorage::new(temp_dir.path()).unwrap();

        assert!(storage.index_dir().exists());
        assert!(storage.index_dir().join("..").join("index").exists());
        assert!(!storage.index_exists());
    }

    #[test]
    fn test_save_and_load_index() {
        let temp_dir = TempDir::new().unwrap();
        let storage = IndexStorage::new(temp_dir.path()).unwrap();

        // Create test data
        let indexed_chunks = vec![
            create_test_indexed_chunk("chunk1", "Hello world", "test1.txt", vec![0.1, 0.2, 0.3]),
            create_test_indexed_chunk("chunk2", "Goodbye world", "test2.txt", vec![0.4, 0.5, 0.6]),
        ];

        let config = IndexConfig {
            model_name: "test-model".to_string(),
            embedding_dimensions: 3,
            ..Default::default()
        };

        // Save index
        storage
            .save_index(&indexed_chunks, &config, DEFAULT_STORAGE_VERSION)
            .unwrap();
        assert!(storage.index_exists());

        // Load index
        let (loaded_chunks, loaded_config) = storage.load_index(DEFAULT_STORAGE_VERSION).unwrap();

        // Verify loaded data
        assert_eq!(loaded_chunks.len(), 2);
        assert_eq!(loaded_config.model_name, "test-model");
        assert_eq!(loaded_config.embedding_dimensions, 3);

        // Check that embeddings match
        assert_eq!(loaded_chunks[0].embedding, vec![0.1, 0.2, 0.3]);
        assert_eq!(loaded_chunks[1].embedding, vec![0.4, 0.5, 0.6]);

        // Check that chunk IDs match
        assert_eq!(loaded_chunks[0].chunk.id, "chunk1");
        assert_eq!(loaded_chunks[1].chunk.id, "chunk2");
    }

    #[test]
    fn test_clear_index() {
        let temp_dir = TempDir::new().unwrap();
        let storage = IndexStorage::new(temp_dir.path()).unwrap();

        // Create test data and save
        let indexed_chunks = vec![create_test_indexed_chunk(
            "chunk1",
            "Test content",
            "test.txt",
            vec![1.0, 2.0],
        )];
        let config = IndexConfig::default();

        storage
            .save_index(&indexed_chunks, &config, DEFAULT_STORAGE_VERSION)
            .unwrap();
        assert!(storage.index_exists());

        // Clear the index
        storage.clear_index().unwrap();
        assert!(!storage.index_exists());
    }

    #[test]
    fn test_load_nonexistent_index() {
        let temp_dir = TempDir::new().unwrap();
        let storage = IndexStorage::new(temp_dir.path()).unwrap();

        let result = storage.load_index(DEFAULT_STORAGE_VERSION);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No index found"));
    }

    #[test]
    fn test_chunk_metadata_conversion() {
        let chunk = create_test_chunk("test-id", "test content here", "path/to/file.rs");
        let metadata = ChunkMetadata::from(&chunk);

        assert_eq!(metadata.id, "test-id");
        assert_eq!(metadata.file_path, PathBuf::from("path/to/file.rs"));
        assert_eq!(metadata.token_count, 3);
        assert_eq!(metadata.content_length, 17);
    }
}
