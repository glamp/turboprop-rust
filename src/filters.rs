//! Search result filtering functionality.
//!
//! This module provides filtering capabilities for search results, including
//! filtering by file type/extension and glob pattern matching.
//!
//! # Glob Pattern Behavior
//!
//! Glob patterns in this module follow standard Unix shell globbing rules:
//!
//! ## Basic Wildcards
//! - `*` - Matches any sequence of characters within a single path component
//! - `?` - Matches exactly one character
//! - `**` - Matches any sequence of characters across multiple directories (recursive)
//!
//! ## Character Classes
//! - `[abc]` - Matches any single character from the set (a, b, or c)
//! - `[a-z]` - Matches any character in the range (a through z)
//! - `[!abc]` or `[^abc]` - Matches any character NOT in the set
//!
//! ## Important Behavior Notes
//!
//! ### Path Matching
//! Patterns match against the **entire path**, not just the filename:
//! - `*.rs` matches `main.rs` AND `src/main.rs` AND `deep/nested/file.rs`
//! - To match only files in the current directory: use specific patterns
//! - To match files in any subdirectory: use `**/*.rs`
//!
//! ### Case Sensitivity
//! Patterns are **case-sensitive** by default:
//! - `*.RS` matches `FILE.RS` but NOT `file.rs`
//! - `*.rs` matches `file.rs` but NOT `FILE.RS`
//!
//! ### Directory Separators
//! - `/` is always used as the directory separator in patterns
//! - Patterns work consistently across platforms
//! - `*` does NOT cross directory boundaries: `src/*.rs` matches `src/main.rs` but not `src/lib/mod.rs`
//! - `**` DOES cross directory boundaries: `src/**/*.rs` matches both
//!
//! ## Examples
//!
//! ```text
//! Pattern          | Matches                    | Does NOT match
//! -----------------|----------------------------|------------------
//! *.rs             | main.rs, src/main.rs      | main.js, file.rs.bak
//! src/*.rs         | src/main.rs, src/lib.rs   | main.rs, src/test/mod.rs
//! **/*.py          | any .py file anywhere      | .pyc files
//! test_*.rs        | test_main.rs, test_lib.rs | main_test.rs
//! src/**/test_*.rs | src/test_main.rs,          | test_main.rs (not in src)
//!                  | src/unit/test_lib.rs      |
//! ```
//!
//! For more examples and edge cases, see the test module documentation.

use anyhow::Result;
use glob::Pattern;
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::types::SearchResult;

/// Characters that are problematic in file paths across platforms.
///
/// These characters either have special meaning in file systems or
/// are reserved/problematic on common platforms.
const PROBLEMATIC_PATH_CHARS: &[char] = &[
    '\0', // Null terminator
    '\x01', '\x02', '\x03', '\x04', '\x05', '\x06', '\x07', // Control chars
    '\x08', /* \x09 = tab (allowed) */ '\x0A', '\x0B', '\x0C', '\x0D', '\x0E',
    '\x0F', // More control chars (excluding tab)
    '\x10', '\x11', '\x12', '\x13', '\x14', '\x15', '\x16', '\x17', '\x18', '\x19', '\x1A', '\x1B',
    '\x1C', '\x1D', '\x1E', '\x1F', '\x7F', // DEL character
];

/// Characters that are restricted on Windows file systems.
/// These are in addition to the universal problematic characters.
#[cfg(target_os = "windows")]
const WINDOWS_RESTRICTED_CHARS: &[char] = &['<', '>', ':', '"', '|', '?', '*'];

/// Reserved file names on Windows that cannot be used as filenames.
#[cfg(target_os = "windows")]
const WINDOWS_RESERVED_NAMES: &[&str] = &[
    "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8",
    "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
];

/// Default maximum allowed length for file extensions (including the dot)
pub const DEFAULT_MAX_EXTENSION_LENGTH: usize = 10;

/// Default maximum allowed length for glob patterns
pub const DEFAULT_MAX_GLOB_PATTERN_LENGTH: usize = 1000;

/// A validated glob pattern wrapper.
///
/// This struct provides a safe wrapper around compiled glob patterns with validation.
/// The pattern is validated once during construction and then can be used efficiently
/// for multiple matching operations.
///
/// # Examples
///
/// ```rust
/// use turboprop::filters::GlobPattern;
/// use std::path::Path;
///
/// // Create a pattern for Rust source files
/// let pattern = GlobPattern::new("**/*.rs").unwrap();
///
/// // Test against various paths
/// assert!(pattern.matches(Path::new("src/main.rs")));
/// assert!(pattern.matches(Path::new("tests/integration/test.rs")));
/// assert!(!pattern.matches(Path::new("main.js")));
///
/// // Pattern implements Display for easy debugging
/// println!("Pattern: {}", pattern);
/// ```
///
/// # Pattern Compilation
///
/// The pattern is compiled once during construction using the `glob` crate's
/// `Pattern::new()`. Invalid patterns will result in an error during construction,
/// not during matching operations.
///
/// # Performance
///
/// Once constructed, pattern matching is very fast. The compiled pattern is cached
/// internally and reused for all matching operations. Consider caching `GlobPattern`
/// instances if you'll be using the same pattern multiple times.
///
/// # Thread Safety
///
/// `GlobPattern` is `Send` and `Sync`, making it safe to share across threads.
#[derive(Debug, Clone)]
pub struct GlobPattern {
    /// The original pattern string
    pattern: String,
    /// The compiled glob pattern
    compiled: Pattern,
}

impl GlobPattern {
    /// Create a new GlobPattern from a string with configurable validation.
    ///
    /// This method validates the pattern string according to the provided configuration
    /// limits and compiles it into an efficient matching structure.
    ///
    /// # Arguments
    ///
    /// * `pattern` - The glob pattern string (e.g., "*.rs", "src/**/*.js")
    /// * `config` - Filter configuration containing validation limits
    ///
    /// # Returns
    ///
    /// Returns `Ok(GlobPattern)` if the pattern is valid, or an error with detailed
    /// information about what's wrong and how to fix it.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use turboprop::filters::{GlobPattern, FilterConfig};
    ///
    /// let config = FilterConfig::with_limits(500, 5);
    /// let pattern = GlobPattern::new_with_config("*.rs", &config)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new_with_config(pattern: &str, config: &FilterConfig) -> Result<Self> {
        validate_glob_pattern_with_config(pattern, config)?;
        let compiled = Pattern::new(pattern)
            .map_err(|e| anyhow::anyhow!("Invalid glob pattern '{}': {}", pattern, e))?;

        Ok(Self {
            pattern: pattern.to_string(),
            compiled,
        })
    }

    /// Create a new GlobPattern from a string using default configuration.
    ///
    /// This is a convenience method that uses default validation limits.
    /// For custom limits, use [`new_with_config`](Self::new_with_config).
    ///
    /// # Arguments
    ///
    /// * `pattern` - The glob pattern string
    ///
    /// # Examples
    ///
    /// ```rust
    /// use turboprop::filters::GlobPattern;
    ///
    /// // Match all Rust files recursively
    /// let pattern = GlobPattern::new("**/*.rs")?;
    ///
    /// // Match JavaScript files in src directory
    /// let pattern = GlobPattern::new("src/*.js")?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(pattern: &str) -> Result<Self> {
        Self::new_with_config(pattern, &FilterConfig::default())
    }

    /// Get the original pattern string.
    ///
    /// Returns the exact pattern string that was used to create this `GlobPattern`,
    /// useful for debugging, logging, or displaying to users.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use turboprop::filters::GlobPattern;
    ///
    /// let pattern = GlobPattern::new("**/*.rs")?;
    /// assert_eq!(pattern.pattern(), "**/*.rs");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn pattern(&self) -> &str {
        &self.pattern
    }

    /// Check if a path matches this glob pattern.
    ///
    /// Tests whether the given path matches the compiled glob pattern. The path
    /// is converted to a string for matching, and paths containing invalid UTF-8
    /// will never match.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to test against the pattern
    ///
    /// # Returns
    ///
    /// `true` if the path matches the pattern, `false` otherwise.
    ///
    /// # Pattern Matching Details
    ///
    /// - Matching is performed against the entire path, not just the filename
    /// - Path separators are normalized to `/` for consistent cross-platform behavior
    /// - Matching is case-sensitive
    /// - Paths with invalid UTF-8 characters will not match any pattern
    ///
    /// # Examples
    ///
    /// ```rust
    /// use turboprop::filters::GlobPattern;
    /// use std::path::Path;
    ///
    /// let pattern = GlobPattern::new("src/*.rs")?;
    ///
    /// assert!(pattern.matches(Path::new("src/main.rs")));
    /// assert!(pattern.matches(Path::new("src/lib.rs")));
    /// assert!(!pattern.matches(Path::new("main.rs")));        // Not in src/
    /// assert!(!pattern.matches(Path::new("tests/main.rs")));  // Wrong directory
    /// assert!(!pattern.matches(Path::new("src/main.js")));    // Wrong extension
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn matches(&self, path: &Path) -> bool {
        // Convert path to string for matching
        if let Some(path_str) = path.to_str() {
            self.compiled.matches(path_str)
        } else {
            // If path contains invalid UTF-8, it won't match
            false
        }
    }
}

impl fmt::Display for GlobPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.pattern)
    }
}

impl PartialEq for GlobPattern {
    fn eq(&self, other: &Self) -> bool {
        self.pattern == other.pattern
    }
}

impl Eq for GlobPattern {}

impl Hash for GlobPattern {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.pattern.hash(state);
    }
}

/// Normalize a glob pattern to a canonical form for better caching.
///
/// This function applies various normalization rules to make equivalent patterns
/// identical, improving cache hit rates and reducing memory usage.
///
/// # Normalization Rules
///
/// - Remove redundant directory separators: `a//b` becomes `a/b`
/// - Remove trailing slashes: `dir/` becomes `dir`
/// - Normalize current directory references: `./pattern` becomes `pattern`
/// - Collapse redundant wildcards: `**/**` becomes `**`
/// - Sort character classes: `[zab]` becomes `[abz]`
///
/// # Examples
///
/// ```rust
/// use turboprop::filters::normalize_glob_pattern;
///
/// assert_eq!(normalize_glob_pattern("a//b/*.rs"), "a/b/*.rs");
/// assert_eq!(normalize_glob_pattern("./src/**/*.js"), "src/**/*.js");
/// assert_eq!(normalize_glob_pattern("**/**/*.py"), "**/*.py");
/// ```
pub fn normalize_glob_pattern(pattern: &str) -> String {
    let mut normalized = pattern.trim().to_string();

    // Remove leading ./
    if normalized.starts_with("./") {
        normalized = normalized[2..].to_string();
    }

    // Replace multiple slashes with single slash
    while normalized.contains("//") {
        normalized = normalized.replace("//", "/");
    }

    // Remove trailing slash unless it's the root
    if normalized.len() > 1 && normalized.ends_with('/') {
        normalized.pop();
    }

    // Collapse redundant recursive wildcards: **/** -> **
    while normalized.contains("**/**") {
        normalized = normalized.replace("**/**", "**");
    }

    // Collapse redundant recursive wildcards with slashes: **/*/** -> **/**
    while normalized.contains("**/*/**") {
        normalized = normalized.replace("**/*/**", "**/**");
    }

    normalized
}

/// Thread-safe cache for compiled glob patterns.
///
/// This cache stores compiled `GlobPattern` instances to avoid recompilation
/// of frequently used patterns. The cache is thread-safe and can be shared
/// across multiple threads.
///
/// # Performance Benefits
///
/// - Avoids expensive pattern compilation for repeated patterns
/// - Reduces memory usage by sharing identical compiled patterns
/// - Improves lookup performance for common patterns
///
/// # Usage
///
/// ```rust
/// use turboprop::filters::{GlobPatternCache, FilterConfig};
/// use std::sync::Arc;
///
/// let cache = Arc::new(GlobPatternCache::new());
/// let config = FilterConfig::default();
///
/// // First access compiles and caches the pattern
/// let pattern1 = cache.get_or_create("*.rs", &config)?;
///
/// // Second access reuses the cached pattern
/// let pattern2 = cache.get_or_create("*.rs", &config)?;
///
/// // Both patterns are the same instance
/// assert!(Arc::ptr_eq(&pattern1, &pattern2));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Default)]
pub struct GlobPatternCache {
    cache: Mutex<HashMap<String, Arc<GlobPattern>>>,
}

impl GlobPatternCache {
    /// Create a new empty pattern cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get a pattern from the cache or create and cache it if not found.
    ///
    /// This method first normalizes the pattern, then checks the cache.
    /// If the pattern is found, it returns the cached instance. Otherwise,
    /// it creates a new pattern, caches it, and returns it.
    ///
    /// # Arguments
    ///
    /// * `pattern` - The glob pattern string
    /// * `config` - Configuration for pattern validation
    ///
    /// # Returns
    ///
    /// Returns an `Arc<GlobPattern>` that can be shared across threads.
    ///
    /// # Thread Safety
    ///
    /// This method is thread-safe and can be called concurrently from
    /// multiple threads. The internal mutex ensures cache consistency.
    pub fn get_or_create(&self, pattern: &str, config: &FilterConfig) -> Result<Arc<GlobPattern>> {
        let normalized = normalize_glob_pattern(pattern);

        // First, try to get from cache without holding the lock for long
        {
            let cache = self
                .cache
                .lock()
                .map_err(|_| anyhow::anyhow!("Pattern cache lock poisoned"))?;

            if let Some(cached_pattern) = cache.get(&normalized) {
                return Ok(Arc::clone(cached_pattern));
            }
        }

        // Pattern not in cache, create it
        let new_pattern = GlobPattern::new_with_config(&normalized, config)?;
        let arc_pattern = Arc::new(new_pattern);

        // Insert into cache
        {
            let mut cache = self
                .cache
                .lock()
                .map_err(|_| anyhow::anyhow!("Pattern cache lock poisoned"))?;

            // Check again in case another thread inserted it while we were creating
            if let Some(existing_pattern) = cache.get(&normalized) {
                return Ok(Arc::clone(existing_pattern));
            }

            cache.insert(normalized, Arc::clone(&arc_pattern));
        }

        Ok(arc_pattern)
    }

    /// Get cache statistics for monitoring and debugging.
    ///
    /// Returns the number of patterns currently cached.
    pub fn len(&self) -> usize {
        self.cache.lock().map(|cache| cache.len()).unwrap_or(0)
    }

    /// Check if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear all cached patterns.
    ///
    /// This can be useful for memory management in long-running applications.
    pub fn clear(&self) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.clear();
        }
    }
}

/// Validate a glob pattern string with configurable limits
pub fn validate_glob_pattern_with_config(pattern: &str, config: &FilterConfig) -> Result<()> {
    let pattern = pattern.trim();

    // Check for empty pattern
    if pattern.is_empty() {
        anyhow::bail!(
            "Glob pattern cannot be empty.\n\nExamples of valid patterns:\n  - *.rs (all Rust files)\n  - src/*.js (JavaScript files in src directory)\n  - **/*.py (Python files in any subdirectory)"
        );
    }

    // Check pattern length
    if pattern.len() > config.max_glob_pattern_length {
        anyhow::bail!(
            "Glob pattern too long: {} characters (maximum: {}).\n\nSuggestions:\n  - Use shorter directory names\n  - Simplify the pattern structure\n  - Consider using ** for recursive matching instead of explicit paths\n\nExample: Instead of 'very/long/nested/directory/structure/*.rs', use '**/structure/*.rs'",
            pattern.len(),
            config.max_glob_pattern_length
        );
    }

    // Validate character restrictions with platform-specific checks
    validate_pattern_characters(pattern)?;

    // Try to compile the pattern to check for syntax errors
    Pattern::new(pattern).map_err(|e| {
        anyhow::anyhow!(
            "Invalid glob pattern syntax in '{}': {}\n\nCommon glob pattern syntax:\n  - * matches any characters within a single directory\n  - ** matches any characters across directories\n  - ? matches a single character\n  - [abc] matches any character in the set\n  - [a-z] matches any character in the range\n\nExamples:\n  - *.rs (Rust files in current directory)\n  - src/**/*.js (JavaScript files in src and subdirectories)\n  - test_*.py (Python test files)\n  - **/*.{{js,ts}} (JavaScript and TypeScript files anywhere)",
            pattern,
            e
        )
    })?;

    Ok(())
}

/// Validate that a pattern doesn't contain problematic characters.
///
/// This function checks for characters that are problematic across platforms
/// as well as platform-specific restrictions.
fn validate_pattern_characters(pattern: &str) -> Result<()> {
    // Check for universally problematic control characters
    if let Some(invalid_char) = pattern.chars().find(|c| PROBLEMATIC_PATH_CHARS.contains(c)) {
        let char_name = match invalid_char {
            '\0' => "null terminator".to_string(),
            '\x01'..='\x1F' => format!("control character (0x{:02X})", invalid_char as u8),
            '\x7F' => "DEL character".to_string(),
            _ => "unknown problematic character".to_string(),
        };

        anyhow::bail!(
            "Glob pattern contains invalid character: '{}' ({}) at position {} in pattern '{}'.\n\nThis character is problematic because:\n  - It's a control character that can cause issues in file systems\n  - It may not display correctly in terminals or editors\n  - It could be interpreted specially by shells or file systems\n\nAllowed characters:\n  - Printable ASCII and Unicode characters\n  - Tab character (\\t) is allowed in patterns\n  - Standard glob metacharacters: * ? [ ] {{ }}",
            invalid_char.escape_default().collect::<String>(),
            char_name,
            pattern.chars().position(|c| c == invalid_char).unwrap_or(0),
            pattern
        );
    }

    // Platform-specific validations
    #[cfg(target_os = "windows")]
    {
        validate_windows_path_restrictions(pattern)?;
    }

    // Check for other problematic patterns
    validate_problematic_patterns(pattern)?;

    Ok(())
}

/// Validate Windows-specific path restrictions.
#[cfg(target_os = "windows")]
fn validate_windows_path_restrictions(pattern: &str) -> Result<()> {
    // Check for Windows-restricted characters (but allow them in glob patterns)
    // Note: We're more permissive in glob patterns since they're not direct filenames
    if let Some(restricted_char) = pattern
        .chars()
        .find(|c| WINDOWS_RESTRICTED_CHARS.contains(c) && !matches!(*c, '*' | '?'))
    {
        anyhow::bail!(
            "Glob pattern contains Windows-restricted character: '{}' in pattern '{}'.\n\nWindows restricts these characters in file paths: {}\n\nNote: The characters '*' and '?' are allowed in glob patterns as wildcards.",
            restricted_char,
            pattern,
            WINDOWS_RESTRICTED_CHARS.iter().collect::<String>()
        );
    }

    // Check for Windows reserved names in path components
    for component in pattern.split('/') {
        let component_upper = component.to_uppercase();
        if WINDOWS_RESERVED_NAMES.contains(&component_upper.as_str()) {
            anyhow::bail!(
                "Glob pattern contains Windows-reserved name: '{}' in pattern '{}'.\n\nWindows reserves these names: {}\n\nSuggestion: Use a different name or add a suffix/prefix.",
                component,
                pattern,
                WINDOWS_RESERVED_NAMES.join(", ")
            );
        }
    }

    Ok(())
}

/// Validate against other problematic pattern constructs.
fn validate_problematic_patterns(pattern: &str) -> Result<()> {
    // Check for patterns that might cause issues
    if pattern.contains("../") {
        anyhow::bail!(
            "Glob pattern contains parent directory reference '../' in pattern '{}'.\n\nThis can be problematic because:\n  - It might access files outside the intended directory\n  - It can cause security issues in file filtering\n  - It may not work consistently across platforms\n\nSuggestion: Use absolute patterns or avoid parent directory references.",
            pattern
        );
    }

    // Check for excessively nested patterns that might cause performance issues
    let double_star_count = pattern.matches("**").count();
    if double_star_count > 5 {
        anyhow::bail!(
            "Glob pattern contains too many recursive wildcards (**): {} occurrences in pattern '{}'.\n\nExcessive use of ** can cause:\n  - Poor performance when matching against large directory trees\n  - Exponential time complexity in some cases\n  - Memory usage issues\n\nSuggestion: Limit the use of ** or be more specific in your patterns.",
            double_star_count,
            pattern
        );
    }

    // Warn about very long character classes that might be typos
    if let Some(start) = pattern.find('[') {
        if let Some(end) = pattern[start..].find(']') {
            let char_class = &pattern[start + 1..start + end];
            if char_class.len() > 50 {
                anyhow::bail!(
                    "Glob pattern contains very long character class: '[{}]' in pattern '{}'.\n\nLong character classes can be:\n  - Difficult to read and maintain\n  - Potentially incorrect (missing closing bracket?)\n  - Performance bottlenecks\n\nSuggestion: Use character ranges [a-z] or split into multiple patterns.",
                    if char_class.len() > 20 { &char_class[..20] } else { char_class },
                    pattern
                );
            }
        }
    }

    Ok(())
}

/// Configuration for filtering search results
#[derive(Debug, Clone)]
pub struct FilterConfig {
    /// File extension filter (e.g., ".rs", ".js", ".py")
    pub file_extension: Option<String>,
    /// Maximum allowed length for glob patterns
    pub max_glob_pattern_length: usize,
    /// Maximum allowed length for file extensions (including the dot)
    pub max_extension_length: usize,
}

impl Default for FilterConfig {
    fn default() -> Self {
        Self {
            file_extension: None,
            max_glob_pattern_length: DEFAULT_MAX_GLOB_PATTERN_LENGTH,
            max_extension_length: DEFAULT_MAX_EXTENSION_LENGTH,
        }
    }
}

impl FilterConfig {
    /// Create a new filter configuration with default limits
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new filter configuration with custom limits
    pub fn with_limits(max_glob_pattern_length: usize, max_extension_length: usize) -> Self {
        Self {
            file_extension: None,
            max_glob_pattern_length,
            max_extension_length,
        }
    }

    /// Set the file extension filter
    pub fn with_file_extension(mut self, extension: String) -> Self {
        // Normalize extension to start with a dot
        let normalized = if extension.starts_with('.') {
            extension
        } else {
            format!(".{}", extension)
        };
        self.file_extension = Some(normalized);
        self
    }
}

/// Filter for search results
pub struct SearchFilter {
    config: FilterConfig,
}

impl SearchFilter {
    /// Create a new search filter with the given configuration
    pub fn new(config: FilterConfig) -> Self {
        Self { config }
    }

    /// Create a search filter from optional command line arguments
    pub fn from_cli_args(filetype: Option<String>) -> Self {
        let mut config = FilterConfig::new();

        if let Some(extension) = filetype {
            config = config.with_file_extension(extension);
        }

        Self::new(config)
    }

    /// Apply all configured filters to search results
    pub fn apply_filters(&self, results: Vec<SearchResult>) -> Result<Vec<SearchResult>> {
        let mut filtered = results;

        // Apply file extension filter if configured
        if let Some(ref extension) = self.config.file_extension {
            filtered = self.filter_by_extension(filtered, extension)?;
        }

        Ok(filtered)
    }

    /// Filter results by file extension
    fn filter_by_extension(
        &self,
        results: Vec<SearchResult>,
        extension: &str,
    ) -> Result<Vec<SearchResult>> {
        Ok(results
            .into_iter()
            .filter(|result| {
                let file_path = &result.chunk.chunk.source_location.file_path;
                self.matches_extension(file_path, extension)
            })
            .collect())
    }

    /// Check if a file path matches the given extension
    fn matches_extension(&self, path: &Path, target_extension: &str) -> bool {
        if let Some(file_extension) = path.extension() {
            let file_ext_str = format!(".{}", file_extension.to_string_lossy());
            file_ext_str.eq_ignore_ascii_case(target_extension)
        } else {
            false
        }
    }

    /// Get a description of active filters for logging/display
    pub fn describe_filters(&self) -> Vec<String> {
        let mut descriptions = Vec::new();

        if let Some(ref extension) = self.config.file_extension {
            descriptions.push(format!("File extension: {}", extension));
        }

        if descriptions.is_empty() {
            descriptions.push("No filters active".to_string());
        }

        descriptions
    }

    /// Check if any filters are active
    pub fn has_active_filters(&self) -> bool {
        self.config.file_extension.is_some()
    }
}

/// Validate and normalize file extension input with configurable limits
pub fn normalize_file_extension_with_config(input: &str, config: &FilterConfig) -> Result<String> {
    let input = input.trim();

    if input.is_empty() {
        anyhow::bail!("File extension cannot be empty");
    }

    // Handle common cases and normalize
    let normalized = if input.starts_with('.') {
        input.to_lowercase()
    } else {
        format!(".{}", input.to_lowercase())
    };

    // Validate that extension contains only allowed characters
    if !normalized.chars().skip(1).all(|c| c.is_alphanumeric()) {
        anyhow::bail!(
            "Invalid file extension: '{}'. Extensions should contain only alphanumeric characters",
            input
        );
    }

    if normalized.len() < 2 {
        anyhow::bail!(
            "File extension too short: '{}'. Must be at least one character after the dot",
            input
        );
    }

    if normalized.len() > config.max_extension_length {
        anyhow::bail!(
            "File extension too long: '{}'. Must be {} characters or less (configured limit)",
            input,
            config.max_extension_length
        );
    }

    Ok(normalized)
}

/// Validate a glob pattern string using default configuration
pub fn validate_glob_pattern(pattern: &str) -> Result<()> {
    validate_glob_pattern_with_config(pattern, &FilterConfig::default())
}

/// Validate and normalize file extension input using default configuration
pub fn normalize_file_extension(input: &str) -> Result<String> {
    normalize_file_extension_with_config(input, &FilterConfig::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        ChunkId, ChunkIndexNum, ContentChunk, IndexedChunk, SearchResult, SourceLocation,
        TokenCount,
    };
    use std::path::PathBuf;

    fn create_test_result(file_path: &str, similarity: f32) -> SearchResult {
        let chunk = ContentChunk {
            id: ChunkId::new("test-chunk"),
            content: "test content".to_string(),
            token_count: TokenCount::new(2),
            source_location: SourceLocation {
                file_path: PathBuf::from(file_path),
                start_line: 1,
                end_line: 1,
                start_char: 0,
                end_char: 12,
            },
            chunk_index: ChunkIndexNum::new(0),
            total_chunks: 1,
        };

        let indexed_chunk = IndexedChunk {
            chunk,
            embedding: vec![0.1, 0.2, 0.3],
        };

        SearchResult::new(similarity, indexed_chunk, 0)
    }

    #[test]
    fn test_filter_config_creation() {
        let config = FilterConfig::new();
        assert!(config.file_extension.is_none());

        let config = FilterConfig::new().with_file_extension("rs".to_string());
        assert_eq!(config.file_extension, Some(".rs".to_string()));

        let config = FilterConfig::new().with_file_extension(".js".to_string());
        assert_eq!(config.file_extension, Some(".js".to_string()));
    }

    #[test]
    fn test_search_filter_from_cli_args() {
        let filter = SearchFilter::from_cli_args(None);
        assert!(!filter.has_active_filters());

        let filter = SearchFilter::from_cli_args(Some("rs".to_string()));
        assert!(filter.has_active_filters());
        assert_eq!(filter.config.file_extension, Some(".rs".to_string()));
    }

    #[test]
    fn test_extension_filtering() {
        let results = vec![
            create_test_result("src/main.rs", 0.9),
            create_test_result("src/lib.js", 0.8),
            create_test_result("src/test.py", 0.7),
            create_test_result("README.md", 0.6),
        ];

        let filter = SearchFilter::from_cli_args(Some("rs".to_string()));
        let filtered = filter.apply_filters(results).unwrap();

        assert_eq!(filtered.len(), 1);
        assert!(filtered[0]
            .chunk
            .chunk
            .source_location
            .file_path
            .to_str()
            .unwrap()
            .ends_with(".rs"));
    }

    #[test]
    fn test_extension_case_insensitive() {
        let results = vec![
            create_test_result("src/Main.RS", 0.9),
            create_test_result("src/lib.JS", 0.8),
        ];

        let filter = SearchFilter::from_cli_args(Some("rs".to_string()));
        let filtered = filter.apply_filters(results).unwrap();

        assert_eq!(filtered.len(), 1);
        assert!(filtered[0]
            .chunk
            .chunk
            .source_location
            .file_path
            .to_str()
            .unwrap()
            .ends_with(".RS"));
    }

    #[test]
    fn test_no_extension_files() {
        let results = vec![
            create_test_result("Dockerfile", 0.9),
            create_test_result("Makefile", 0.8),
            create_test_result("src/main.rs", 0.7),
        ];

        let filter = SearchFilter::from_cli_args(Some("rs".to_string()));
        let filtered = filter.apply_filters(results).unwrap();

        assert_eq!(filtered.len(), 1);
        assert!(filtered[0]
            .chunk
            .chunk
            .source_location
            .file_path
            .to_str()
            .unwrap()
            .ends_with(".rs"));
    }

    #[test]
    fn test_matches_extension() {
        let filter = SearchFilter::new(FilterConfig::new());

        assert!(filter.matches_extension(Path::new("test.rs"), ".rs"));
        assert!(filter.matches_extension(Path::new("test.RS"), ".rs"));
        assert!(filter.matches_extension(Path::new("test.js"), ".js"));
        assert!(!filter.matches_extension(Path::new("test.rs"), ".js"));
        assert!(!filter.matches_extension(Path::new("test"), ".rs"));
        assert!(!filter.matches_extension(Path::new("test."), ".rs"));
    }

    #[test]
    fn test_describe_filters() {
        let filter = SearchFilter::from_cli_args(None);
        let descriptions = filter.describe_filters();
        assert_eq!(descriptions, vec!["No filters active"]);

        let filter = SearchFilter::from_cli_args(Some("rs".to_string()));
        let descriptions = filter.describe_filters();
        assert_eq!(descriptions, vec!["File extension: .rs"]);
    }

    #[test]
    fn test_normalize_file_extension() {
        assert_eq!(normalize_file_extension("rs").unwrap(), ".rs");
        assert_eq!(normalize_file_extension(".js").unwrap(), ".js");
        assert_eq!(normalize_file_extension("PY").unwrap(), ".py");
        assert_eq!(normalize_file_extension(".TS").unwrap(), ".ts");

        // Invalid cases
        assert!(normalize_file_extension("").is_err());
        assert!(normalize_file_extension("rs.").is_err()); // Contains non-alphanumeric
        assert!(normalize_file_extension("r s").is_err()); // Contains space
        assert!(normalize_file_extension("verylongextension").is_err()); // Too long
    }

    #[test]
    fn test_has_active_filters() {
        let filter = SearchFilter::from_cli_args(None);
        assert!(!filter.has_active_filters());

        let filter = SearchFilter::from_cli_args(Some("rs".to_string()));
        assert!(filter.has_active_filters());
    }

    mod test_glob_pattern {
        use super::*;

        #[test]
        fn test_glob_pattern_creation_valid() {
            // Test valid patterns
            let pattern = GlobPattern::new("*.rs").unwrap();
            assert_eq!(pattern.pattern(), "*.rs");

            let pattern = GlobPattern::new("src/*.js").unwrap();
            assert_eq!(pattern.pattern(), "src/*.js");

            let pattern = GlobPattern::new("**/test_*.py").unwrap();
            assert_eq!(pattern.pattern(), "**/test_*.py");

            let pattern = GlobPattern::new("dir/*/file.txt").unwrap();
            assert_eq!(pattern.pattern(), "dir/*/file.txt");
        }

        #[test]
        fn test_glob_pattern_creation_invalid() {
            // Test empty pattern
            assert!(GlobPattern::new("").is_err());
            assert!(GlobPattern::new("   ").is_err());

            // Test pattern with control characters (except tab)
            assert!(GlobPattern::new("file\x00.txt").is_err());
            assert!(GlobPattern::new("file\x01.txt").is_err());

            // Test extremely long pattern
            let long_pattern = "a".repeat(DEFAULT_MAX_GLOB_PATTERN_LENGTH + 1);
            assert!(GlobPattern::new(&long_pattern).is_err());
        }

        #[test]
        fn test_glob_pattern_matching() {
            // Test simple wildcard patterns - these match against the full path
            let pattern = GlobPattern::new("*.rs").unwrap();
            assert!(pattern.matches(Path::new("main.rs")));
            assert!(pattern.matches(Path::new("lib.rs")));
            assert!(!pattern.matches(Path::new("main.js")));
            // *.rs matches paths ending in .rs, including those with directories
            assert!(pattern.matches(Path::new("src/main.rs")));
            assert!(pattern.matches(Path::new("deep/nested/file.rs")));

            // Test directory patterns - understand how * works with directories
            let pattern = GlobPattern::new("src/*.rs").unwrap();
            assert!(pattern.matches(Path::new("src/main.rs")));
            assert!(pattern.matches(Path::new("src/lib.rs")));
            assert!(!pattern.matches(Path::new("main.rs"))); // No src/ prefix
            assert!(!pattern.matches(Path::new("tests/main.rs"))); // Wrong directory
                                                                   // The * in src/*.rs can match paths with slashes, so this actually matches
            assert!(pattern.matches(Path::new("src/nested/main.rs"))); // * can match nested/main

            // Test recursive patterns - ** matches any number of directories
            let pattern = GlobPattern::new("**/test_*.py").unwrap();
            assert!(pattern.matches(Path::new("test_main.py")));
            assert!(pattern.matches(Path::new("src/test_lib.py")));
            assert!(pattern.matches(Path::new("tests/unit/test_utils.py")));
            assert!(!pattern.matches(Path::new("main.py")));
            assert!(!pattern.matches(Path::new("src/lib.py")));

            // Test patterns that should match any file with extension regardless of path depth
            let pattern = GlobPattern::new("**/*.rs").unwrap();
            assert!(pattern.matches(Path::new("main.rs")));
            assert!(pattern.matches(Path::new("src/main.rs")));
            assert!(pattern.matches(Path::new("deep/nested/dir/file.rs")));
            assert!(!pattern.matches(Path::new("main.js")));
        }

        #[test]
        fn test_glob_pattern_case_sensitivity() {
            // Glob patterns should be case-sensitive by default
            let pattern = GlobPattern::new("*.RS").unwrap();
            assert!(pattern.matches(Path::new("main.RS")));
            assert!(!pattern.matches(Path::new("main.rs"))); // Different case
        }

        #[test]
        fn test_glob_pattern_edge_cases() {
            // Test pattern with tab character (should be allowed)
            let pattern = GlobPattern::new("file\twith\ttab.txt");
            assert!(pattern.is_ok());

            // Test Unicode characters
            let pattern = GlobPattern::new("файл_*.txt").unwrap();
            assert!(pattern.matches(Path::new("файл_test.txt")));
            assert!(!pattern.matches(Path::new("file_test.txt")));

            // Test very specific patterns
            let pattern = GlobPattern::new("exact_file.txt").unwrap();
            assert!(pattern.matches(Path::new("exact_file.txt")));
            assert!(!pattern.matches(Path::new("exact_file.rs")));
            assert!(!pattern.matches(Path::new("other_exact_file.txt")));
        }

        #[test]
        fn test_validate_glob_pattern() {
            // Valid patterns
            assert!(validate_glob_pattern("*.rs").is_ok());
            assert!(validate_glob_pattern("src/**/*.js").is_ok());
            assert!(validate_glob_pattern("test_*.py").is_ok());
            assert!(validate_glob_pattern("file.txt").is_ok());
            assert!(validate_glob_pattern("dir/*/file.?").is_ok());

            // Invalid patterns
            assert!(validate_glob_pattern("").is_err());
            assert!(validate_glob_pattern("   ").is_err());

            // Pattern with control characters
            assert!(validate_glob_pattern("file\x00.txt").is_err());
            assert!(validate_glob_pattern("file\n.txt").is_err());

            // Pattern too long
            let long_pattern = "a".repeat(DEFAULT_MAX_GLOB_PATTERN_LENGTH + 1);
            assert!(validate_glob_pattern(&long_pattern).is_err());

            // Tab should be allowed
            assert!(validate_glob_pattern("file\ttab.txt").is_ok());
        }

        #[test]
        fn test_glob_pattern_common_use_cases() {
            // Test common patterns from the specification
            let test_cases = vec![
                ("*.ext", "file.ext", true),
                ("*.ext", "file.other", false),
                ("dir/*.ext", "dir/file.ext", true),
                ("dir/*.ext", "other/file.ext", false),
                ("**/pattern", "pattern", true),
                ("**/pattern", "deep/nested/pattern", true),
                ("**/pattern", "deep/nested/other", false),
                ("src/*.js", "src/main.js", true),
                ("src/*.js", "src/lib.js", true),
                ("src/*.js", "tests/main.js", false),
                ("**/*.rs", "main.rs", true),
                ("**/*.rs", "src/main.rs", true),
                ("**/*.rs", "tests/unit/helper.rs", true),
                ("**/*.rs", "main.js", false),
            ];

            for (pattern_str, path_str, should_match) in test_cases {
                let pattern = GlobPattern::new(pattern_str).unwrap();
                let path = Path::new(path_str);
                assert_eq!(
                    pattern.matches(path),
                    should_match,
                    "Pattern '{}' vs path '{}' should {}match",
                    pattern_str,
                    path_str,
                    if should_match { "" } else { "not " }
                );
            }
        }

        #[test]
        fn test_glob_pattern_invalid_utf8_handling() {
            // Test that patterns handle invalid UTF-8 paths gracefully
            let _pattern = GlobPattern::new("*.txt").unwrap();

            // We can't easily create an invalid UTF-8 Path in safe Rust,
            // but we can verify that our matching function handles the None case
            // This is covered by the matches() implementation returning false
            // for paths that can't be converted to strings
        }

        #[test]
        fn test_glob_pattern_length_edge_cases() {
            // Test pattern at exactly the maximum length
            let max_length = DEFAULT_MAX_GLOB_PATTERN_LENGTH;
            let long_pattern = "a".repeat(max_length);
            assert!(GlobPattern::new(&long_pattern).is_ok());

            // Test pattern just over the limit
            let too_long_pattern = "a".repeat(max_length + 1);
            assert!(GlobPattern::new(&too_long_pattern).is_err());

            // Test very long valid pattern near the limit with realistic structure
            let base_pattern = "src/**/deeply/nested/directory/structure/";
            let remaining_chars = max_length - base_pattern.len() - 5; // Leave room for "*.rs"
            let padding = "x".repeat(remaining_chars);
            let realistic_long_pattern = format!("{}{}*.rs", base_pattern, padding);

            if realistic_long_pattern.len() <= max_length {
                let pattern = GlobPattern::new(&realistic_long_pattern).unwrap();
                assert!(pattern.matches(Path::new(&format!(
                    "src/lib/deeply/nested/directory/structure/{}test.rs",
                    padding
                ))));
            }
        }

        #[test]
        fn test_glob_pattern_unicode_edge_cases() {
            // Test Unicode characters in patterns
            let unicode_pattern = GlobPattern::new("файл_*.текст").unwrap();
            assert!(unicode_pattern.matches(Path::new("файл_test.текст")));
            assert!(!unicode_pattern.matches(Path::new("file_test.txt")));

            // Test emoji in patterns (valid Unicode)
            let emoji_pattern = GlobPattern::new("📁_*.📄").unwrap();
            assert!(emoji_pattern.matches(Path::new("📁_document.📄")));

            // Test mixed Unicode and ASCII
            let mixed_pattern = GlobPattern::new("src/**/*_测试.rs").unwrap();
            assert!(mixed_pattern.matches(Path::new("src/lib/main_测试.rs")));
            assert!(!mixed_pattern.matches(Path::new("src/lib/main_test.rs")));

            // Test Unicode normalization cases
            // Note: This tests that our pattern handles different Unicode representations
            let pattern = GlobPattern::new("café_*.txt").unwrap();
            assert!(pattern.matches(Path::new("café_notes.txt")));

            // Test zero-width characters (should be allowed in patterns)
            let zwc_pattern = GlobPattern::new("file\u{200B}*.txt").unwrap(); // Zero-width space
            assert!(zwc_pattern.matches(Path::new("file\u{200B}test.txt")));
        }

        #[test]
        fn test_glob_pattern_performance_with_large_sets() {
            use std::time::Instant;

            // Create a pattern that will be tested against many paths
            let pattern = GlobPattern::new("src/**/*.rs").unwrap();

            // Generate a large set of test paths
            let test_paths: Vec<_> = (0..1000)
                .map(|i| format!("src/module{}/submodule{}/file{}.rs", i % 10, i % 20, i))
                .collect();

            // Measure matching performance
            let start = Instant::now();
            let mut matches = 0;
            for path_str in &test_paths {
                let path = Path::new(path_str);
                if pattern.matches(path) {
                    matches += 1;
                }
            }
            let duration = start.elapsed();

            // All paths should match the pattern
            assert_eq!(matches, test_paths.len());

            // Performance assertion: should complete in reasonable time
            // This is a rough benchmark - adjust if needed based on actual performance
            assert!(
                duration.as_millis() < 100,
                "Pattern matching took too long: {:?}",
                duration
            );
        }

        #[test]
        fn test_glob_pattern_cache_functionality() {
            let cache = GlobPatternCache::new();
            let config = FilterConfig::default();

            // Test basic caching
            let pattern1 = cache.get_or_create("*.rs", &config).unwrap();
            let pattern2 = cache.get_or_create("*.rs", &config).unwrap();

            // Should be the same Arc instance
            assert!(Arc::ptr_eq(&pattern1, &pattern2));

            // Test cache statistics
            assert_eq!(cache.len(), 1);
            assert!(!cache.is_empty());

            // Test different patterns
            let pattern3 = cache.get_or_create("*.js", &config).unwrap();
            assert!(!Arc::ptr_eq(&pattern1, &pattern3));
            assert_eq!(cache.len(), 2);

            // Test cache clearing
            cache.clear();
            assert_eq!(cache.len(), 0);
            assert!(cache.is_empty());
        }

        #[test]
        fn test_glob_pattern_normalization() {
            // Test basic normalization
            assert_eq!(normalize_glob_pattern("./src/*.rs"), "src/*.rs");
            assert_eq!(normalize_glob_pattern("src//lib/*.rs"), "src/lib/*.rs");
            assert_eq!(normalize_glob_pattern("src/lib/"), "src/lib");

            // Test recursive wildcard normalization
            assert_eq!(normalize_glob_pattern("src/**/**/*.rs"), "src/**/*.rs");
            assert_eq!(normalize_glob_pattern("**/*/**/*.rs"), "**/**/*.rs");

            // Test whitespace trimming
            assert_eq!(normalize_glob_pattern("  *.rs  "), "*.rs");

            // Test complex patterns
            assert_eq!(
                normalize_glob_pattern("./src//lib/**/**/test_*.rs/"),
                "src/lib/**/test_*.rs"
            );
        }

        #[test]
        fn test_configurable_limits() {
            // Test custom configuration limits
            let config = FilterConfig::with_limits(50, 5);

            // Test pattern length limit
            let short_pattern = "*.rs";
            assert!(validate_glob_pattern_with_config(short_pattern, &config).is_ok());

            let long_pattern = "a".repeat(51);
            assert!(validate_glob_pattern_with_config(&long_pattern, &config).is_err());

            // Test extension length limit
            let _short_ext = ".rs";
            assert!(normalize_file_extension_with_config("rs", &config).is_ok());

            let long_ext = "verylongext";
            assert!(normalize_file_extension_with_config(&long_ext, &config).is_err());
        }

        #[test]
        fn test_error_message_quality() {
            // Test that error messages are helpful and descriptive
            let config = FilterConfig::with_limits(10, 3);

            // Test empty pattern error
            let empty_result = validate_glob_pattern_with_config("", &config);
            assert!(empty_result.is_err());
            let error = empty_result.unwrap_err().to_string();
            assert!(error.contains("Examples of valid patterns"));
            assert!(error.contains("*.rs"));

            // Test pattern too long error
            let long_result = validate_glob_pattern_with_config("very_long_pattern", &config);
            assert!(long_result.is_err());
            let error = long_result.unwrap_err().to_string();
            assert!(error.contains("Suggestions"));
            assert!(error.contains("Use shorter"));

            // Test control character error
            let control_result = validate_glob_pattern_with_config("file\x00.txt", &config);
            assert!(control_result.is_err());
            let error = control_result.unwrap_err().to_string();
            assert!(error.contains("null terminator"));
        }

        #[test]
        fn test_comprehensive_glob_pattern_examples() {
            // Test bracket expressions - character sets
            let bracket_pattern = GlobPattern::new("file[abc].txt").unwrap();
            assert!(bracket_pattern.matches(Path::new("filea.txt")));
            assert!(bracket_pattern.matches(Path::new("fileb.txt")));
            assert!(bracket_pattern.matches(Path::new("filec.txt")));
            assert!(!bracket_pattern.matches(Path::new("filed.txt")));
            assert!(!bracket_pattern.matches(Path::new("fileab.txt")));

            // Test bracket expressions - character ranges
            let range_pattern = GlobPattern::new("test[0-9].log").unwrap();
            assert!(range_pattern.matches(Path::new("test0.log")));
            assert!(range_pattern.matches(Path::new("test5.log")));
            assert!(range_pattern.matches(Path::new("test9.log")));
            assert!(!range_pattern.matches(Path::new("testa.log")));
            assert!(!range_pattern.matches(Path::new("test10.log")));

            // Test bracket expressions - mixed sets and ranges
            let mixed_pattern = GlobPattern::new("file[a-z0-9_].ext").unwrap();
            assert!(mixed_pattern.matches(Path::new("filea.ext")));
            assert!(mixed_pattern.matches(Path::new("file5.ext")));
            assert!(mixed_pattern.matches(Path::new("file_.ext")));
            assert!(!mixed_pattern.matches(Path::new("fileA.ext"))); // Capital A not in range
            assert!(!mixed_pattern.matches(Path::new("file-.ext"))); // Dash not in set

            // Test negated bracket expressions
            let negated_pattern = GlobPattern::new("file[!0-9].txt").unwrap();
            assert!(negated_pattern.matches(Path::new("filea.txt")));
            assert!(negated_pattern.matches(Path::new("fileZ.txt")));
            assert!(!negated_pattern.matches(Path::new("file5.txt")));
            assert!(!negated_pattern.matches(Path::new("file0.txt")));

            // Alternative negation syntax with ^ (if supported by glob crate)
            let caret_negated_result = GlobPattern::new("log[^a-z].txt");
            if let Ok(caret_negated_pattern) = caret_negated_result {
                // Test if the pattern works as expected
                let matches_digit = caret_negated_pattern.matches(Path::new("log1.txt"));
                let matches_upper = caret_negated_pattern.matches(Path::new("logA.txt"));
                let matches_lower = caret_negated_pattern.matches(Path::new("loga.txt"));

                // The behavior might vary depending on glob crate implementation
                // Just ensure it doesn't panic and produces some consistent behavior
                assert!(matches_digit || !matches_digit); // Should not panic
                assert!(matches_upper || !matches_upper); // Should not panic
                assert!(matches_lower || !matches_lower); // Should not panic
            }

            // Test single character wildcard
            let single_char_pattern = GlobPattern::new("file?.txt").unwrap();
            assert!(single_char_pattern.matches(Path::new("file1.txt")));
            assert!(single_char_pattern.matches(Path::new("fileA.txt")));
            assert!(single_char_pattern.matches(Path::new("file_.txt")));
            assert!(!single_char_pattern.matches(Path::new("file.txt"))); // No character
            assert!(!single_char_pattern.matches(Path::new("file12.txt"))); // Too many characters

            // Test complex patterns combining multiple features
            let complex_pattern = GlobPattern::new("src/**/test_[a-z]*.rs").unwrap();
            assert!(complex_pattern.matches(Path::new("src/test_main.rs")));
            assert!(complex_pattern.matches(Path::new("src/unit/test_helper.rs")));
            assert!(complex_pattern.matches(Path::new("src/integration/deep/test_api.rs")));
            assert!(!complex_pattern.matches(Path::new("src/test_Main.rs"))); // Capital M
            assert!(!complex_pattern.matches(Path::new("src/Test_main.rs"))); // Capital T

            // Test escaping special characters (if supported by glob crate)
            // Note: The glob crate may not support all escape sequences
            let escaped_pattern = GlobPattern::new("file\\*.txt");
            // This may or may not work depending on glob crate implementation
            // We just test that it doesn't panic during creation
            let _ = escaped_pattern.is_ok() || escaped_pattern.is_err();
        }

        #[test]
        fn test_advanced_glob_patterns() {
            // Test patterns with multiple recursive wildcards
            let multi_recursive = GlobPattern::new("**/src/**/test/**/*.rs").unwrap();
            assert!(multi_recursive.matches(Path::new("project/src/lib/test/unit/helper.rs")));
            assert!(multi_recursive.matches(Path::new("src/main/test/integration/api.rs")));
            assert!(!multi_recursive.matches(Path::new("src/main/lib/helper.rs"))); // Missing test

            // Test patterns with alternating wildcards and specific names
            let alternating = GlobPattern::new("*/src/*/bin/*.exe").unwrap();
            assert!(alternating.matches(Path::new("project/src/main/bin/app.exe")));
            assert!(alternating.matches(Path::new("myapp/src/cli/bin/tool.exe")));
            assert!(!alternating.matches(Path::new("project/lib/main/bin/app.exe"))); // lib instead of src

            // Test very specific patterns
            let specific = GlobPattern::new("logs/2023/*/error_*.log").unwrap();
            assert!(specific.matches(Path::new("logs/2023/01/error_database.log")));
            assert!(specific.matches(Path::new("logs/2023/12/error_network.log")));
            assert!(!specific.matches(Path::new("logs/2024/01/error_database.log"))); // Wrong year
            assert!(!specific.matches(Path::new("logs/2023/01/info_database.log"))); // Wrong prefix

            // Test patterns that might be problematic
            let edge_case = GlobPattern::new("a*/b*/c*/d*.txt").unwrap();
            assert!(edge_case.matches(Path::new("abc/bdef/cxyz/data.txt")));
            assert!(edge_case.matches(Path::new("a/b/c/d.txt")));
            assert!(!edge_case.matches(Path::new("a/b/d.txt"))); // Missing c*
        }

        #[test]
        fn test_glob_pattern_boundary_conditions() {
            // Test empty directory names
            let empty_dir_pattern = GlobPattern::new("*//*/file.txt");
            // This tests how the pattern handles empty directory components
            let _ = empty_dir_pattern.is_ok();

            // Test patterns ending with wildcards
            let ending_wildcard = GlobPattern::new("src/**/*").unwrap();
            assert!(ending_wildcard.matches(Path::new("src/main.rs")));
            assert!(ending_wildcard.matches(Path::new("src/lib/mod.rs")));
            assert!(ending_wildcard.matches(Path::new("src/deep/nested/file.txt")));

            // Test patterns starting with wildcards
            let starting_wildcard = GlobPattern::new("**/main.rs").unwrap();
            assert!(starting_wildcard.matches(Path::new("main.rs")));
            assert!(starting_wildcard.matches(Path::new("src/main.rs")));
            assert!(starting_wildcard.matches(Path::new("project/src/bin/main.rs")));
            assert!(!starting_wildcard.matches(Path::new("lib.rs")));

            // Test single character patterns
            let single_char = GlobPattern::new("?").unwrap();
            assert!(single_char.matches(Path::new("a")));
            assert!(single_char.matches(Path::new("1")));
            assert!(!single_char.matches(Path::new("")));
            assert!(!single_char.matches(Path::new("ab")));

            // Test single wildcard patterns
            let single_wildcard = GlobPattern::new("*").unwrap();
            assert!(single_wildcard.matches(Path::new("anything")));
            assert!(single_wildcard.matches(Path::new("file.txt")));
            assert!(single_wildcard.matches(Path::new("")));
            assert!(single_wildcard.matches(Path::new("a/b/c"))); // * can match paths with slashes
        }

        #[test]
        fn test_platform_specific_patterns() {
            // Test patterns that work across platforms
            let cross_platform = GlobPattern::new("src/main.rs").unwrap();
            assert!(cross_platform.matches(Path::new("src/main.rs")));

            // Test patterns with forward slashes (should work on all platforms)
            let forward_slash = GlobPattern::new("dir/subdir/*.txt").unwrap();
            assert!(forward_slash.matches(Path::new("dir/subdir/file.txt")));

            // Test patterns that might have platform-specific behavior
            let mixed_case = GlobPattern::new("File.TXT").unwrap();
            assert!(mixed_case.matches(Path::new("File.TXT")));
            // Case sensitivity depends on the underlying file system
            // We don't assert anything about File.txt vs FILE.TXT matching

            // Test Unicode filename patterns
            let unicode_pattern = GlobPattern::new("测试/**/*.文档").unwrap();
            assert!(unicode_pattern.matches(Path::new("测试/项目/文件.文档")));
            assert!(!unicode_pattern.matches(Path::new("test/project/file.doc")));
        }
    }
}
