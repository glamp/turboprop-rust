# TurboProp Troubleshooting Guide

This guide helps you diagnose and resolve common issues with TurboProp.

## Quick Diagnostics

Before diving into specific issues, run these commands to get system information:

```bash
# Check TurboProp version and basic info
tp --version

# Check system resources
tp index --repo . --dry-run  # If supported

# Enable debug logging
export RUST_LOG=turboprop=debug
tp --help

# Check configuration
cat .turboprop.yml  # In your project root
```

## Installation Issues

### "cargo: command not found"

**Problem**: Rust/Cargo is not installed or not in PATH.

**Solution**:
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Verify installation
cargo --version
```

**For existing Rust installations**:
```bash
# Add Cargo to PATH
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### "failed to run custom build command for `openssl-sys`"

**Problem**: Missing OpenSSL development libraries (Linux).

**Solution**:
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install libssl-dev pkg-config

# CentOS/RHEL/Fedora
sudo yum install openssl-devel pkg-config
# Or for newer versions:
sudo dnf install openssl-devel pkg-config

# Then retry installation
cargo install turboprop
```

### "Permission denied" during installation

**Problem**: Insufficient permissions to install to system directories.

**Solutions**:
```bash
# Option 1: Install to user directory
cargo install --root ~/.local turboprop
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Option 2: Use sudo (if needed)
sudo cargo install turboprop --root /usr/local

# Option 3: Build and copy manually
cargo build --release
mkdir -p ~/.local/bin
cp target/release/tp ~/.local/bin/
```

### Build fails with "linker `cc` not found"

**Problem**: Missing C compiler/build tools.

**Solution**:
```bash
# Ubuntu/Debian
sudo apt install build-essential

# CentOS/RHEL/Fedora
sudo yum groupinstall "Development Tools"
# Or:
sudo dnf groupinstall "Development Tools"

# macOS
xcode-select --install

# Then retry
cargo install turboprop
```

## Indexing Issues

### "No index found in repository"

**Problem**: Trying to search without creating an index first.

**Solution**:
```bash
# Create an index first
tp index --repo .

# Then search
tp search "your query" --repo .
```

### "Permission denied" when creating index

**Problem**: No write permissions in the target directory.

**Solutions**:
```bash
# Check permissions
ls -la .turboprop/

# Fix permissions
chmod 755 .turboprop/
chmod 644 .turboprop/*

# Or use a different location
tp index --repo . --index-dir ~/turboprop-index
tp search "query" --repo . --index-dir ~/turboprop-index
```

### Model download fails

**Problem**: Cannot download embedding model from Hugging Face.

**Common causes and solutions**:

1. **Network connectivity**:
   ```bash
   # Test connectivity
   ping huggingface.co
   curl -I https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
   ```

2. **Corporate firewall/proxy**:
   ```bash
   # Set proxy environment variables
   export HTTP_PROXY=http://proxy.company.com:8080
   export HTTPS_PROXY=http://proxy.company.com:8080
   export NO_PROXY=localhost,127.0.0.1
   tp index --repo .
   ```

3. **Hugging Face Hub issues**:
   ```bash
   # Try a different model
   tp index --repo . --model "sentence-transformers/paraphrase-MiniLM-L6-v2"
   
   # Or use local cache directory
   mkdir -p ~/turboprop-models
   tp index --repo . --cache-dir ~/turboprop-models
   ```

4. **Disk space issues**:
   ```bash
   # Check available space
   df -h ~/.cache/turboprop/
   
   # Clean old models
   rm -rf ~/.cache/turboprop/
   ```

### "Out of memory" during indexing

**Problem**: Insufficient RAM for indexing operation.

**Solutions**:
```bash
# Reduce batch size
tp index --repo . --batch-size 8

# Reduce worker threads
tp index --repo . --worker-threads 1

# Limit file size
tp index --repo . --max-filesize 500kb

# Use swap (Linux/macOS)
sudo swapon --show  # Check existing swap
# Add swap if needed (Linux)
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Very slow indexing

**Problem**: Indexing takes too long.

**Diagnosis**:
```bash
# Enable verbose output to see progress
tp index --repo . --verbose

# Check system resources
top  # Or htop
iostat 1  # If available
```

**Solutions**:
```bash
# Increase parallelism (if you have resources)
tp index --repo . --worker-threads 8 --batch-size 64

# Exclude unnecessary files
tp index --repo . --exclude "**/node_modules/**" --exclude "**/target/**"

# Limit file size
tp index --repo . --max-filesize 1mb

# Use faster storage for cache
tp index --repo . --cache-dir /tmp/turboprop-cache
```

### "File too large" warnings

**Problem**: Many files are being skipped due to size limits.

**Solutions**:
```bash
# Increase the limit
tp index --repo . --max-filesize 5mb

# See which files are being skipped
tp index --repo . --verbose | grep "too large"

# Exclude specific large files
tp index --repo . --exclude "**/*.min.js" --exclude "**/bundle.*"
```

## Search Issues

### No search results

**Problem**: Search returns no results even though relevant code exists.

**Diagnosis steps**:
1. **Verify index exists**:
   ```bash
   ls -la .turboprop/
   # Should contain index files
   ```

2. **Check if files were indexed**:
   ```bash
   tp index --repo . --verbose
   # Look for "Indexed X files" message
   ```

3. **Try broader search terms**:
   ```bash
   tp search "user" --repo .  # Instead of "user authentication jwt token"
   ```

4. **Lower the similarity threshold**:
   ```bash
   tp search "query" --repo . --threshold 0.1
   ```

5. **Check file type filters**:
   ```bash
   # Remove filetype filter
   tp search "query" --repo .
   # Instead of
   tp search "query" --repo . --filetype .js
   ```

### Poor search quality

**Problem**: Search returns irrelevant results or misses relevant code.

**Solutions**:
1. **Use better search terms**:
   ```bash
   # Instead of single words
   tp search "auth" --repo .
   # Use descriptive phrases
   tp search "user authentication with jwt tokens" --repo .
   ```

2. **Try different models**:
   ```bash
   # Higher quality model (larger)
   tp index --repo . --model "sentence-transformers/all-mpnet-base-v2"
   tp search "query" --repo .
   ```

3. **Adjust similarity threshold**:
   ```bash
   # More strict (higher quality, fewer results)
   tp search "query" --repo . --threshold 0.7
   # More lenient (more results, lower quality)
   tp search "query" --repo . --threshold 0.2
   ```

4. **Use text output for debugging**:
   ```bash
   tp search "query" --repo . --output text --limit 20
   ```

### "Index is outdated" or stale results

**Problem**: Search returns results from old versions of files.

**Solution**:
```bash
# Force reindex
tp index --repo . --force

# Or enable watch mode for automatic updates
tp index --repo . --watch
```

### Search crashes or hangs

**Problem**: Search command doesn't complete.

**Diagnosis**:
```bash
# Enable debug logging
export RUST_LOG=turboprop=debug
tp search "query" --repo .

# Check system resources
top  # Look for high CPU/memory usage
```

**Solutions**:
```bash
# Try with limits
tp search "query" --repo . --limit 5

# Increase timeout (if supported)
timeout 30s tp search "query" --repo .

# Rebuild index
rm -rf .turboprop/
tp index --repo .
```

## Glob Pattern Issues

Glob patterns provide powerful file filtering but can be tricky to get right. Here are common issues and solutions.

### Pattern doesn't match expected files

**Problem**: Your glob pattern isn't matching files you expect it to match.

**Common causes and solutions**:

1. **Case sensitivity issues**:
   ```bash
   # Problem: Pattern is case-sensitive by default
   tp search "function" --filter "*.RS"    # Won't match file.rs
   
   # Solution: Use correct case
   tp search "function" --filter "*.rs"    # Matches file.rs
   ```

2. **Path structure misunderstanding**:
   ```bash
   # Problem: * doesn't cross directory boundaries
   tp search "test" --filter "src/*.rs"    # Won't match src/lib/mod.rs
   
   # Solution: Use ** for recursive matching
   tp search "test" --filter "src/**/*.rs" # Matches src/lib/mod.rs
   ```

3. **Literal vs. pattern interpretation**:
   ```bash
   # Problem: Special characters not escaped
   tp search "config" --filter "file[1].json"  # Treats [1] as character class
   
   # Solution: Quote or escape if needed
   tp search "config" --filter "file\[1\].json"  # Literal brackets
   ```

4. **Relative path confusion**:
   ```bash
   # Test what files exist
   find . -name "*.rs" | head -5
   
   # Then design pattern to match those paths
   tp search "query" --filter "**/*.rs"  # For any depth
   tp search "query" --filter "src/*.rs" # For src/ only
   ```

### Pattern matches too many files

**Problem**: Your glob pattern is too broad and matching unwanted files.

**Solutions**:

1. **Be more specific about directories**:
   ```bash
   # Too broad
   tp search "test" --filter "*.rs"
   
   # More specific
   tp search "test" --filter "tests/*.rs"
   tp search "test" --filter "src/tests/**/*.rs"
   ```

2. **Use character classes to limit matches**:
   ```bash
   # Instead of
   tp search "handler" --filter "*_handler.rs"
   
   # Use more specific pattern
   tp search "handler" --filter "*_[a-z]*_handler.rs"
   ```

3. **Combine with file type filters**:
   ```bash
   # Narrow down with both filters
   tp search "async" --filetype .rs --filter "src/**/*"
   ```

### Complex patterns not working

**Problem**: Advanced glob patterns with braces, brackets, or multiple wildcards aren't working as expected.

**Debugging steps**:

1. **Test pattern components separately**:
   ```bash
   # Start simple and build up
   tp search "query" --filter "*.js"           # Basic wildcard
   tp search "query" --filter "src/*.js"       # Add directory
   tp search "query" --filter "src/**/*.js"    # Add recursion
   tp search "query" --filter "src/**/*.{js,ts}" # Add alternatives
   ```

2. **Check brace expansion syntax**:
   ```bash
   # Correct: no spaces in braces
   tp search "query" --filter "*.{js,ts,jsx}"
   
   # Incorrect: spaces will cause issues
   tp search "query" --filter "*.{js, ts, jsx}"
   ```

3. **Validate bracket expressions**:
   ```bash
   # Correct character ranges
   tp search "query" --filter "file[0-9]*.rs"     # Numbers
   tp search "query" --filter "file[a-z]*.rs"     # Lowercase
   tp search "query" --filter "file[A-Z]*.rs"     # Uppercase
   
   # Negation
   tp search "query" --filter "file[!0-9]*.rs"    # Not numbers
   ```

### Glob patterns vs. regular expressions

**Problem**: Trying to use regex syntax in glob patterns.

**Key differences**:
- Glob: `*.js` (shell-style wildcards)
- Regex: `.*\.js$` (regular expression)

**Common mistakes**:
```bash
# Don't use regex syntax in --filter
tp search "query" --filter ".*\.js$"     # Wrong (regex syntax)
tp search "query" --filter "*.js"        # Correct (glob syntax)

# Don't use + or other regex quantifiers
tp search "query" --filter "test+.rs"    # Wrong (+ is literal)
tp search "query" --filter "test*.rs"    # Correct (use * for repetition)
```

### Performance issues with patterns

**Problem**: Complex glob patterns are making searches very slow.

**Solutions**:

1. **Avoid excessive wildcards**:
   ```bash
   # Can be slow on large codebases
   tp search "query" --filter "**/**/**/*.rs"
   
   # Simpler and faster
   tp search "query" --filter "**/*.rs"
   ```

2. **Be as specific as possible**:
   ```bash
   # Slower (searches everywhere)
   tp search "api" --filter "**/*.js"
   
   # Faster (limited scope)
   tp search "api" --filter "src/api/**/*.js"
   ```

3. **Use file type filter for extensions**:
   ```bash
   # Optimized path for extensions
   tp search "async" --filetype .rs
   
   # Less optimized
   tp search "async" --filter "*.rs"
   ```

### Unicode and special characters

**Problem**: Patterns with Unicode or special characters not working correctly.

**Solutions**:

1. **Unicode in patterns**:
   ```bash
   # Unicode characters are supported
   tp search "测试" --filter "测试_*.js"
   tp search "café" --filter "café_*.py"
   ```

2. **Spaces in file names**:
   ```bash
   # Quote the pattern if it contains spaces
   tp search "config" --filter "*config file*.json"
   tp search "test" --filter "test */*.rs"
   ```

3. **Special shell characters**:
   ```bash
   # Some characters may need quoting
   tp search "data" --filter "file\$money.js"     # Escape $
   tp search "backup" --filter "*.backup~"        # ~ is usually OK
   ```

### Platform-specific path issues

**Problem**: Patterns work on one operating system but not another.

**Solutions**:

1. **Always use forward slashes**:
   ```bash
   # Correct on all platforms
   tp search "module" --filter "src/modules/*.rs"
   
   # Wrong on Unix-like systems
   tp search "module" --filter "src\\modules\\*.rs"
   ```

2. **Case sensitivity varies by filesystem**:
   ```bash
   # Test both cases if unsure
   tp search "readme" --filter "*README*"
   tp search "readme" --filter "*readme*"
   tp search "readme" --filter "*[Rr][Ee][Aa][Dd][Mm][Ee]*"
   ```

### Debugging glob patterns

**Problem**: Need to understand what files a pattern would match before using it in search.

**Debugging techniques**:

1. **Use find command to test patterns** (Unix-like systems):
   ```bash
   # Test the pattern with find first
   find . -path "./src/*.js" -type f
   find . -path "./**/*.{js,ts}" -type f
   ```

2. **Use ls with bash globbing**:
   ```bash
   # Enable extended globbing in bash
   shopt -s globstar
   ls src/**/*.rs  # See what files match
   ```

3. **Start simple and add complexity**:
   ```bash
   # Step 1: Basic pattern
   tp search "test" --filter "*.rs" --limit 5
   
   # Step 2: Add directory
   tp search "test" --filter "src/*.rs" --limit 5
   
   # Step 3: Add recursion
   tp search "test" --filter "src/**/*.rs" --limit 5
   ```

4. **Use verbose output to see what's happening**:
   ```bash
   export RUST_LOG=turboprop=debug
   tp search "query" --filter "pattern" --limit 3
   ```

### Common pattern recipes

**Problem**: Need examples of patterns for common use cases.

**Recipe collection**:

```bash
# All source files in any language
tp search "function" --filter "**/*.{rs,js,py,go,java,cpp,c,h}"

# Test files only  
tp search "assert" --filter "**/*{test,spec}*.{rs,js,py}"
tp search "mock" --filter "**/{test,tests,spec,specs}/**/*.{rs,js,py}"

# Configuration files
tp search "database" --filter "**/*.{json,yaml,yml,toml,ini,conf,cfg}"

# Source files excluding tests
tp search "business logic" --filter "src/**/*.{rs,js,py}" 

# Files in specific frameworks/directories
tp search "component" --filter "src/components/**/*.{jsx,tsx,vue}"
tp search "model" --filter "src/models/**/*.{rs,py,js}"
tp search "handler" --filter "src/{api,handlers,routes}/**/*.{rs,js,py}"

# Documentation and text files
tp search "installation" --filter "**/*.{md,rst,txt,adoc}"

# Build and config files in project root
tp search "version" --filter "{package,Cargo,build,webpack,vite}.{json,toml,js,ts}"

# Files modified recently (use with find)
# find . -mtime -7 -name "*.rs" | head -10  # Files changed in last 7 days
```

### Glob pattern validation

**Problem**: Want to validate patterns before using them.

**Manual validation**:
```bash
# Check pattern syntax (will show validation error if invalid)
tp search --help | grep -A 20 "filter.*PATTERN"

# Test with a simple query first
tp search "test" --filter "your_pattern_here" --limit 1

# Use dry-run approach: search for something that definitely exists
echo "test content" > test_file.rs
tp search "test content" --filter "*.rs" --limit 1
rm test_file.rs
```

# MCP Server Troubleshooting

## MCP Server Issues

### Server Won't Start

**Error**: `Repository path does not exist`
```
Error: Repository path does not exist: /path/to/project
```

**Solutions**:
- Verify the path exists: `ls -la /path/to/project`
- Use absolute paths instead of relative paths
- Check directory permissions: `ls -ld /path/to/project`

**Error**: `Failed to create file watcher`
```
Error: Failed to create file watcher: too many open files
```

**Solutions**:
- Increase file descriptor limit: `ulimit -n 4096`
- Add to shell profile: `echo 'ulimit -n 4096' >> ~/.bashrc`
- On macOS, check system limits: `launchctl limit maxfiles`

### Agent Integration Issues

**Problem**: Agent can't connect to MCP server

**Symptoms**:
- "MCP server not available" 
- "Connection refused"
- "Server timeout"

**Solutions**:
1. **Verify server is running**:
   ```bash
   ps aux | grep "tp mcp"
   ```

2. **Check agent configuration**:
   - Verify JSON syntax in MCP config file
   - Ensure `tp` command is in PATH
   - Try absolute path to `tp` binary

3. **Test server manually**:
   ```bash
   tp mcp --repo . --verbose
   ```

4. **Check agent logs** for specific error messages

### Search Issues

**Problem**: Search returns no results

**Debugging steps**:
1. **Wait for indexing to complete**:
   ```bash
   tp mcp --repo . --verbose
   # Look for "Index initialization completed" message
   ```

2. **Verify files are being indexed**:
   ```bash
   tp mcp --repo . --debug 2>debug.log
   # Check debug.log for file discovery logs
   ```

3. **Test with known content**:
   - Search for exact strings that exist in your code
   - Try broader queries

4. **Check file filters**:
   - Review `.turboprop.yml` include/exclude patterns
   - Test without filters: `tp mcp --repo . --force-rebuild`

**Problem**: Search results are irrelevant

**Solutions**:
1. **Increase similarity threshold**:
   ```yaml
   # .turboprop.yml
   search:
     similarity_threshold: 0.5  # Higher = more strict
   ```

2. **Use more specific queries**:
   - Instead of "function", use "authentication function"
   - Include context: "JWT token validation function"

3. **Filter by file type**:
   ```json
   {
     "query": "database connection",
     "filetype": ".rs",
     "limit": 10
   }
   ```

### Performance Issues

**Problem**: High memory usage

**Symptoms**:
- Server uses > 2GB RAM
- System becomes slow
- Out of memory errors

**Solutions**:
1. **Reduce batch size**:
   ```yaml
   # .turboprop.yml
   embedding:
     batch_size: 8  # Default is 32
   ```

2. **Limit file size**:
   ```bash
   tp mcp --repo . --max-filesize 500kb
   ```

3. **Use restrictive patterns**:
   ```yaml
   file_discovery:
     include_patterns:
       - "src/**/*.rs"  # Only source files
     exclude_patterns:
       - "target/**"    # Exclude build artifacts
       - "*.log"        # Exclude logs
   ```

**Problem**: Slow search responses

**Solutions**:
1. **Use smaller model**:
   ```bash
   tp mcp --repo . --model sentence-transformers/all-MiniLM-L6-v2
   ```

2. **Limit concurrent searches**:
   ```yaml
   # .turboprop.yml
   mcp:
     max_concurrent_searches: 3
   ```

3. **Optimize queries**:
   - Use specific terms
   - Set appropriate limits
   - Filter by file type

### File Watching Issues

**Problem**: Index doesn't update when files change

**Debugging**:
1. **Check file watching logs**:
   ```bash
   tp mcp --repo . --debug 2>&1 | grep -i watch
   ```

2. **Test file changes**:
   - Make a simple change to a tracked file
   - Check if server logs show the change

3. **Verify file patterns**:
   - Ensure changed files match include patterns
   - Check that files aren't excluded

**Problem**: Too many file change events

**Symptoms**:
- High CPU usage
- Frequent index updates
- Server logs show constant file changes

**Solutions**:
1. **Increase debounce time**:
   ```yaml
   # .turboprop.yml
   mcp:
     update_debounce_ms: 2000  # Wait 2 seconds
   ```

2. **Exclude noisy directories**:
   ```yaml
   file_discovery:
     exclude_patterns:
       - "target/**"      # Build output
       - ".git/**"        # Git metadata
       - "node_modules/**" # Dependencies
       - "*.tmp"          # Temporary files
   ```

### Logging and Debugging

**Enable comprehensive logging**:
```bash
# Maximum debug output
RUST_LOG=debug tp mcp --repo . --debug 2>mcp-debug.log

# Focus on specific modules
RUST_LOG=turboprop::mcp=debug tp mcp --repo . 2>mcp-debug.log
```

**Useful log patterns to search for**:
- Index initialization: `grep -i "index.*init" mcp-debug.log`
- File changes: `grep -i "file.*change" mcp-debug.log`  
- Search queries: `grep -i "search.*query" mcp-debug.log`
- Errors: `grep -i error mcp-debug.log`

**Check system resources**:
```bash
# Monitor during operation
top -p $(pgrep "tp")

# Memory usage over time
while true; do
  ps -o pid,rss,vsz,comm -p $(pgrep "tp")
  sleep 5
done
```

## Getting Help

If these solutions don't resolve your issue:

1. **Gather diagnostic information**:
   - TurboProp version: `tp --version`
   - Operating system: `uname -a`
   - Available memory: `free -h` (Linux) or `vm_stat` (macOS)
   - Disk space: `df -h`

2. **Create a minimal reproduction**:
   - Try with a small test repository
   - Use default configuration
   - Document exact steps to reproduce

3. **File an issue** with:
   - Diagnostic information
   - Configuration files
   - Relevant log excerpts
   - Steps to reproduce the problem

See the [GitHub issues page](https://github.com/turboprop-org/turboprop/issues) for existing solutions and to report new problems.

## Performance Issues

### High CPU usage

**Problem**: TurboProp uses too much CPU.

**Solutions**:
```bash
# Reduce worker threads
tp index --repo . --worker-threads 2

# Lower process priority
nice -n 10 tp index --repo .

# Use during off-hours
echo "tp index --repo ." | at 2am
```

### High memory usage

**Problem**: TurboProp uses too much RAM.

**Solutions**:
```bash
# Reduce batch size
tp index --repo . --batch-size 8

# Reduce concurrent files
tp index --repo . --max-concurrent-files 10

# Monitor memory usage
watch -n 1 "ps aux | grep tp | head -n 5"
```

### Disk space issues

**Problem**: Index or cache uses too much disk space.

**Solutions**:
```bash
# Check index size
du -sh .turboprop/

# Check model cache size
du -sh ~/.cache/turboprop/

# Clean old models
rm -rf ~/.cache/turboprop/models/*

# Enable compression
tp index --repo . --compression-enabled

# Reduce backup count
echo "backup_count: 1" >> .turboprop.yml
```

## Configuration Issues

### Configuration not being loaded

**Problem**: Settings in `.turboprop.yml` are ignored.

**Diagnosis**:
```bash
# Check if file exists and is readable
ls -la .turboprop.yml
cat .turboprop.yml

# Check YAML syntax
python3 -c "import yaml; yaml.safe_load(open('.turboprop.yml'))"
```

**Common issues**:
1. **Wrong filename**: Should be `.turboprop.yml` (with dot)
2. **Wrong location**: Should be in project root or home directory
3. **YAML syntax errors**: Use a YAML validator
4. **Permissions**: File must be readable

### Environment variables not working

**Problem**: `TURBOPROP_*` environment variables are ignored.

**Solutions**:
```bash
# Check if variables are set
env | grep TURBOPROP

# Make sure they're exported
export TURBOPROP_MODEL="sentence-transformers/all-MiniLM-L12-v2"

# Add to shell profile for persistence
echo 'export TURBOPROP_MODEL="sentence-transformers/all-MiniLM-L12-v2"' >> ~/.bashrc
source ~/.bashrc
```

## Model-Specific Issues

### Model not found or unavailable

**Problem**: Error when trying to use a specific model.

**Common causes and solutions**:

1. **Model name typo**:
   ```bash
   # Check available models first
   tp model list
   
   # Use exact model name
   tp index --repo . --model "sentence-transformers/all-MiniLM-L6-v2"
   ```

2. **Model not yet downloaded**:
   ```bash
   # Download explicitly before use
   tp model download "Qwen/Qwen3-Embedding-0.6B"
   
   # Then index
   tp index --repo . --model "Qwen/Qwen3-Embedding-0.6B"
   ```

3. **Model temporarily unavailable**:
   ```bash
   # Try alternative model
   tp model list  # See available alternatives
   tp index --repo . --model "sentence-transformers/all-MiniLM-L12-v2"
   ```

### GGUF model loading failures

**Problem**: Nomic code model or other GGUF models fail to load.

**Diagnosis**:
```bash
# Check model info and requirements
tp model info "nomic-embed-code.Q5_K_S.gguf"

# Check available memory
free -h  # Linux
vm_stat  # macOS

# Enable debug logging
export RUST_LOG=turboprop=debug
tp index --repo . --model "nomic-embed-code.Q5_K_S.gguf"
```

**Solutions**:

1. **Insufficient memory**:
   ```bash
   # Check system memory requirements
   tp model info "nomic-embed-code.Q5_K_S.gguf"
   
   # Reduce batch size for large models
   tp index --repo . --model "nomic-embed-code.Q5_K_S.gguf" --batch-size 4
   
   # Close other applications to free memory
   # Add swap space if needed (Linux)
   sudo fallocate -l 4G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

2. **Model file corruption**:
   ```bash
   # Clear model cache and re-download
   tp model clear "nomic-embed-code.Q5_K_S.gguf"
   tp model download "nomic-embed-code.Q5_K_S.gguf"
   ```

3. **Architecture incompatibility**:
   ```bash
   # Check if model is compatible with your system
   uname -m  # Should show x86_64 or arm64
   
   # Try alternative models if incompatible
   tp model list | grep -E "(sentence-transformers|Qwen)"
   ```

### Qwen3 model issues

**Problem**: Qwen/Qwen3-Embedding-0.6B model fails to work or gives poor results.

**Common issues**:

1. **Missing Hugging Face token for gated models**:
   ```bash
   # Some models require authentication
   export HF_TOKEN="your_huggingface_token"
   tp model download "Qwen/Qwen3-Embedding-0.6B"
   ```

2. **Not using instructions properly**:
   ```bash
   # Qwen3 works best with instructions
   tp index --repo . \
     --model "Qwen/Qwen3-Embedding-0.6B" \
     --instruction "Represent this code for semantic search"
   
   # Use consistent instructions for search
   tp search "authentication" \
     --model "Qwen/Qwen3-Embedding-0.6B" \
     --instruction "Find code related to user authentication"
   ```

3. **PyTorch dependency issues**:
   ```bash
   # Qwen3 may require additional dependencies
   pip install torch transformers  # If using Python backend
   
   # Or use alternative models
   tp index --repo . --model "sentence-transformers/all-MiniLM-L12-v2"
   ```

### Model download and caching issues

**Problem**: Model downloads fail, are slow, or take up too much space.

**Solutions**:

1. **Network/firewall issues**:
   ```bash
   # Test connectivity
   curl -I https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
   
   # Configure proxy if needed
   export HTTP_PROXY=http://proxy.company.com:8080
   export HTTPS_PROXY=http://proxy.company.com:8080
   tp model download "model-name"
   
   # Use alternative cache directory
   export TURBOPROP_CACHE_DIR="/custom/path"
   tp model download "model-name"
   ```

2. **Disk space issues**:
   ```bash
   # Check cache directory size
   du -sh ~/.turboprop/models/
   du -sh ~/.cache/fastembed/
   
   # Clear unused models
   tp model clear
   
   # Or clear specific models
   tp model clear "large-model-name"
   
   # Use custom cache location with more space
   tp index --repo . --cache-dir /mnt/large-drive/turboprop-cache
   ```

3. **Slow downloads**:
   ```bash
   # Download models in advance
   tp model download "sentence-transformers/all-MiniLM-L6-v2"
   tp model download "sentence-transformers/all-MiniLM-L12-v2"
   
   # Use verbose mode to monitor progress
   tp model download "model-name" --verbose
   
   # Consider using smaller models for slower connections
   tp model list | grep -E "(MiniLM-L6|small)"
   ```

### Model performance issues

**Problem**: Model inference is too slow or uses too much memory.

**Diagnosis**:
```bash
# Benchmark different models
tp benchmark --models "sentence-transformers/all-MiniLM-L6-v2,nomic-embed-code.Q5_K_S.gguf"

# Monitor resource usage
top -p $(pgrep tp)  # Linux
top | grep tp       # macOS

# Check model requirements
tp model info "current-model-name"
```

**Solutions**:

1. **Switch to faster model**:
   ```bash
   # Use lightweight model for speed
   tp index --repo . --model "sentence-transformers/all-MiniLM-L6-v2"
   
   # Compare performance
   time tp search "test query" --model "sentence-transformers/all-MiniLM-L6-v2"
   time tp search "test query" --model "sentence-transformers/all-MiniLM-L12-v2"
   ```

2. **Optimize batch processing**:
   ```bash
   # Reduce batch size for memory-constrained systems
   tp index --repo . --batch-size 8
   
   # Increase batch size for systems with more memory
   tp index --repo . --batch-size 64
   ```

3. **Use model-specific optimizations**:
   ```yaml
   # .turboprop.yml
   models:
     "nomic-embed-code.Q5_K_S.gguf":
       batch_size: 4  # Smaller batches for large models
       cache_embeddings: true
     
     "sentence-transformers/all-MiniLM-L6-v2":
       batch_size: 32  # Larger batches for small models
   ```

### Model compatibility and migration issues

**Problem**: Switching between models causes issues or poor results.

**Solutions**:

1. **Re-index when switching models**:
   ```bash
   # Always re-index when changing models
   tp index --repo . --model "new-model-name" --force-rebuild
   
   # Use same model for search as indexing
   tp search "query" --model "new-model-name"
   ```

2. **Compare model results**:
   ```bash
   # Create separate indexes for comparison
   mkdir model-comparison
   cd model-comparison
   
   # Index with different models
   tp index --repo ../your-project --model "sentence-transformers/all-MiniLM-L6-v2"
   tp search "test query" > results-miniLM-L6.txt
   
   tp index --repo ../your-project --model "nomic-embed-code.Q5_K_S.gguf" --force-rebuild
   tp search "test query" > results-nomic.txt
   
   # Compare results
   diff results-miniLM-L6.txt results-nomic.txt
   ```

3. **Handle dimension mismatches**:
   ```bash
   # Different models have different embedding dimensions
   # Always re-index when switching models
   
   # Check model dimensions
   tp model info "sentence-transformers/all-MiniLM-L6-v2"  # 384 dims
   tp model info "nomic-embed-code.Q5_K_S.gguf"           # 768 dims
   tp model info "Qwen/Qwen3-Embedding-0.6B"             # 1024 dims
   ```

### Instruction-based embedding issues

**Problem**: Qwen3 instructions don't seem to improve results.

**Solutions**:

1. **Use appropriate instructions**:
   ```bash
   # For code search
   tp index --repo . \
     --model "Qwen/Qwen3-Embedding-0.6B" \
     --instruction "Represent this code for semantic search"
   
   # For documentation search
   tp index --repo . \
     --model "Qwen/Qwen3-Embedding-0.6B" \
     --instruction "Represent this documentation for question answering"
   
   # For API endpoint search
   tp search "user authentication" \
     --model "Qwen/Qwen3-Embedding-0.6B" \
     --instruction "Find API endpoints related to user authentication"
   ```

2. **Consistent instruction usage**:
   ```bash
   # Use same instruction for indexing and searching
   # Index with instruction
   tp index --repo . \
     --model "Qwen/Qwen3-Embedding-0.6B" \
     --instruction "Represent this code for semantic search"
   
   # Search with same instruction
   tp search "database connection" \
     --model "Qwen/Qwen3-Embedding-0.6B" \
     --instruction "Represent this code for semantic search"
   ```

3. **Test different instructions**:
   ```bash
   # Try various instruction styles
   tp search "error handling" \
     --instruction "Find code that handles errors and exceptions"
   
   tp search "error handling" \
     --instruction "Represent error handling code for retrieval"
   
   tp search "error handling" \
     --instruction "Query: error handling code"
   ```

### Model validation and health checks

**Problem**: Need to verify model is working correctly.

**Health check commands**:
```bash
# Basic model validation
tp model info "your-model-name"

# Test model loading
tp index --repo . --model "your-model-name" --limit 1 --verbose

# Benchmark model performance
tp benchmark --models "your-model-name" --text-count 10

# Test search functionality
echo "test function() { return true; }" > test_file.js
tp index --repo . --model "your-model-name"
tp search "test function" --model "your-model-name"
rm test_file.js
```

**Validation checklist**:
```bash
# 1. Model downloads successfully
tp model download "model-name"

# 2. Model loads without errors
tp model info "model-name"

# 3. Can generate embeddings
tp index --repo . --model "model-name" --limit 1

# 4. Search produces reasonable results
tp search "common term in your codebase" --model "model-name" --limit 3

# 5. Performance is acceptable
time tp search "test query" --model "model-name"
```

## Watch Mode Issues

### Watch mode not detecting changes

**Problem**: Files are modified but index isn't updated.

**Solutions**:
```bash
# Check if watch is actually running
ps aux | grep "tp.*watch"

# Increase debounce time for slow filesystems
tp index --repo . --watch --watch-debounce 1000

# Check filesystem limits (Linux)
cat /proc/sys/fs/inotify/max_user_watches
# Increase if needed:
echo fs.inotify.max_user_watches=524288 | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

### Watch mode crashes

**Problem**: Watch mode stops working after some time.

**Solutions**:
```bash
# Run with debug logging
export RUST_LOG=turboprop=debug
tp index --repo . --watch

# Use a process manager
nohup tp index --repo . --watch > turboprop.log 2>&1 &

# Or use systemd (Linux)
cat > ~/.config/systemd/user/turboprop.service << EOF
[Unit]
Description=TurboProp File Watcher
[Service]
ExecStart=/path/to/tp index --repo /path/to/your/repo --watch
Restart=always
[Install]
WantedBy=default.target
EOF

systemctl --user enable turboprop.service
systemctl --user start turboprop.service
```

## Platform-Specific Issues

### macOS Issues

**"tp" cannot be opened because the developer cannot be verified**:
```bash
# Remove quarantine attribute
xattr -d com.apple.quarantine /usr/local/bin/tp

# Or build from source
cargo install turboprop
```

**Spotlight indexing conflicts**:
```bash
# Exclude .turboprop from Spotlight
sudo mdutil -i off .turboprop/
```

### Windows Issues

**Long path names**:
```bash
# Enable long paths in Windows 10+
# Run as Administrator:
# New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

**Antivirus interference**:
- Add `tp.exe` to antivirus exclusions
- Add `.turboprop/` directories to exclusions

### Linux Issues

**File descriptor limits**:
```bash
# Check current limit
ulimit -n

# Increase temporarily
ulimit -n 4096

# Increase permanently
echo "* soft nofile 4096" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 4096" | sudo tee -a /etc/security/limits.conf
```

## Advanced Debugging

### Debug logging

Enable comprehensive debugging:
```bash
export RUST_LOG=turboprop=trace,fastembed=debug
export TURBOPROP_VERBOSE=true
tp index --repo . 2>&1 | tee turboprop-debug.log
```

### Memory profiling

```bash
# Linux - use valgrind (if available)
valgrind --tool=massif tp index --repo .

# macOS - use instruments
# Run through Xcode Instruments

# Cross-platform - use built-in tools
time tp index --repo .
```

### Performance profiling

```bash
# Basic timing
time tp index --repo .
time tp search "query" --repo .

# More detailed (Linux)
strace -c tp index --repo .
perf record tp index --repo .
perf report
```

## Getting Help

If these solutions don't resolve your issue:

1. **Check GitHub Issues**: [github.com/your-org/turboprop-rust/issues](https://github.com/glamp/turboprop-rust/issues)

2. **Create a detailed issue** with:
   - TurboProp version: `tp --version`
   - Operating system and version
   - Rust version: `rustc --version`
   - Complete error message
   - Minimal reproduction steps
   - Configuration file (if applicable)

3. **Include diagnostic information**:
   ```bash
   # System info
   uname -a
   
   # TurboProp debug info
   export RUST_LOG=turboprop=debug
   tp --version 2>&1
   
   # Configuration
   cat .turboprop.yml
   env | grep TURBOPROP
   ```

4. **Provide a minimal test case**:
   ```bash
   # Create a small test repository
   mkdir test-repo
   cd test-repo
   echo "function hello() { console.log('world'); }" > test.js
   tp index --repo .
   tp search "hello function" --repo .
   ```

## Frequently Asked Questions

**Q: Can I use TurboProp with very large repositories (100,000+ files)?**
A: Currently optimized for up to 10,000 files. For larger repositories, consider:
- Using more restrictive file patterns
- Excluding build/dependency directories
- Breaking into smaller sub-repositories

**Q: Does TurboProp work offline?**
A: After initial model download, yes. Models are cached locally.

**Q: Can I use custom embedding models?**
A: Yes, any HuggingFace sentence-transformers model is supported.

**Q: How much disk space does the index require?**
A: Typically 10-30% of your source code size, depending on compression settings.

**Q: Can multiple instances of TurboProp run simultaneously?**
A: Yes, but they should use different cache directories to avoid conflicts.

## See Also

- **[README](README.md)** - Basic usage and getting started
- **[Installation Guide](INSTALLATION.md)** - Platform-specific installation troubleshooting
- **[Model Documentation](MODELS.md)** - Model selection and performance optimization
- **[Configuration Guide](CONFIGURATION.md)** - Advanced configuration options
- **[API Reference](docs/API.md)** - Error handling in programmatic usage