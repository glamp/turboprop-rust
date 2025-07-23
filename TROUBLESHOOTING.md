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

1. **Check GitHub Issues**: [github.com/your-org/turboprop-rust/issues](https://github.com/your-org/turboprop-rust/issues)

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