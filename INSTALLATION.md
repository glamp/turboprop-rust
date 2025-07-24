# TurboProp Installation Guide

This guide covers various methods to install TurboProp on different platforms.

## Quick Installation

### Option 1: Cargo (Recommended)

If you have Rust installed, this is the simplest method:

```bash
cargo install turboprop
```

This will download, compile, and install the latest version from crates.io.

### Option 2: Pre-built Binaries

Download pre-built binaries from our [GitHub Releases](https://github.com/glamp/turboprop-rust/releases):

**Linux (x86_64)**:
```bash
curl -L https://github.com/glamp/turboprop-rust/releases/latest/download/turboprop-linux-x86_64.tar.gz | tar xz
sudo mv tp /usr/local/bin/
```

**macOS**:
```bash
curl -L https://github.com/glamp/turboprop-rust/releases/latest/download/turboprop-macos.tar.gz | tar xz
sudo mv tp /usr/local/bin/
```

**Windows**:
Download `turboprop-windows.zip` from releases and extract `tp.exe` to a directory in your PATH.

## Platform-Specific Installation

### Linux

#### Ubuntu/Debian
```bash
# Install Rust if not already installed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install TurboProp
cargo install turboprop
```

#### CentOS/RHEL/Fedora
```bash
# Install Rust if not already installed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install TurboProp
cargo install turboprop
```

#### Arch Linux
```bash
# Install from AUR (if available)
yay -S turboprop

# Or install via Cargo
cargo install turboprop
```

### macOS

#### Using Homebrew
```bash
# Add our tap (if available)
brew tap glamp/turboprop
brew install turboprop

# Or install via Cargo
cargo install turboprop
```

#### Using MacPorts
```bash
# Install Rust if not already installed
sudo port install rust

# Install TurboProp
cargo install turboprop
```

### Windows

#### Using Scoop
```bash
# Add our bucket (if available)
scoop bucket add turboprop https://github.com/glamp/scoop-turboprop
scoop install turboprop
```

#### Using Chocolatey
```bash
# Install via Chocolatey (if available)
choco install turboprop
```

#### Manual Installation
1. Install Rust from [https://rustup.rs/](https://rustup.rs/)
2. Install TurboProp:
   ```cmd
   cargo install turboprop
   ```

## Building from Source

### Prerequisites

- **Rust**: Version 1.70+ (install from [rustup.rs](https://rustup.rs/))
- **Git**: For cloning the repository
- **OpenSSL**: Required for HTTPS requests (Linux only)

#### Linux Prerequisites
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install git build-essential pkg-config libssl-dev

# CentOS/RHEL/Fedora
sudo yum groupinstall "Development Tools"
sudo yum install git openssl-devel pkg-config

# Or for newer versions:
sudo dnf groupinstall "Development Tools"
sudo dnf install git openssl-devel pkg-config
```

#### macOS Prerequisites
```bash
# Install Xcode command line tools
xcode-select --install

# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Build Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/glamp/turboprop-rust.git
   cd turboprop-rust
   ```

2. **Build the project**:
   ```bash
   cargo build --release
   ```

3. **Install locally**:
   ```bash
   cargo install --path .
   ```

   Or copy the binary manually:
   ```bash
   cp target/release/tp ~/.local/bin/
   # Or system-wide:
   sudo cp target/release/tp /usr/local/bin/
   ```

### Development Build

For development work:

```bash
git clone https://github.com/glamp/turboprop-rust.git
cd turboprop-rust
cargo build
cargo test
cargo run -- --help
```

## Verification

Verify your installation:

```bash
tp --version
```

You should see output like:
```
tp 0.1.0
```

Test basic functionality:
```bash
tp --help
```

## Configuration

### System Requirements

- **Memory**: 512MB RAM minimum, 2GB+ recommended for large codebases
- **Disk Space**: 100MB for installation, additional space for model cache (~500MB-2GB)
- **Network**: Internet connection required for initial model download

### Environment Variables

TurboProp respects these environment variables:

```bash
export TURBOPROP_CACHE_DIR=~/.turboprop-cache  # Model cache location
export TURBOPROP_MODEL=sentence-transformers/all-MiniLM-L6-v2  # Default model
export RUST_LOG=info  # Logging level (error, warn, info, debug, trace)
```

Add these to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.) to make them persistent.

### Model Cache

On first run, TurboProp will download the embedding model (~90MB for the default model). This is cached in:

- **Linux**: `~/.cache/turboprop/`
- **macOS**: `~/Library/Caches/turboprop/`
- **Windows**: `%LOCALAPPDATA%\turboprop\cache\`

You can customize the cache location with `--cache-dir` or the `TURBOPROP_CACHE_DIR` environment variable.

## Troubleshooting Installation

### Common Issues

**"cargo: command not found"**
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

**"failed to run custom build command for `openssl-sys`" (Linux)**
```bash
# Install OpenSSL development files
sudo apt install libssl-dev pkg-config  # Ubuntu/Debian
sudo yum install openssl-devel pkg-config  # CentOS/RHEL
```

**"Permission denied" when copying to /usr/local/bin**
```bash
# Use sudo for system-wide installation
sudo cp target/release/tp /usr/local/bin/

# Or install to user directory
mkdir -p ~/.local/bin
cp target/release/tp ~/.local/bin/
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

**Network/proxy issues**
```bash
# Configure Cargo for proxy
export HTTPS_PROXY=your-proxy:port
export HTTP_PROXY=your-proxy:port
cargo install turboprop
```

**Outdated Rust version**
```bash
# Update Rust
rustup update
```

### Performance Optimization

After installation, you can optimize for your system:

1. **Set appropriate worker threads**:
   ```bash
   # Use number of CPU cores
   tp index --repo . --worker-threads $(nproc)
   ```

2. **Adjust batch size based on memory**:
   ```bash
   # Larger batch for more memory, smaller for less
   tp index --repo . --batch-size 64  # High memory
   tp index --repo . --batch-size 16  # Low memory
   ```

3. **Use SSD cache directory** for better performance:
   ```bash
   tp index --repo . --cache-dir /path/to/ssd/cache
   ```

## Upgrading

### Via Cargo
```bash
cargo install --force turboprop
```

### From Source
```bash
cd turboprop-rust
git pull
cargo build --release
cargo install --path .
```

### Checking for Updates
```bash
# Check installed version
tp --version

# Check latest available version
cargo search turboprop
```

## Uninstallation

### Remove Binary
```bash
# If installed via Cargo
cargo uninstall turboprop

# If installed manually
rm /usr/local/bin/tp  # or wherever you installed it
```

### Remove Cache and Configuration
```bash
# Remove model cache
rm -rf ~/.cache/turboprop/        # Linux
rm -rf ~/Library/Caches/turboprop/ # macOS
rmdir /s %LOCALAPPDATA%\turboprop  # Windows

# Remove any global config
rm ~/.turboprop.yml
```

## Next Steps

After installation:

1. **Quick Test**: Run `tp --help` to ensure everything is working
2. **Index a Repository**: Try `tp index --repo /path/to/your/code`
3. **Search**: Test with `tp search "your query" --repo /path/to/your/code`
4. **Read Documentation**: Check out the full [README](README.md) for usage examples
5. **Configuration**: Set up a `.turboprop.yml` file for your preferred settings

## Support

If you encounter issues during installation:

1. Check this guide for common solutions
2. Review the [Troubleshooting Guide](TROUBLESHOOTING.md)
3. Search [GitHub Issues](https://github.com/glamp/turboprop-rust/issues)
4. Create a new issue with your platform details and error messages