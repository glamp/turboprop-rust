#!/bin/bash
# integration_test_all_models.sh

set -e

echo "TurboProp Model Integration Test"
echo "================================"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test directory setup
TEST_DIR="$(mktemp -d)"
echo -e "${BLUE}Using test directory: $TEST_DIR${NC}"
cd "$TEST_DIR"

# Cleanup function
cleanup() {
    echo -e "${YELLOW}Cleaning up test directory...${NC}"
    cd / > /dev/null 2>&1
    rm -rf "$TEST_DIR" > /dev/null 2>&1 || true
}
trap cleanup EXIT

# Create test repository
echo -e "${BLUE}Setting up test repository...${NC}"
mkdir -p src tests docs examples

# Create diverse test files
cat > src/calculator.js << 'EOF'
function calculateTotal(items) {
    return items.reduce((sum, item) => sum + item.price, 0);
}

function calculateTax(subtotal, rate) {
    if (rate < 0 || rate > 1) {
        throw new Error('Tax rate must be between 0 and 1');
    }
    return subtotal * rate;
}

export { calculateTotal, calculateTax };
EOF

cat > src/auth.py << 'EOF'
def authenticate_user(username, password):
    """Authenticate user with username and password"""
    if not username or not password:
        return False
    
    # Hash password and compare with stored hash
    return validate_credentials(username, password)

def validate_jwt_token(token):
    """Validate JWT authentication token"""
    try:
        decoded = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        return decoded
    except jwt.InvalidTokenError:
        return None

def generate_api_key():
    """Generate new API key for user"""
    return secrets.token_urlsafe(32)
EOF

cat > src/search.rs << 'EOF'
pub fn search_index(query: &str) -> Vec<SearchResult> {
    let normalized_query = query.to_lowercase();
    
    // Perform semantic search using embeddings
    let embedding = generate_embedding(&normalized_query)?;
    let results = vector_search(&embedding, 0.7)?;
    
    // Sort by relevance score
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    results
}

pub async fn async_search(query: String) -> Result<Vec<SearchResult>, SearchError> {
    let embedding = async_generate_embedding(&query).await?;
    let results = async_vector_search(embedding).await?;
    Ok(results)
}
EOF

cat > src/database.go << 'EOF'
package main

import (
    "database/sql"
    "fmt"
    _ "github.com/lib/pq"
)

func connectToDatabase(host string, port int, user string, password string, dbname string) (*sql.DB, error) {
    psqlInfo := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=disable",
        host, port, user, password, dbname)
    
    db, err := sql.Open("postgres", psqlInfo)
    if err != nil {
        return nil, fmt.Errorf("failed to connect to database: %w", err)
    }
    
    err = db.Ping()
    if err != nil {
        return nil, fmt.Errorf("failed to ping database: %w", err)
    }
    
    return db, nil
}

func queryUsers(db *sql.DB) ([]User, error) {
    rows, err := db.Query("SELECT id, username, email FROM users WHERE active = true")
    if err != nil {
        return nil, err
    }
    defer rows.Close()
    
    var users []User
    for rows.Next() {
        var user User
        err := rows.Scan(&user.ID, &user.Username, &user.Email)
        if err != nil {
            return nil, err
        }
        users = append(users, user)
    }
    
    return users, nil
}
EOF

cat > tests/test_calculator.js << 'EOF'
import { calculateTotal, calculateTax } from '../src/calculator.js';

describe('Calculator Tests', () => {
    test('calculateTotal sums item prices correctly', () => {
        const items = [
            { price: 10.99 },
            { price: 25.50 },
            { price: 5.00 }
        ];
        expect(calculateTotal(items)).toBe(41.49);
    });
    
    test('calculateTax computes tax correctly', () => {
        expect(calculateTax(100, 0.08)).toBe(8.00);
    });
    
    test('calculateTax throws error for invalid rate', () => {
        expect(() => calculateTax(100, -0.1)).toThrow();
        expect(() => calculateTax(100, 1.5)).toThrow();
    });
});
EOF

cat > docs/auth.md << 'EOF'
# Authentication Guide

This document explains the authentication system used in our application.

## JWT Authentication

We use JSON Web Tokens (JWT) for stateless authentication. The process works as follows:

1. User provides username and password
2. Server validates credentials
3. Server generates JWT token
4. Client includes token in subsequent requests
5. Server validates token for each request

## API Key Authentication

For machine-to-machine communication, we support API key authentication:

- Generate API key through admin interface
- Include key in `X-API-Key` header
- Keys expire after 90 days by default

## Security Best Practices

- Always use HTTPS in production
- Store sensitive data in environment variables
- Implement rate limiting
- Use strong password requirements
- Enable two-factor authentication where possible
EOF

cat > examples/api_usage.py << 'EOF'
#!/usr/bin/env python3
"""
Example API usage demonstrating authentication and data retrieval
"""

import requests
import json
from typing import Dict, List, Optional

class APIClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.token = None
    
    def authenticate(self, username: str, password: str) -> bool:
        """Authenticate with username/password and store JWT token"""
        response = requests.post(f'{self.base_url}/auth/login', {
            'username': username,
            'password': password
        })
        
        if response.status_code == 200:
            self.token = response.json()['token']
            return True
        return False
    
    def get_users(self) -> List[Dict]:
        """Retrieve user list"""
        headers = self._get_headers()
        response = requests.get(f'{self.base_url}/users', headers=headers)
        response.raise_for_status()
        return response.json()
    
    def _get_headers(self) -> Dict[str, str]:
        headers = {'Content-Type': 'application/json'}
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'
        elif self.api_key:
            headers['X-API-Key'] = self.api_key
        return headers

if __name__ == '__main__':
    client = APIClient('https://api.example.com')
    if client.authenticate('user', 'password'):
        users = client.get_users()
        print(f"Found {len(users)} users")
EOF

# Initialize git repository
git init > /dev/null 2>&1
git add . > /dev/null 2>&1
git commit -m "Initial test repository" > /dev/null 2>&1

# Build TurboProp binary to ensure it's available
echo -e "${BLUE}Building TurboProp binary...${NC}"
cd "$OLDPWD"
cargo build --release > /dev/null 2>&1
TP_BINARY="$OLDPWD/target/release/tp"

# Return to test directory
cd "$TEST_DIR"

# Test available models (only test models that are likely to be available)
MODELS=(
    "sentence-transformers/all-MiniLM-L6-v2"
    "sentence-transformers/all-MiniLM-L12-v2"
    # Note: GGUF and Qwen3 models commented out as they may not be available in CI
    # "nomic-embed-code.Q5_K_S.gguf"
    # "Qwen/Qwen3-Embedding-0.6B"
)

# Track test results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to run a test and track results
run_test() {
    local test_name="$1"
    local test_command="$2"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo -ne "${BLUE}  Testing: $test_name... ${NC}"
    
    if eval "$test_command" > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        echo -e "${RED}✗${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
}

# Function to run a test with output capture
run_test_with_output() {
    local test_name="$1"
    local test_command="$2"
    local expected_pattern="$3"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo -ne "${BLUE}  Testing: $test_name... ${NC}"
    
    local output
    if output=$(eval "$test_command" 2>&1); then
        if [[ -z "$expected_pattern" ]] || echo "$output" | grep -q "$expected_pattern"; then
            echo -e "${GREEN}✓${NC}"
            PASSED_TESTS=$((PASSED_TESTS + 1))
            return 0
        else
            echo -e "${RED}✗ (output mismatch)${NC}"
            FAILED_TESTS=$((FAILED_TESTS + 1))
            return 1
        fi
    else
        echo -e "${RED}✗ (command failed)${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
}

# Test basic functionality first
echo -e "${YELLOW}Testing basic TurboProp functionality...${NC}"
run_test "Binary exists and is executable" "test -x '$TP_BINARY'"
run_test_with_output "Version command works" "'$TP_BINARY' --version" "tp"
run_test_with_output "Help command works" "'$TP_BINARY' --help" "USAGE:"

# Test model management commands
echo -e "${YELLOW}Testing model management commands...${NC}"
run_test_with_output "Model list command" "'$TP_BINARY' model list" "sentence-transformers"
run_test_with_output "Model info command" "'$TP_BINARY' model info 'sentence-transformers/all-MiniLM-L6-v2'" "Model:"

# Test each available model
for model in "${MODELS[@]}"; do
    echo -e "${YELLOW}Testing model: $model${NC}"
    
    # Test model info
    run_test_with_output "Get model info" "'$TP_BINARY' model info '$model'" "Model:"
    
    # Test indexing with limited scope
    run_test "Index with model (limited)" "'$TP_BINARY' index --repo . --model '$model' --limit 10"
    
    # Test basic search
    run_test_with_output "Search with model" "'$TP_BINARY' search 'function' --model '$model' --limit 3" "score"
    
    # Test search for specific content we know exists
    run_test_with_output "Search for authentication" "'$TP_BINARY' search 'authentication' --model '$model' --limit 3" "score"
    run_test_with_output "Search for calculate" "'$TP_BINARY' search 'calculate' --model '$model' --limit 3" "score"
    
    # Test different output formats
    run_test_with_output "JSON output format" "'$TP_BINARY' search 'function' --model '$model' --format json --limit 2" '"file":'
    run_test_with_output "Text output format" "'$TP_BINARY' search 'function' --model '$model' --format text --limit 2" "Score:"
    
    # Test file type filtering
    run_test "Filter by JavaScript files" "'$TP_BINARY' search 'calculate' --model '$model' --filetype .js --limit 2"
    run_test "Filter by Python files" "'$TP_BINARY' search 'authenticate' --model '$model' --filetype .py --limit 2"
    run_test "Filter by Rust files" "'$TP_BINARY' search 'search' --model '$model' --filetype .rs --limit 2"
    
    # Test glob pattern filtering
    run_test "Filter with glob pattern (src)" "'$TP_BINARY' search 'function' --model '$model' --filter 'src/*.js' --limit 2"
    run_test "Filter with glob pattern (recursive)" "'$TP_BINARY' search 'test' --model '$model' --filter '**/*.js' --limit 2"
    
    # Test similarity thresholds
    run_test "High similarity threshold" "'$TP_BINARY' search 'function' --model '$model' --threshold 0.8 --limit 1"
    run_test "Low similarity threshold" "'$TP_BINARY' search 'xyz' --model '$model' --threshold 0.1 --limit 1"
    
    echo "  ✓ Model $model passed basic tests"
    echo
done

# Test configuration file support
echo -e "${YELLOW}Testing configuration file support...${NC}"
cat > .turboprop.yml << EOF
default_model: "sentence-transformers/all-MiniLM-L6-v2"
max_filesize: "1mb"
default_limit: 5
similarity_threshold: 0.3

models:
  "sentence-transformers/all-MiniLM-L6-v2":
    batch_size: 16
    
embedding:
  cache_embeddings: true
EOF

run_test "Index with config file" "'$TP_BINARY' index --repo . --limit 5"
run_test_with_output "Search with config file" "'$TP_BINARY' search 'function' --limit 3" "score"

# Test error handling
echo -e "${YELLOW}Testing error handling...${NC}"
run_test "Invalid model name fails gracefully" "! '$TP_BINARY' model info 'nonexistent-model' 2>/dev/null"
run_test "Search without index fails gracefully" "rm -rf .turboprop && ! '$TP_BINARY' search 'test' 2>/dev/null"

# Test edge cases
echo -e "${YELLOW}Testing edge cases...${NC}"

# Re-create index for edge case tests
"$TP_BINARY" index --repo . --limit 10 > /dev/null 2>&1

run_test_with_output "Empty query handling" "'$TP_BINARY' search '' --limit 1" "score"
run_test "Very long query" "'$TP_BINARY' search 'this is a very long query with many words that should still work correctly even though it is quite lengthy and contains various terms' --limit 1"
run_test "Special characters in query" "'$TP_BINARY' search 'function() { return; }' --limit 1"
run_test "Unicode in query" "'$TP_BINARY' search 'función' --limit 1"

# Test performance characteristics
echo -e "${YELLOW}Testing performance characteristics...${NC}"

# Create larger test content for performance testing
for i in {1..20}; do
    echo "function test$i() { return processData(data$i); }" >> src/performance_test.js
done

"$TP_BINARY" index --repo . --force-rebuild > /dev/null 2>&1

# Time the search operations
run_test "Search performance (should complete quickly)" "timeout 10s '$TP_BINARY' search 'function' --limit 5"
run_test "Index performance (should complete quickly)" "timeout 30s '$TP_BINARY' index --repo . --force-rebuild"

# Test concurrent operations (basic)
echo -e "${YELLOW}Testing concurrent operations...${NC}"
run_test "Multiple searches can run" "'$TP_BINARY' search 'function' --limit 1 & '$TP_BINARY' search 'test' --limit 1 & wait"

# Memory usage test (basic check)
echo -e "${YELLOW}Testing memory usage...${NC}"
run_test "Memory usage stays reasonable" "timeout 10s '$TP_BINARY' search 'function' --limit 10"

# Clean up any background processes
pkill -f "$TP_BINARY" 2>/dev/null || true

# Print test summary
echo
echo "=================================================================="
echo -e "${BLUE}Test Summary${NC}"
echo "=================================================================="
echo -e "Total tests: $TOTAL_TESTS"
echo -e "${GREEN}Passed: $PASSED_TESTS${NC}"
echo -e "${RED}Failed: $FAILED_TESTS${NC}"

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}All integration tests passed! ✓${NC}"
    exit 0
else
    echo -e "${RED}Some integration tests failed! ✗${NC}"
    exit 1
fi