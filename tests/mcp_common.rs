//! Common utilities for MCP testing

use serde_json::{json, Value};
use tempfile::TempDir;

use turboprop::config::TurboPropConfig;

/// Create a test repository with sample code files
pub fn create_test_repository() -> TempDir {
    let temp_dir = TempDir::new().unwrap();
    let root = temp_dir.path();

    // Create directory structure
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::create_dir_all(root.join("tests")).unwrap();
    std::fs::create_dir_all(root.join("docs")).unwrap();

    // Create sample Rust files with comprehensive content for testing
    std::fs::write(
        root.join("src/main.rs"),
        r#"//! Main application entry point

use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let config = load_config()?;
    let auth = setup_authentication(&config)?;
    
    run_application(auth).await
}

fn load_config() -> Result<Config, ConfigError> {
    Config::from_file("config.toml")
}

fn setup_authentication(config: &Config) -> Result<AuthService, AuthError> {
    AuthService::new(config.auth_config())
}

async fn run_application(auth: AuthService) -> Result<(), AppError> {
    let server = HttpServer::new(auth);
    server.listen("0.0.0.0:8080").await
}
"#,
    )
    .unwrap();

    std::fs::write(
        root.join("src/auth.rs"),
        r#"//! Authentication and authorization module

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: u64,
    pub username: String,
    pub email: String,
    pub role: UserRole,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UserRole {
    Admin,
    User,
    Guest,
}

pub struct AuthService {
    users: HashMap<String, User>,
    jwt_secret: String,
}

impl AuthService {
    pub fn new(config: AuthConfig) -> Result<Self, AuthError> {
        Ok(Self {
            users: HashMap::new(),
            jwt_secret: config.jwt_secret,
        })
    }
    
    pub fn authenticate_user(&self, username: &str, password: &str) -> Result<User, AuthError> {
        if let Some(user) = self.users.get(username) {
            if self.verify_password(password, &user.username) {
                Ok(user.clone())
            } else {
                Err(AuthError::InvalidCredentials)
            }
        } else {
            Err(AuthError::UserNotFound)
        }
    }
    
    pub fn generate_jwt_token(&self, user: &User) -> Result<String, AuthError> {
        // Mock JWT generation
        Ok(format!("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.{}.signature", user.id))
    }
    
    pub fn validate_jwt_token(&self, token: &str) -> Result<User, AuthError> {
        if token.starts_with("eyJ") {
            // Mock validation - in real code, properly verify JWT
            let user_id = self.extract_user_id_from_token(token)?;
            self.get_user_by_id(user_id)
        } else {
            Err(AuthError::InvalidToken)
        }
    }
    
    fn verify_password(&self, password: &str, username: &str) -> bool {
        // Mock password verification
        !password.is_empty() && !username.is_empty()
    }
    
    fn extract_user_id_from_token(&self, token: &str) -> Result<u64, AuthError> {
        // Mock extraction
        Ok(1)
    }
    
    fn get_user_by_id(&self, id: u64) -> Result<User, AuthError> {
        // Mock user lookup
        Ok(User {
            id,
            username: "test_user".to_string(),
            email: "test@example.com".to_string(),
            role: UserRole::User,
        })
    }
}

#[derive(Debug)]
pub enum AuthError {
    InvalidCredentials,
    UserNotFound,
    InvalidToken,
    TokenExpired,
}
"#,
    )
    .unwrap();

    std::fs::write(
        root.join("src/utils.rs"),
        r#"//! Utility functions and error handling

use std::fmt;

/// Calculate the total of a list of numbers
pub fn calculate_total(numbers: &[f64]) -> f64 {
    numbers.iter().sum()
}

/// Format error messages consistently
pub fn format_error_message(context: &str, error: &dyn std::error::Error) -> String {
    format!("Error in {}: {}", context, error)
}

/// Handle database connection errors
pub fn handle_database_error(error: DatabaseError) -> Result<(), AppError> {
    match error {
        DatabaseError::ConnectionFailed => {
            log::error!("Database connection failed");
            Err(AppError::DatabaseUnavailable)
        }
        DatabaseError::QueryTimeout => {
            log::warn!("Database query timed out");
            Err(AppError::RequestTimeout)
        }
        DatabaseError::InvalidQuery => {
            log::error!("Invalid database query");
            Err(AppError::InternalError)
        }
    }
}

#[derive(Debug)]
pub enum DatabaseError {
    ConnectionFailed,
    QueryTimeout,
    InvalidQuery,
}

#[derive(Debug)]
pub enum AppError {
    DatabaseUnavailable,
    RequestTimeout,
    InternalError,
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AppError::DatabaseUnavailable => write!(f, "Database is unavailable"),
            AppError::RequestTimeout => write!(f, "Request timed out"),
            AppError::InternalError => write!(f, "Internal error occurred"),
        }
    }
}

impl std::error::Error for AppError {}
"#,
    )
    .unwrap();

    // Create API documentation
    std::fs::write(
        root.join("docs/API.md"),
        r#"# API Documentation

## Authentication Endpoints

### POST /auth/login

Authenticate a user with username and password.

**Request Body:**
```json
{
  "username": "string",
  "password": "string"
}
```

**Response:**
```json
{
  "token": "jwt_token_string",
  "user": {
    "id": 123,
    "username": "string",
    "email": "string",
    "role": "User"
  }
}
```

### POST /auth/validate

Validate a JWT token.

**Headers:**
```
Authorization: Bearer jwt_token_string
```

**Response:**
```json
{
  "valid": true,
  "user": {
    "id": 123,
    "username": "string",
    "email": "string",
    "role": "User"
  }
}
```

## Error Handling

All endpoints return consistent error responses:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable error message"
  }
}
```

Common error codes:
- `INVALID_CREDENTIALS`: Authentication failed
- `TOKEN_EXPIRED`: JWT token has expired
- `USER_NOT_FOUND`: User does not exist
- `INTERNAL_ERROR`: Server error occurred
"#,
    )
    .unwrap();

    // Create configuration file
    std::fs::write(
        root.join(".turboprop.yml"),
        r#"# TurboProp configuration for test repository
max_filesize: "2mb"
model: "sentence-transformers/all-MiniLM-L6-v2"
default_limit: 10
similarity_threshold: 0.3

file_discovery:
  include_patterns:
    - "**/*.rs"
    - "**/*.md"
  exclude_patterns:
    - "target/**"
    - ".git/**"
"#,
    )
    .unwrap();

    temp_dir
}

/// Create a test configuration
#[allow(dead_code)]
pub fn create_test_config() -> TurboPropConfig {
    let mut config = TurboPropConfig::default();
    config.embedding.model_name = "sentence-transformers/all-MiniLM-L6-v2".to_string();
    config.search.default_limit = 10;
    config
}

/// Create a standard initialize request for testing
#[allow(dead_code)]
pub fn create_initialize_request(id: u64) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            },
            "capabilities": {}
        }
    })
}

/// Create a tools/list request for testing
#[allow(dead_code)]
pub fn create_tools_list_request(id: u64) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": "tools/list"
    })
}

/// Create a search tool call request for testing
#[allow(dead_code)]
pub fn create_search_request(id: u64, query: &str, limit: Option<usize>) -> Value {
    let mut arguments = json!({
        "query": query
    });

    if let Some(limit) = limit {
        arguments["limit"] = json!(limit);
    }

    json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": "tools/call",
        "params": {
            "name": "semantic_search",
            "arguments": arguments
        }
    })
}

/// Assert that a JSON-RPC response is successful
#[allow(dead_code)]
pub fn assert_successful_response(response: &Value, expected_id: u64) {
    assert_eq!(response["jsonrpc"], "2.0");
    assert_eq!(response["id"], expected_id);
    assert!(response["error"].is_null());
    assert!(response["result"].is_object());
}

/// Assert that a JSON-RPC response contains an error
#[allow(dead_code)]
pub fn assert_error_response(response: &Value, expected_id: u64, expected_code: i32) {
    assert_eq!(response["jsonrpc"], "2.0");
    assert_eq!(response["id"], expected_id);
    assert!(response["result"].is_null());
    assert!(response["error"].is_object());
    assert_eq!(response["error"]["code"], expected_code);
    assert!(response["error"]["message"].is_string());
}

/// Helper to create invalid JSON-RPC requests for error testing
#[allow(dead_code)]
pub fn create_invalid_request(id: u64, error_type: &str) -> Value {
    match error_type {
        "invalid_version" => json!({
            "jsonrpc": "1.0", // Invalid version
            "id": id,
            "method": "initialize"
        }),
        "missing_method" => json!({
            "jsonrpc": "2.0",
            "id": id
            // Missing method field
        }),
        "null_id" => json!({
            "jsonrpc": "2.0",
            "id": null,
            "method": "tools/list"
        }),
        "malformed_params" => json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": "tools/call",
            "params": "not_an_object" // Should be object
        }),
        _ => json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": "unknown_method"
        }),
    }
}
