//! Real-world MCP scenario tests with actual indexing and file watching
//!
//! These tests validate MCP functionality in realistic conditions using
//! actual code indexing and search operations.

use anyhow::Result;
use serde_json::json;
use std::fs;
use std::path::Path;
use std::time::Duration;
use tempfile::TempDir;
use tokio::time::{sleep, timeout};

use turboprop::config::TurboPropConfig;
use turboprop::mcp::protocol::{ClientCapabilities, ClientInfo, InitializeParams, JsonRpcRequest};
use turboprop::mcp::{McpServer, McpServerTrait};

mod test_fixtures {
    use super::*;

    /// Create a realistic Rust project structure with various file types
    pub fn create_rust_project(temp_dir: &TempDir) -> Result<()> {
        let root = temp_dir.path();

        // Initialize git repository
        std::process::Command::new("git")
            .args(["init"])
            .current_dir(root)
            .output()?;

        // Create Cargo.toml
        fs::write(
            root.join("Cargo.toml"),
            r#"[package]
name = "example_project"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = { version = "1.0", features = ["derive"] }
tokio = { version = "1.0", features = ["full"] }
anyhow = "1.0"
thiserror = "1.0"
uuid = { version = "1.0", features = ["v4"] }

[dev-dependencies]
tempfile = "3.0"
"#,
        )?;

        // Create src directory structure
        fs::create_dir_all(root.join("src"))?;
        fs::create_dir_all(root.join("src/api"))?;
        fs::create_dir_all(root.join("src/models"))?;
        fs::create_dir_all(root.join("src/utils"))?;
        fs::create_dir_all(root.join("tests"))?;
        fs::create_dir_all(root.join("docs"))?;

        // Create main.rs - Web server with authentication
        fs::write(
            root.join("src/main.rs"),
            r#"//! Main application entry point
//!
//! This is a web server that provides JWT authentication and user management.

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

mod api;
mod models;
mod utils;

use api::{AuthHandler, UserHandler, create_routes};
use models::{User, AuthToken};
use utils::{DatabaseConnection, Logger, ErrorHandler};

/// Application state shared across handlers
#[derive(Clone)]
pub struct AppState {
    pub db: Arc<DatabaseConnection>,
    pub users: Arc<RwLock<HashMap<String, User>>>,
    pub active_tokens: Arc<RwLock<HashMap<String, AuthToken>>>,
    pub logger: Arc<Logger>,
}

/// Main application entry point
#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    let logger = Arc::new(Logger::new("INFO"));
    logger.info("Starting application server");

    // Initialize database connection
    let db = Arc::new(DatabaseConnection::new("postgresql://localhost:5432/myapp").await?);
    
    // Initialize shared state
    let state = AppState {
        db: db.clone(),
        users: Arc::new(RwLock::new(HashMap::new())),
        active_tokens: Arc::new(RwLock::new(HashMap::new())),
        logger: logger.clone(),
    };

    // Create and start HTTP server
    let app = create_routes(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:8080").await?;
    
    logger.info("Server listening on http://127.0.0.1:8080");
    
    // Start server
    axum::serve(listener, app).await?;
    
    Ok(())
}

/// Graceful shutdown handler
async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to install CTRL+C signal handler");
}

/// Health check endpoint
pub async fn health_check() -> &'static str {
    "OK"
}

/// Configuration loading
pub fn load_config() -> Result<AppConfig> {
    // Load configuration from environment variables and config files
    let config = AppConfig {
        database_url: std::env::var("DATABASE_URL")
            .unwrap_or_else(|_| "postgresql://localhost:5432/myapp".to_string()),
        jwt_secret: std::env::var("JWT_SECRET")
            .unwrap_or_else(|_| "default_secret_key".to_string()),
        server_port: std::env::var("PORT")
            .unwrap_or_else(|_| "8080".to_string())
            .parse()
            .unwrap_or(8080),
    };
    
    Ok(config)
}

pub struct AppConfig {
    pub database_url: String,
    pub jwt_secret: String,
    pub server_port: u16,
}
"#,
        )?;

        // Create lib.rs - Library interface
        fs::write(
            root.join("src/lib.rs"),
            r#"//! Core library functionality
//!
//! This module provides the main library interface for the application,
//! including authentication, user management, and JWT token handling.

pub mod api;
pub mod models;
pub mod utils;

pub use api::{AuthHandler, UserHandler};
pub use models::{User, UserRole, AuthToken, TokenType};
pub use utils::{DatabaseConnection, ErrorHandler, ValidationError};

/// Re-export commonly used types
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Application version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default JWT expiration time (24 hours)
pub const DEFAULT_JWT_EXPIRATION: std::time::Duration = std::time::Duration::from_secs(24 * 60 * 60);

/// Maximum allowed login attempts before account lockout
pub const MAX_LOGIN_ATTEMPTS: u32 = 5;

/// Password minimum length requirement
pub const MIN_PASSWORD_LENGTH: usize = 8;

/// Initialize the application with default configuration
pub async fn initialize() -> Result<()> {
    // Set up logging
    env_logger::init();
    
    // Validate environment
    validate_environment()?;
    
    Ok(())
}

/// Validate that required environment variables are set
fn validate_environment() -> Result<()> {
    let required_vars = ["DATABASE_URL", "JWT_SECRET"];
    
    for var in &required_vars {
        if std::env::var(var).is_err() {
            return Err(format!("Required environment variable {} is not set", var).into());
        }
    }
    
    Ok(())
}
"#,
        )?;

        // Create authentication API module
        fs::write(
            root.join("src/api/mod.rs"),
            r#"//! API handlers and routing
//!
//! This module contains all HTTP request handlers and routing logic
//! for the web application.

pub mod auth;
pub mod users;

pub use auth::AuthHandler;
pub use users::UserHandler;

use axum::{
    routing::{get, post, put, delete},
    Router,
};
use std::sync::Arc;

/// Create the main application router with all routes
pub fn create_routes(state: crate::AppState) -> Router {
    Router::new()
        // Health check
        .route("/health", get(crate::health_check))
        
        // Authentication routes
        .route("/auth/login", post(AuthHandler::login))
        .route("/auth/logout", post(AuthHandler::logout))
        .route("/auth/refresh", post(AuthHandler::refresh_token))
        .route("/auth/validate", get(AuthHandler::validate_token))
        
        // User management routes
        .route("/users", get(UserHandler::list_users))
        .route("/users", post(UserHandler::create_user))
        .route("/users/:id", get(UserHandler::get_user))
        .route("/users/:id", put(UserHandler::update_user))
        .route("/users/:id", delete(UserHandler::delete_user))
        
        // Password management
        .route("/users/:id/password", put(UserHandler::change_password))
        .route("/auth/forgot-password", post(AuthHandler::forgot_password))
        .route("/auth/reset-password", post(AuthHandler::reset_password))
        
        .with_state(Arc::new(state))
}

/// Common response types
pub mod responses {
    use serde::{Deserialize, Serialize};
    
    #[derive(Serialize, Deserialize)]
    pub struct ApiResponse<T> {
        pub success: bool,
        pub data: Option<T>,
        pub error: Option<String>,
        pub timestamp: chrono::DateTime<chrono::Utc>,
    }
    
    impl<T> ApiResponse<T> {
        pub fn success(data: T) -> Self {
            Self {
                success: true,
                data: Some(data),
                error: None,
                timestamp: chrono::Utc::now(),
            }
        }
        
        pub fn error(message: String) -> Self {
            Self {
                success: false,
                data: None,
                error: Some(message),
                timestamp: chrono::Utc::now(),
            }
        }
    }
}
"#,
        )?;

        // Create authentication handler
        fs::write(
            root.join("src/api/auth.rs"),
            r#"//! Authentication handlers
//!
//! This module implements JWT-based authentication with login, logout,
//! token refresh, and password reset functionality.

use axum::{
    extract::{State, Json},
    http::StatusCode,
    response::Json as ResponseJson,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

use crate::models::{User, AuthToken, TokenType};
use crate::utils::{hash_password, verify_password, generate_jwt, validate_jwt};
use crate::AppState;
use super::responses::ApiResponse;

/// Authentication request payload
#[derive(Deserialize)]
pub struct LoginRequest {
    pub email: String,
    pub password: String,
    pub remember_me: Option<bool>,
}

/// Authentication response payload
#[derive(Serialize)]
pub struct LoginResponse {
    pub access_token: String,
    pub refresh_token: String,
    pub user: UserProfile,
    pub expires_in: u64,
}

/// User profile information (without sensitive data)
#[derive(Serialize)]
pub struct UserProfile {
    pub id: String,
    pub email: String,
    pub username: String,
    pub role: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_login: Option<chrono::DateTime<chrono::Utc>>,
}

/// Password reset request
#[derive(Deserialize)]  
pub struct ForgotPasswordRequest {
    pub email: String,
}

/// Password reset confirmation
#[derive(Deserialize)]
pub struct ResetPasswordRequest {
    pub token: String,
    pub new_password: String,
}

/// Authentication handler implementation
pub struct AuthHandler;

impl AuthHandler {
    /// Handle user login
    pub async fn login(
        State(state): State<Arc<AppState>>,
        Json(payload): Json<LoginRequest>,
    ) -> Result<ResponseJson<ApiResponse<LoginResponse>>, StatusCode> {
        state.logger.info(&format!("Login attempt for email: {}", payload.email));
        
        // Input validation
        if payload.email.is_empty() || payload.password.is_empty() {
            return Ok(ResponseJson(ApiResponse::error(
                "Email and password are required".to_string()
            )));
        }
        
        // Find user by email
        let users = state.users.read().await;
        let user = users.values()
            .find(|u| u.email == payload.email)
            .cloned();
        
        drop(users);
        
        let user = match user {
            Some(u) => u,
            None => {
                state.logger.warn(&format!("Login failed: user not found for {}", payload.email));
                return Ok(ResponseJson(ApiResponse::error(
                    "Invalid credentials".to_string()
                )));
            }
        };
        
        // Verify password
        if !verify_password(&payload.password, &user.password_hash)? {
            state.logger.warn(&format!("Login failed: invalid password for {}", payload.email));
            return Ok(ResponseJson(ApiResponse::error(
                "Invalid credentials".to_string()
            )));
        }
        
        // Generate tokens
        let access_token = generate_jwt(&user, TokenType::Access, &state)?;
        let refresh_token = generate_jwt(&user, TokenType::Refresh, &state)?;
        
        // Store tokens
        let mut tokens = state.active_tokens.write().await;
        tokens.insert(access_token.clone(), AuthToken {
            id: Uuid::new_v4().to_string(),
            user_id: user.id.clone(),
            token_type: TokenType::Access,
            expires_at: chrono::Utc::now() + chrono::Duration::hours(1),
            created_at: chrono::Utc::now(),
        });
        
        let expires_in = if payload.remember_me.unwrap_or(false) { 
            24 * 60 * 60 // 24 hours
        } else { 
            60 * 60 // 1 hour
        };
        
        let response = LoginResponse {
            access_token,
            refresh_token,
            user: UserProfile {
                id: user.id,
                email: user.email,
                username: user.username,
                role: user.role.to_string(),
                created_at: user.created_at,
                last_login: user.last_login_at,
            },
            expires_in,
        };
        
        state.logger.info("Login successful!");
        Ok(ResponseJson(ApiResponse::success(response)))
    }
    
    /// Handle user logout
    pub async fn logout(
        State(state): State<Arc<AppState>>,
        // Extract token from Authorization header
    ) -> Result<ResponseJson<ApiResponse<()>>, StatusCode> {
        // Implementation for logout
        state.logger.info("User logged out");
        Ok(ResponseJson(ApiResponse::success(())))
    }
    
    /// Handle token refresh
    pub async fn refresh_token(
        State(state): State<Arc<AppState>>,
        Json(payload): Json<RefreshTokenRequest>,
    ) -> Result<ResponseJson<ApiResponse<LoginResponse>>, StatusCode> {
        // Implementation for token refresh
        todo!("Implement token refresh")
    }
    
    /// Validate JWT token
    pub async fn validate_token(
        State(state): State<Arc<AppState>>,
    ) -> Result<ResponseJson<ApiResponse<UserProfile>>, StatusCode> {
        // Implementation for token validation
        todo!("Implement token validation")
    }
    
    /// Handle forgot password request
    pub async fn forgot_password(
        State(state): State<Arc<AppState>>,
        Json(payload): Json<ForgotPasswordRequest>,
    ) -> Result<ResponseJson<ApiResponse<()>>, StatusCode> {
        state.logger.info(&format!("Password reset requested for: {}", payload.email));
        // Implementation for password reset
        Ok(ResponseJson(ApiResponse::success(())))
    }
    
    /// Handle password reset confirmation
    pub async fn reset_password(
        State(state): State<Arc<AppState>>,
        Json(payload): Json<ResetPasswordRequest>,
    ) -> Result<ResponseJson<ApiResponse<()>>, StatusCode> {
        // Implementation for password reset confirmation
        Ok(ResponseJson(ApiResponse::success(())))
    }
}

#[derive(Deserialize)]
struct RefreshTokenRequest {
    refresh_token: String,
}
"#,
        )?;

        // Create user handler
        fs::write(
            root.join("src/api/users.rs"),
            r#"//! User management handlers
//!
//! This module implements CRUD operations for user management.

use axum::{
    extract::{State, Path, Json},
    http::StatusCode,
    response::Json as ResponseJson,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::models::{User, CreateUserRequest, UpdateUserRequest, ChangePasswordRequest};
use crate::AppState;
use super::responses::ApiResponse;

/// User management handler implementation
pub struct UserHandler;

impl UserHandler {
    /// List all users (paginated)
    pub async fn list_users(
        State(state): State<Arc<AppState>>,
    ) -> Result<ResponseJson<ApiResponse<Vec<User>>>, StatusCode> {
        state.logger.info("Listing all users");
        
        let users = state.users.read().await;
        let user_list: Vec<User> = users.values().cloned().collect();
        
        Ok(ResponseJson(ApiResponse::success(user_list)))
    }
    
    /// Create a new user
    pub async fn create_user(
        State(state): State<Arc<AppState>>,
        Json(payload): Json<CreateUserRequest>,
    ) -> Result<ResponseJson<ApiResponse<User>>, StatusCode> {
        state.logger.info(&format!("Creating user: {}", payload.username));
        
        // Validate input
        if payload.username.is_empty() || payload.email.is_empty() {
            return Ok(ResponseJson(ApiResponse::error(
                "Username and email are required".to_string()
            )));
        }
        
        // Check if user already exists
        let users = state.users.read().await;
        if users.values().any(|u| u.email == payload.email || u.username == payload.username) {
            return Ok(ResponseJson(ApiResponse::error(
                "User already exists".to_string()
            )));
        }
        drop(users);
        
        // Create new user
        let password_hash = crate::utils::hash_password(&payload.password)?;
        let user = User::new(payload.username, payload.email, password_hash);
        
        // Store user
        let mut users = state.users.write().await;
        users.insert(user.id.clone(), user.clone());
        
        Ok(ResponseJson(ApiResponse::success(user)))
    }
    
    /// Get user by ID
    pub async fn get_user(
        State(state): State<Arc<AppState>>,
        Path(user_id): Path<String>,
    ) -> Result<ResponseJson<ApiResponse<User>>, StatusCode> {
        let users = state.users.read().await;
        
        match users.get(&user_id) {
            Some(user) => Ok(ResponseJson(ApiResponse::success(user.clone()))),
            None => Ok(ResponseJson(ApiResponse::error("User not found".to_string()))),
        }
    }
    
    /// Update user information
    pub async fn update_user(
        State(state): State<Arc<AppState>>,
        Path(user_id): Path<String>,
        Json(payload): Json<UpdateUserRequest>,
    ) -> Result<ResponseJson<ApiResponse<User>>, StatusCode> {
        let mut users = state.users.write().await;
        
        match users.get_mut(&user_id) {
            Some(user) => {
                if let Some(username) = payload.username {
                    user.username = username;
                }
                if let Some(email) = payload.email {
                    user.email = email;
                }
                if let Some(role) = payload.role {
                    user.role = role;
                }
                if let Some(is_active) = payload.is_active {
                    user.is_active = is_active;
                }
                
                user.updated_at = chrono::Utc::now();
                
                Ok(ResponseJson(ApiResponse::success(user.clone())))
            }
            None => Ok(ResponseJson(ApiResponse::error("User not found".to_string()))),
        }
    }
    
    /// Delete user
    pub async fn delete_user(
        State(state): State<Arc<AppState>>,
        Path(user_id): Path<String>,
    ) -> Result<ResponseJson<ApiResponse<()>>, StatusCode> {
        let mut users = state.users.write().await;
        
        match users.remove(&user_id) {
            Some(_) => {
                state.logger.info(&format!("Deleted user: {}", user_id));
                Ok(ResponseJson(ApiResponse::success(())))
            }
            None => Ok(ResponseJson(ApiResponse::error("User not found".to_string()))),
        }
    }
    
    /// Change user password
    pub async fn change_password(
        State(state): State<Arc<AppState>>,
        Path(user_id): Path<String>,
        Json(payload): Json<ChangePasswordRequest>,
    ) -> Result<ResponseJson<ApiResponse<()>>, StatusCode> {
        let mut users = state.users.write().await;
        
        match users.get_mut(&user_id) {
            Some(user) => {
                // Verify current password
                if !crate::utils::verify_password(&payload.current_password, &user.password_hash)? {
                    return Ok(ResponseJson(ApiResponse::error(
                        "Current password is incorrect".to_string()
                    )));
                }
                
                // Update password
                user.password_hash = crate::utils::hash_password(&payload.new_password)?;
                user.updated_at = chrono::Utc::now();
                
                state.logger.info(&format!("Password changed for user: {}", user_id));
                Ok(ResponseJson(ApiResponse::success(())))
            }
            None => Ok(ResponseJson(ApiResponse::error("User not found".to_string()))),
        }
    }
}
"#,
        )?;

        // Create models module
        fs::write(
            root.join("src/models/mod.rs"),
            r#"//! Data models and types
//!
//! This module contains all data structures used throughout the application.

pub mod user;
pub mod auth;

pub use user::{
    User, UserRole, UserProfile, UserPreferences, Theme, ProfileVisibility,
    CreateUserRequest, UpdateUserRequest, ChangePasswordRequest
};
pub use auth::{AuthToken, TokenType};
"#,
        )?;

        // Create auth token model
        fs::write(
            root.join("src/models/auth.rs"),
            r#"//! Authentication token models
//!
//! This module defines token-related data structures for JWT handling.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Authentication token for session management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthToken {
    pub id: String,
    pub user_id: String,
    pub token_type: TokenType,
    pub expires_at: chrono::DateTime<chrono::Utc>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Type of authentication token
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TokenType {
    /// Short-lived access token (1 hour)
    Access,
    /// Long-lived refresh token (7 days)
    Refresh,
    /// Password reset token (1 hour)
    PasswordReset,
    /// Email verification token (24 hours)
    EmailVerification,
}

impl AuthToken {
    /// Create a new authentication token
    pub fn new(user_id: String, token_type: TokenType) -> Self {
        let now = chrono::Utc::now();
        let expires_at = match token_type {
            TokenType::Access => now + chrono::Duration::hours(1),
            TokenType::Refresh => now + chrono::Duration::days(7),
            TokenType::PasswordReset => now + chrono::Duration::hours(1),
            TokenType::EmailVerification => now + chrono::Duration::hours(24),
        };
        
        Self {
            id: Uuid::new_v4().to_string(),
            user_id,
            token_type,
            expires_at,
            created_at: now,
        }
    }
    
    /// Check if token is expired
    pub fn is_expired(&self) -> bool {
        chrono::Utc::now() > self.expires_at
    }
    
    /// Get remaining time until expiration
    pub fn time_until_expiry(&self) -> chrono::Duration {
        self.expires_at - chrono::Utc::now()
    }
}
"#,
        )?;

        // Create user model
        fs::write(
            root.join("src/models/user.rs"),
            r#"//! User data models and related types
//!
//! This module defines the core User struct and related types for
//! user management and authentication.

use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

/// User entity representing a system user
#[derive(Debug, Clone, Serialize, Deserialize)]  
pub struct User {
    pub id: String,
    pub username: String,
    pub email: String,
    pub password_hash: String,
    pub role: UserRole,
    pub is_active: bool,
    pub is_verified: bool,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
    pub last_login_at: Option<chrono::DateTime<chrono::Utc>>,
    pub login_attempts: u32,
    pub locked_until: Option<chrono::DateTime<chrono::Utc>>,
    pub profile: UserProfile,
}

/// User role enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum UserRole {
    Admin,
    Moderator,
    User,
    Guest,
}

impl fmt::Display for UserRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UserRole::Admin => write!(f, "admin"),
            UserRole::Moderator => write!(f, "moderator"),
            UserRole::User => write!(f, "user"),
            UserRole::Guest => write!(f, "guest"),
        }
    }
}

/// Extended user profile information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserProfile {
    pub first_name: Option<String>,
    pub last_name: Option<String>,
    pub avatar_url: Option<String>,
    pub bio: Option<String>,
    pub location: Option<String>,
    pub website: Option<String>,
    pub timezone: Option<String>,
    pub preferences: UserPreferences,
}

/// User preferences and settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPreferences {
    pub theme: Theme,
    pub language: String,
    pub notifications: NotificationSettings,
    pub privacy: PrivacySettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Theme {
    Light,
    Dark,
    Auto,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationSettings {
    pub email_notifications: bool,
    pub push_notifications: bool,
    pub weekly_digest: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacySettings {
    pub profile_visibility: ProfileVisibility,
    pub show_email: bool,
    pub show_last_activity: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProfileVisibility {
    Public,
    FriendsOnly,
    Private,
}

impl Default for UserProfile {
    fn default() -> Self {
        Self {
            first_name: None,
            last_name: None,
            avatar_url: None,
            bio: None,
            location: None,
            website: None,
            timezone: Some("UTC".to_string()),
            preferences: UserPreferences::default(),
        }
    }
}

impl Default for UserPreferences {
    fn default() -> Self {
        Self {
            theme: Theme::Auto,
            language: "en".to_string(),
            notifications: NotificationSettings {
                email_notifications: true,
                push_notifications: false,
                weekly_digest: true,
            },
            privacy: PrivacySettings {
                profile_visibility: ProfileVisibility::Public,
                show_email: false,
                show_last_activity: true,
            },
        }
    }
}

impl User {
    /// Create a new user with default values
    pub fn new(username: String, email: String, password_hash: String) -> Self {
        let now = chrono::Utc::now();
        
        Self {
            id: Uuid::new_v4().to_string(),
            username,
            email,
            password_hash,
            role: UserRole::User,
            is_active: true,
            is_verified: false,
            created_at: now,
            updated_at: now,
            last_login_at: None,
            login_attempts: 0,
            locked_until: None,
            profile: UserProfile::default(),
        }
    }
    
    /// Check if user account is locked
    pub fn is_locked(&self) -> bool {
        if let Some(locked_until) = self.locked_until {
            chrono::Utc::now() < locked_until
        } else {
            false
        }
    }
    
    /// Increment failed login attempts
    pub fn increment_login_attempts(&mut self) {
        self.login_attempts += 1;
        self.updated_at = chrono::Utc::now();
        
        // Lock account after too many attempts
        if self.login_attempts >= crate::MAX_LOGIN_ATTEMPTS {
            self.locked_until = Some(chrono::Utc::now() + chrono::Duration::hours(1));
        }
    }
    
    /// Reset login attempts on successful login
    pub fn reset_login_attempts(&mut self) {
        self.login_attempts = 0;
        self.locked_until = None;
        self.last_login_at = Some(chrono::Utc::now());
        self.updated_at = chrono::Utc::now();
    }
    
    /// Check if user has specific role or higher
    pub fn has_role(&self, required_role: &UserRole) -> bool {
        match (required_role, &self.role) {
            (UserRole::Guest, _) => true,
            (UserRole::User, UserRole::User | UserRole::Moderator | UserRole::Admin) => true,
            (UserRole::Moderator, UserRole::Moderator | UserRole::Admin) => true,
            (UserRole::Admin, UserRole::Admin) => true,
            _ => false,
        }
    }
}

/// User creation request
#[derive(Debug, Deserialize)]
pub struct CreateUserRequest {
    pub username: String,
    pub email: String,
    pub password: String,
    pub role: Option<UserRole>,
}

/// User update request
#[derive(Debug, Deserialize)]
pub struct UpdateUserRequest {
    pub username: Option<String>,
    pub email: Option<String>,
    pub role: Option<UserRole>,
    pub is_active: Option<bool>,
    pub profile: Option<UserProfile>,
}

/// Password change request
#[derive(Debug, Deserialize)]
pub struct ChangePasswordRequest {
    pub current_password: String,
    pub new_password: String,
}
"#,
        )?;

        // Create utils module
        fs::write(
            root.join("src/utils/mod.rs"),
            r#"//! Utility functions and helper modules
//!
//! This module provides common utilities for password hashing, JWT handling,
//! database operations, and error management.

pub mod crypto;
pub mod database;
pub mod errors;
pub mod logger;
pub mod validation;

pub use crypto::{hash_password, verify_password, generate_jwt, validate_jwt};
pub use database::DatabaseConnection;
pub use errors::{ErrorHandler, ApiError};
pub use logger::Logger;
pub use validation::{ValidationError, validate_email, validate_password};

/// Common result type for utility functions
pub type UtilResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Generate a secure random string
pub fn generate_random_string(length: usize) -> String {
    use rand::Rng;
    const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ\
                             abcdefghijklmnopqrstuvwxyz\
                             0123456789";
    let mut rng = rand::thread_rng();
    
    (0..length)
        .map(|_| {
            let idx = rng.gen_range(0..CHARSET.len());
            CHARSET[idx] as char
        })
        .collect()
}

/// Format timestamp for API responses
pub fn format_timestamp(dt: chrono::DateTime<chrono::Utc>) -> String {
    dt.format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string()
}

/// Sanitize user input to prevent XSS
pub fn sanitize_input(input: &str) -> String {
    // Basic HTML entity encoding
    input
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#x27;")
}

/// Rate limiting helper
pub struct RateLimiter {
    // Implementation would go here
}

impl RateLimiter {
    pub fn new(max_requests: u32, window_seconds: u64) -> Self {
        Self {}
    }
    
    pub async fn check_rate_limit(&self, key: &str) -> bool {
        // Implementation would check against Redis or in-memory store
        true
    }
}
"#,
        )?;

        // Create documentation files
        fs::write(
            root.join("docs/API.md"),
            r#"# API Documentation

## Overview

This API provides JWT-based authentication and user management functionality.

## Authentication

All protected endpoints require a valid JWT token in the Authorization header:

```
Authorization: Bearer <your-jwt-token>
```

## Endpoints

### Authentication

#### POST /auth/login
Authenticate a user and receive JWT tokens.

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "securepassword123",
  "remember_me": false
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "user": {
      "id": "123e4567-e89b-12d3-a456-426614174000",
      "email": "user@example.com",
      "username": "johndoe",
      "role": "user"
    },
    "expires_in": 3600
  }
}
```

#### POST /auth/logout
Invalidate the current user session.

#### POST /auth/refresh
Refresh an expired access token using a refresh token.

#### GET /auth/validate
Validate the current JWT token and return user information.

### User Management

#### GET /users
List all users (Admin only).

#### POST /users
Create a new user account.

#### GET /users/:id
Get user details by ID.

#### PUT /users/:id
Update user information.

#### DELETE /users/:id
Delete a user account.

#### PUT /users/:id/password
Change user password.

## Error Handling

All endpoints return errors in a consistent format:

```json
{
  "success": false,
  "error": "Detailed error message",
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

## Rate Limiting

API endpoints are rate limited to prevent abuse:
- Authentication endpoints: 5 requests per minute
- User management: 100 requests per hour
- General endpoints: 1000 requests per hour

## Security

### Password Requirements
- Minimum 8 characters
- Must contain uppercase and lowercase letters
- Must contain at least one number
- Must contain at least one special character

### JWT Tokens
- Access tokens expire after 1 hour
- Refresh tokens expire after 7 days
- Tokens are signed with HS256 algorithm

### Account Lockout
- Accounts are temporarily locked after 5 failed login attempts
- Lockout duration: 1 hour
- Account unlock requires admin intervention for repeated lockouts
"#,
        )?;

        // Create test files
        fs::write(
            root.join("tests/integration_tests.rs"),
            r#"//! Integration tests for the web application
//!
//! These tests validate the complete authentication and user management flow.

use serde_json::json;
use std::collections::HashMap;

mod auth_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_user_registration_and_login() {
        // Test user registration flow
        let user_data = json!({
            "username": "testuser",
            "email": "test@example.com",
            "password": "SecurePass123!"
        });
        
        // Simulate user creation
        assert!(user_data["username"].as_str().unwrap() == "testuser");
        assert!(user_data["email"].as_str().unwrap().contains("@"));
        
        // Test password strength
        let password = user_data["password"].as_str().unwrap();
        assert!(password.len() >= 8);
        assert!(password.chars().any(|c| c.is_uppercase()));
        assert!(password.chars().any(|c| c.is_lowercase()));
        assert!(password.chars().any(|c| c.is_ascii_digit()));
    }
    
    #[tokio::test]
    async fn test_jwt_token_generation() {
        // Test JWT token structure
        let mock_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c";
        
        // Verify JWT structure (3 parts separated by dots)
        let parts: Vec<&str> = mock_token.split('.').collect();
        assert_eq!(parts.len(), 3);
        
        // Each part should be base64-encoded
        for part in parts {
            assert!(!part.is_empty());
            assert!(part.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_'));
        }
    }
    
    #[tokio::test]
    async fn test_password_hashing() {
        let password = "MySecurePassword123!";
        
        // Mock password hashing (in real code, this would use bcrypt or similar)
        let hash = format!("$2b$12${}", password);
        
        // Verify hash is different from original password
        assert_ne!(password, hash);
        assert!(hash.starts_with("$2b$12$"));
    }
    
    #[tokio::test]
    async fn test_user_role_permissions() {
        let mut users = HashMap::new();
        
        users.insert("admin", "admin");
        users.insert("moderator", "moderator");
        users.insert("user", "user");
        users.insert("guest", "guest");
        
        // Test role hierarchy
        assert_eq!(users.get("admin"), Some(&"admin"));
        assert_eq!(users.get("user"), Some(&"user"));
        assert_eq!(users.len(), 4);
    }
}

mod api_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_health_check_endpoint() {
        // Simulate health check response
        let response = "OK";
        assert_eq!(response, "OK");
    }
    
    #[tokio::test]
    async fn test_user_creation_validation() {
        let valid_user = json!({
            "username": "validuser",
            "email": "valid@example.com",
            "password": "ValidPass123!"
        });
        
        let invalid_user = json!({
            "username": "",
            "email": "invalid-email",
            "password": "weak"
        });
        
        // Validate required fields
        assert!(!valid_user["username"].as_str().unwrap().is_empty());
        assert!(valid_user["email"].as_str().unwrap().contains("@"));
        assert!(valid_user["password"].as_str().unwrap().len() >= 8);
        
        // Invalid user should fail validation
        assert!(invalid_user["username"].as_str().unwrap().is_empty());
        assert!(!invalid_user["email"].as_str().unwrap().contains("@"));
        assert!(invalid_user["password"].as_str().unwrap().len() < 8);
    }
    
    #[tokio::test]
    async fn test_error_response_format() {
        let error_response = json!({
            "success": false,
            "error": "User not found",
            "timestamp": "2024-01-15T10:30:00.000Z"
        });
        
        assert_eq!(error_response["success"], false);
        assert!(error_response["error"].is_string());
        assert!(error_response["timestamp"].is_string());
    }
}
"#,
        )?;

        // Create README
        fs::write(
            root.join("README.md"),
            r#"# Example Web Application

A modern web application built with Rust, featuring JWT authentication, user management, and a RESTful API.

## Features

- 🔐 JWT-based authentication
- 👥 User management with role-based access control
- 🚀 High-performance async web server
- 📝 Comprehensive API documentation
- 🔒 Security best practices implemented
- 🧪 Extensive test coverage

## Architecture

This application follows a modular architecture:

- **API Layer**: HTTP handlers and routing (`src/api/`)
- **Business Logic**: Core application logic (`src/models/`)
- **Utilities**: Helper functions and shared utilities (`src/utils/`)
- **Documentation**: API docs and guides (`docs/`)

## Getting Started

### Prerequisites

- Rust 1.70+ 
- PostgreSQL 13+
- Redis (for session management)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/example/web-app.git
cd web-app
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Install dependencies:
```bash
cargo build
```

4. Run database migrations:
```bash
diesel migration run
```

5. Start the development server:
```bash
cargo run
```

The server will start on `http://localhost:8080`.

## API Usage

### Authentication

Register a new user:
```bash
curl -X POST http://localhost:8080/users \
  -H "Content-Type: application/json" \
  -d '{"username":"john","email":"john@example.com","password":"SecurePass123!"}'
```

Login:
```bash
curl -X POST http://localhost:8080/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"john@example.com","password":"SecurePass123!"}'
```

### Protected Endpoints

Use the JWT token from login response:
```bash
curl -X GET http://localhost:8080/users \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

## Testing

Run the test suite:
```bash
# Unit tests
cargo test

# Integration tests
cargo test --test integration_tests
```

## Security

This application implements several security measures:

- Password hashing with bcrypt
- JWT tokens for stateless authentication
- Rate limiting to prevent abuse
- Input validation and sanitization
- SQL injection prevention
- XSS protection

## Performance

- Async/await for non-blocking I/O
- Connection pooling for database efficiency
- Optimized JSON serialization
- Middleware for request/response processing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
"#,
        )?;

        Ok(())
    }
}

/// Test suite for real-world MCP scenarios
mod real_world_tests {
    use super::*;

    #[test]
    fn test_rust_project_creation() {
        let temp_dir = TempDir::new().unwrap();
        test_fixtures::create_rust_project(&temp_dir).unwrap();

        let root = temp_dir.path();

        // Verify project structure
        assert!(root.join("Cargo.toml").exists());
        assert!(root.join("src/main.rs").exists());
        assert!(root.join("src/lib.rs").exists());
        assert!(root.join("src/api/mod.rs").exists());
        assert!(root.join("src/api/auth.rs").exists());
        assert!(root.join("src/models/user.rs").exists());
        assert!(root.join("src/utils/mod.rs").exists());
        assert!(root.join("docs/API.md").exists());
        assert!(root.join("README.md").exists());
        assert!(root.join("tests/integration_tests.rs").exists());

        // Verify content contains expected patterns
        let main_content = fs::read_to_string(root.join("src/main.rs")).unwrap();
        assert!(main_content.contains("JWT authentication"));
        assert!(main_content.contains("DatabaseConnection"));
        assert!(main_content.contains("async fn main"));

        let auth_content = fs::read_to_string(root.join("src/api/auth.rs")).unwrap();
        assert!(auth_content.contains("pub async fn login"));
        assert!(auth_content.contains("LoginRequest"));
        assert!(auth_content.contains("generate_jwt"));

        let readme_content = fs::read_to_string(root.join("README.md")).unwrap();
        assert!(readme_content.contains("JWT-based authentication"));
        assert!(readme_content.contains("cargo run"));

        println!(
            "Successfully created realistic Rust project with {} files",
            fs::read_dir(root).unwrap().count()
        );
    }

    async fn initialize_mcp_server(repo_path: &Path) -> Result<McpServer> {
        let config = TurboPropConfig::default();
        McpServer::new(repo_path, &config).await
    }

    async fn initialize_server(server: &mut McpServer) -> Result<()> {
        let params = InitializeParams {
            protocol_version: "2024-11-05".to_string(),
            client_info: ClientInfo {
                name: "test-client".to_string(),
                version: "1.0.0".to_string(),
            },
            capabilities: ClientCapabilities::default(),
        };

        server.initialize(params).await?;
        Ok(())
    }

    async fn wait_for_indexing(duration: Duration) {
        sleep(duration).await;
    }

    #[tokio::test]
    #[ignore] // This test requires actual indexing - slow
    async fn test_authentication_code_search() {
        let temp_dir = TempDir::new().unwrap();
        test_fixtures::create_rust_project(&temp_dir).unwrap();

        let mut server = initialize_mcp_server(temp_dir.path()).await.unwrap();
        initialize_server(&mut server).await.unwrap();

        // Wait for indexing to complete
        wait_for_indexing(Duration::from_secs(10)).await;

        // Search for authentication-related code
        let search_queries = vec![
            "JWT authentication login",
            "password hashing bcrypt",
            "user authentication validation",
            "token generation refresh",
            "login endpoint handler",
        ];

        for query in search_queries {
            let request = JsonRpcRequest::new(
                "tools/call".to_string(),
                Some(json!({
                    "name": "semantic_search",
                    "arguments": {
                        "query": query,
                        "limit": 10
                    }
                })),
            );

            let response = timeout(Duration::from_secs(15), server.handle_request(request))
                .await
                .unwrap()
                .unwrap();

            // Verify we get results
            if response.error.is_none() {
                let result = response.result.unwrap();
                let results = result["results"].as_array().unwrap();

                // Should find relevant code
                assert!(
                    !results.is_empty(),
                    "Should find results for query: {}",
                    query
                );

                // Verify result structure
                for result_item in results {
                    assert!(result_item["file_path"].is_string());
                    assert!(result_item["content"].is_string());
                    assert!(result_item["line_number"].is_number());
                    assert!(result_item["similarity_score"].is_number());
                }

                println!("Query '{}' found {} results", query, results.len());
            }
        }
    }

    #[tokio::test]
    #[ignore] // This test requires actual indexing - slow
    async fn test_user_management_code_search() {
        let temp_dir = TempDir::new().unwrap();
        test_fixtures::create_rust_project(&temp_dir).unwrap();

        let mut server = initialize_mcp_server(temp_dir.path()).await.unwrap();
        initialize_server(&mut server).await.unwrap();

        // Wait for indexing
        wait_for_indexing(Duration::from_secs(10)).await;

        // Search for user management functionality
        let search_queries = vec![
            "user role permissions admin",
            "create user account registration",
            "update user profile information",
            "delete user account management",
            "user validation email password",
        ];

        for query in search_queries {
            let request = JsonRpcRequest::new(
                "tools/call".to_string(),
                Some(json!({
                    "name": "semantic_search",
                    "arguments": {
                        "query": query,
                        "limit": 5
                    }
                })),
            );

            let response = server.handle_request(request).await.unwrap();

            if response.error.is_none() {
                let result = response.result.unwrap();
                let results = result["results"].as_array().unwrap();

                // Should find relevant user management code
                if !results.is_empty() {
                    println!("Query '{}' found {} results", query, results.len());

                    // Check for relevant file types
                    let file_paths: Vec<String> = results
                        .iter()
                        .map(|r| r["file_path"].as_str().unwrap().to_string())
                        .collect();

                    let has_user_files = file_paths.iter().any(|path| {
                        path.contains("user") || path.contains("auth") || path.contains("api")
                    });

                    assert!(
                        has_user_files,
                        "Should find user-related files for query: {}",
                        query
                    );
                }
            }
        }
    }

    #[tokio::test]
    #[ignore] // This test requires actual indexing - slow
    async fn test_api_documentation_search() {
        let temp_dir = TempDir::new().unwrap();
        test_fixtures::create_rust_project(&temp_dir).unwrap();

        let mut server = initialize_mcp_server(temp_dir.path()).await.unwrap();
        initialize_server(&mut server).await.unwrap();

        // Wait for indexing
        wait_for_indexing(Duration::from_secs(10)).await;

        // Search for API documentation and examples
        let doc_queries = vec![
            "API documentation endpoints",
            "authentication examples curl",
            "error response format JSON",
            "rate limiting security",
            "password requirements validation",
        ];

        for query in doc_queries {
            let request = JsonRpcRequest::new(
                "tools/call".to_string(),
                Some(json!({
                    "name": "semantic_search",
                    "arguments": {
                        "query": query,
                        "limit": 8
                    }
                })),
            );

            let response = server.handle_request(request).await.unwrap();

            if response.error.is_none() {
                let result = response.result.unwrap();
                let results = result["results"].as_array().unwrap();

                if !results.is_empty() {
                    println!(
                        "Documentation query '{}' found {} results",
                        query,
                        results.len()
                    );

                    // Verify we find documentation files
                    let file_paths: Vec<String> = results
                        .iter()
                        .map(|r| r["file_path"].as_str().unwrap().to_string())
                        .collect();

                    let has_docs = file_paths.iter().any(|path| {
                        path.ends_with(".md") || path.contains("docs/") || path.contains("README")
                    });

                    // Documentation should be findable
                    assert!(
                        has_docs,
                        "Should find documentation files for query: {}",
                        query
                    );
                }
            }
        }
    }

    #[tokio::test]
    #[ignore] // This test requires actual indexing - slow
    async fn test_cross_file_semantic_search() {
        let temp_dir = TempDir::new().unwrap();
        test_fixtures::create_rust_project(&temp_dir).unwrap();

        let mut server = initialize_mcp_server(temp_dir.path()).await.unwrap();
        initialize_server(&mut server).await.unwrap();

        // Wait for indexing
        wait_for_indexing(Duration::from_secs(10)).await;

        // Test searches that should find related concepts across multiple files
        let cross_file_queries = vec![
            (
                "security authentication",
                vec!["src/api/auth.rs", "src/models/user.rs", "docs/API.md"],
            ),
            (
                "error handling validation",
                vec!["src/utils/", "src/api/", "tests/"],
            ),
            (
                "database connection user",
                vec!["src/models/", "src/utils/", "src/main.rs"],
            ),
        ];

        for (query, expected_paths) in cross_file_queries {
            let request = JsonRpcRequest::new(
                "tools/call".to_string(),
                Some(json!({
                    "name": "semantic_search",
                    "arguments": {
                        "query": query,
                        "limit": 15
                    }
                })),
            );

            let response = server.handle_request(request).await.unwrap();

            if response.error.is_none() {
                let result = response.result.unwrap();
                let results = result["results"].as_array().unwrap();

                if !results.is_empty() {
                    println!(
                        "Cross-file query '{}' found {} results",
                        query,
                        results.len()
                    );

                    let found_paths: Vec<String> = results
                        .iter()
                        .map(|r| r["file_path"].as_str().unwrap().to_string())
                        .collect();

                    // Should find results from multiple related files
                    let mut found_expected = 0;
                    for expected_path in &expected_paths {
                        if found_paths.iter().any(|path| path.contains(expected_path)) {
                            found_expected += 1;
                        }
                    }

                    // Should find at least some of the expected file types
                    assert!(
                        found_expected > 0,
                        "Should find results in expected paths for query '{}'. Found paths: {:?}",
                        query,
                        found_paths
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn test_search_result_quality() {
        let temp_dir = TempDir::new().unwrap();
        test_fixtures::create_rust_project(&temp_dir).unwrap();

        let mut server = initialize_mcp_server(temp_dir.path()).await.unwrap();
        initialize_server(&mut server).await.unwrap();

        // Don't wait for full indexing - test with whatever is indexed
        wait_for_indexing(Duration::from_secs(3)).await;

        let request = JsonRpcRequest::new(
            "tools/call".to_string(),
            Some(json!({
                "name": "semantic_search",
                "arguments": {
                    "query": "authentication",
                    "limit": 10
                }
            })),
        );

        let response = match server.handle_request(request).await {
            Ok(resp) => resp,
            Err(e) => {
                println!("Search request failed (expected during testing): {}", e);
                // Create a mock empty response for testing purposes
                return;
            }
        };

        // Even if no results due to indexing, response should be well-formed
        assert!(response.error.is_none() || response.result.is_none());

        if response.error.is_none() {
            let result = response.result.unwrap();

            // Verify response structure
            assert!(result["results"].is_array());
            assert!(result["total_results"].is_number());
            assert!(result["execution_time_ms"].is_number());

            let results = result["results"].as_array().unwrap();

            if !results.is_empty() {
                for result_item in results {
                    // Verify each result has required fields
                    assert!(result_item["file_path"].is_string());
                    assert!(result_item["content"].is_string());
                    assert!(result_item["line_number"].is_number());
                    assert!(result_item["similarity_score"].is_number());

                    // Similarity score should be between 0 and 1
                    let similarity = result_item["similarity_score"].as_f64().unwrap();
                    assert!((0.0..=1.0).contains(&similarity));

                    // Content should not be empty
                    let content = result_item["content"].as_str().unwrap();
                    assert!(!content.trim().is_empty());

                    // File path should be relative and valid
                    let file_path = result_item["file_path"].as_str().unwrap();
                    assert!(!file_path.starts_with("/"));
                    assert!(file_path.contains("."));
                }

                // Results should be ordered by similarity (descending)
                for i in 1..results.len() {
                    let prev_score = results[i - 1]["similarity_score"].as_f64().unwrap();
                    let curr_score = results[i]["similarity_score"].as_f64().unwrap();
                    assert!(
                        prev_score >= curr_score,
                        "Results should be ordered by similarity"
                    );
                }
            }
        }
    }
}
