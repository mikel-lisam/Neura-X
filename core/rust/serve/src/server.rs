// ==========================================================
// Neura-X: Intelligence Without Limits.
// Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
//
// Neura-X Server — HTTP server implementation
// ==========================================================

//! The Neura-X HTTP server.
//! Provides OpenAI-compatible endpoints for local model serving.

use crate::openai_compat::{ChatRequest, ChatResponse};
use std::path::PathBuf;

/// Server configuration.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub model_path: PathBuf,
}

impl Default for ServerConfig {
    fn default() -> Self {
        ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 8000,
            model_path: PathBuf::from("model.nex"),
        }
    }
}

/// The Neura-X server.
pub struct NeuraServer {
    config: ServerConfig,
    running: bool,
}

impl NeuraServer {
    /// Create a new server with the given configuration.
    pub fn new(config: ServerConfig) -> Self {
        NeuraServer {
            config,
            running: false,
        }
    }

    /// Start the server.
    pub fn start(&mut self) -> Result<(), String> {
        println!("⚡ Neura-X Server starting...");
        println!("   Model: {}", self.config.model_path.display());
        println!("   URL: http://{}:{}/v1", self.config.host, self.config.port);
        println!("   Endpoints:");
        println!("     - POST /v1/chat/completions");
        println!("     - POST /v1/completions");
        println!("     - GET  /v1/models");
        println!("   Status: Ready");

        self.running = true;

        // In production, this starts an HTTP server (e.g., using axum or actix-web).
        // For now, we simulate the server loop.
        Ok(())
    }

    /// Stop the server.
    pub fn stop(&mut self) {
        self.running = false;
        println!("Neura-X Server stopped.");
    }

    /// Handle a chat completion request.
    pub fn handle_chat_completion(&self, request: &ChatRequest) -> ChatResponse {
        // In production, this runs inference through the Neura-X engine.
        let response_content = format!(
            "[Neura-X] Response to: {:?}",
            request.messages.last().map(|m| &m.content)
        );

        ChatResponse::new(&request.model, &response_content)
    }

    /// Check if the server is running.
    pub fn is_running(&self) -> bool {
        self.running
    }

    /// Get the server URL.
    pub fn url(&self) -> String {
        format!("http://{}:{}/v1", self.config.host, self.config.port)
    }
}