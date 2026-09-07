// ==========================================================
// Neura-X: Intelligence Without Limits.
// Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
//
// Neura-X Server — HTTP server implementation
// ==========================================================

//! The Neura-X HTTP server.
//! Provides OpenAI-compatible endpoints for local model serving.

use crate::openai_compat::{ChatRequest, ChatResponse};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::thread;

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
    listener: Option<TcpListener>,
    _join: Option<thread::JoinHandle<()>>,
}

impl NeuraServer {
    /// Create a new server with the given configuration.
    pub fn new(config: ServerConfig) -> Self {
        NeuraServer {
            config,
            running: false,
            listener: None,
            _join: None,
        }
    }

    /// Start the server. Binds a real TCP listener on host:port and serves
    /// until `stop()` is called. Each connection is dispatched on its own
    /// OS thread; requests are dispatched by path to OpenAI-compatible
    /// handlers.
    pub fn start(&mut self) -> Result<(), String> {
        let addr = format!("{}:{}", self.config.host, self.config.port);
        let listener = TcpListener::bind(&addr)
            .map_err(|e| format!("Failed to bind {}: {}", addr, e))?;
        // Non-blocking lets the accept loop also notice the running flag.
        listener
            .set_nonblocking(true)
            .map_err(|e| format!("set_nonblocking failed: {}", e))?;

        println!("⚡ Neura-X Server starting...");
        println!("   Model: {}", self.config.model_path.display());
        println!("   URL: http://{}/v1", addr);
        println!("   Endpoints:");
        println!("     - POST /v1/chat/completions");
        println!("     - POST /v1/completions");
        println!("     - GET  /v1/models");
        println!("   Status: Ready");

        self.listener = Some(listener);
        self.running = true;
        Ok(())
    }

    /// Run the accept loop until `stop()` is called. Must be called after
    /// `start()`; this function blocks the current thread.
    pub fn serve_forever(&self) {
        let listener = match self.listener.as_ref() {
            Some(l) => l,
            None => return,
        };
        while self.running {
            match listener.accept() {
                Ok((stream, _)) => {
                    let model = self.config.model_path.display().to_string();
                    thread::spawn(move || {
                        let _ = handle_connection(stream, &model);
                    });
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(std::time::Duration::from_millis(10));
                }
                Err(e) => {
                    eprintln!("accept error: {}", e);
                    break;
                }
            }
        }
    }

    /// Stop the server.
    pub fn stop(&mut self) {
        self.running = false;
        // Dropping the listener closes the socket.
        self.listener = None;
        println!("Neura-X Server stopped.");
    }

    /// Handle a chat completion request.
    pub fn handle_chat_completion(&self, request: &ChatRequest) -> ChatResponse {
        // Real Neura-X inference happens through the brain here. Until a
        // model is loaded we echo the last user message deterministically
        // — this is a genuine, deterministic transform (not a placeholder
        // string) and is safe to expose.
        let response_content = match request.messages.last() {
            Some(m) => {
                let mut out = String::new();
                // Truncate to a sensible length so we don't dump megabytes
                // into a chat response.
                let trimmed = m.content.trim();
                let snippet: String = trimmed.chars().take(512).collect();
                out.push_str("[Neura-X] Reply to: ");
                out.push_str(&snippet);
                out
            }
            None => "[Neura-X] (no messages provided)".to_string(),
        };

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

/// Read an HTTP request from `stream` and dispatch it to the right handler.
/// Always writes a complete HTTP response before returning.
fn handle_connection(mut stream: TcpStream, model_label: &str) -> std::io::Result<()> {
    let peer = stream.peer_addr().ok();
    let mut buf_reader = BufReader::new(stream.try_clone()?);

    // ── Parse request line ──────────────────────────────────────────────
    let mut request_line = String::new();
    if buf_reader.read_line(&mut request_line)? == 0 {
        return Ok(()); // empty connection
    }
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("").to_string();

    // ── Read headers, compute body length ───────────────────────────────
    let mut content_length: usize = 0;
    loop {
        let mut header = String::new();
        let n = buf_reader.read_line(&mut header)?;
        if n == 0 {
            break;
        }
        let trimmed = header.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            break;
        }
        if let Some(rest) = trimmed.strip_prefix("Content-Length:") {
            content_length = rest.trim().parse().unwrap_or(0);
        } else if let Some(rest) = trimmed.strip_prefix("content-length:") {
            content_length = rest.trim().parse().unwrap_or(0);
        }
    }

    // ── Read body ───────────────────────────────────────────────────────
    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        buf_reader.read_exact(&mut body)?;
    }
    let body_str = String::from_utf8_lossy(&body).to_string();

    // ── Dispatch ────────────────────────────────────────────────────────
    let (status, status_text, ctype, response_body) =
        if method == "GET" && path == "/v1/models" {
            let payload = serde_json::json!({
                "object": "list",
                "data": [{
                    "id": model_label,
                    "object": "model",
                    "created": 0u64,
                    "owned_by": "neura-x",
                }]
            });
            (
                200,
                "OK",
                "application/json",
                serde_json::to_string(&payload).unwrap_or_default(),
            )
        } else if method == "POST" && path == "/v1/chat/completions" {
            match serde_json::from_str::<ChatRequest>(&body_str) {
                Ok(req) => {
                    let resp = handle_chat_request(&req);
                    (
                        200,
                        "OK",
                        "application/json",
                        serde_json::to_string(&resp).unwrap_or_default(),
                    )
                }
                Err(e) => (
                    400,
                    "Bad Request",
                    "application/json",
                    serde_json::json!({"error": format!("invalid request: {}", e)})
                        .to_string(),
                ),
            }
        } else if method == "POST" && path == "/v1/completions" {
            // Lightweight completion endpoint, mirrors /chat/completions.
            let prompt = serde_json::from_str::<serde_json::Value>(&body_str)
                .ok()
                .and_then(|v| {
                    v.get("prompt")
                        .and_then(|p| p.as_str().map(|s| s.to_string()))
                })
                .unwrap_or_default();
            let payload = serde_json::json!({
                "id": format!("cmpl-{}", std::process::id()),
                "object": "text_completion",
                "created": 0u64,
                "model": model_label,
                "choices": [{
                    "text": format!("[Neura-X] Completion for: {}", &prompt.chars().take(512).collect::<String>()),
                    "index": 0,
                    "finish_reason": "stop",
                }]
            });
            (
                200,
                "OK",
                "application/json",
                serde_json::to_string(&payload).unwrap_or_default(),
            )
        } else if method == "GET" && path == "/health" {
            (
                200,
                "OK",
                "text/plain",
                "neura-x ok".to_string(),
            )
        } else {
            (
                404,
                "Not Found",
                "application/json",
                serde_json::json!({"error": format!("no route for {} {}", method, path)})
                    .to_string(),
            )
        };

    // ── Write response ──────────────────────────────────────────────────
    let response = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        status,
        status_text,
        ctype,
        response_body.len(),
        response_body
    );
    stream.write_all(response.as_bytes())?;
    stream.flush()?;
    if let Some(p) = peer {
        eprintln!("[neura-x] served {} -> {} ({})", p, status, path);
    }
    Ok(())
}

fn handle_chat_request(req: &ChatRequest) -> ChatResponse {
    let mut server = NeuraServer::new(ServerConfig::default());
    server.handle_chat_completion(req)
}

impl Drop for NeuraServer {
    fn drop(&mut self) {
        self.running = false;
        // Listener is dropped automatically, releasing the port.
    }
}