// ==========================================================
// Neura-X: Intelligence Without Limits.
// Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
//
// Neura-X Server — Library root
// ==========================================================

//! # Neura-X Server
//!
//! OpenAI-compatible local server for Neura-X models.

pub mod openai_compat;
pub mod server;

pub use openai_compat::{ChatRequest, ChatResponse, ChatMessage};
pub use server::NeuraServer;

pub const VERSION: &str = "1.0.0";