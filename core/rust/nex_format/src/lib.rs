// ==========================================================
// Neura-X: Intelligence Without Limits.
// Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
//
// Neura-X .nex File Format — Library root
// ==========================================================

//! # Neura-X .nex File Format
//!
//! The proprietary file format for Neura-X models.
//! Contains Fractal Seeds, routing tables, holographic memory,
//! and the Founder's Lock cryptographic signature.

pub mod serializer;
pub mod deserializer;
pub mod founders_lock;

pub use serializer::NexSerializer;
pub use deserializer::NexDeserializer;
pub use founders_lock::FoundersLock;

/// .nex magic number: "NEXX" in ASCII.
pub const NEX_MAGIC: u32 = 0x4E455858;

/// .nex file format version.
pub const NEX_FORMAT_VERSION: u32 = 1;

/// Header size in bytes.
pub const NEX_HEADER_SIZE: usize = 128;

/// Founder's name (immutable).
pub const FOUNDER_NAME: &str = "Edusei Mikel Lisamba";

/// Institution.
pub const INSTITUTION: &str = "Open University of Kenya";

/// Model type enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ModelType {
    LLM,
    ImageGen,
    VideoGen,
    AudioGen,
    SpeechSynthesis,
    SpeechRecognition,
    MoE,
    Predictive,
    Vision,
    Agent,
    Composed,
}

/// .nex file header.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NexHeader {
    pub magic: u32,
    pub version: u32,
    pub founder: String,
    pub model_type: ModelType,
    pub conceptual_params: u64,
    pub architecture: String,
    pub fractal_function: String,
    pub precision: String,
    pub checksum: u64,
}

impl NexHeader {
    /// Create a new header with default values.
    pub fn new(model_type: ModelType, conceptual_params: u64) -> Self {
        NexHeader {
            magic: NEX_MAGIC,
            version: NEX_FORMAT_VERSION,
            founder: FOUNDER_NAME.to_string(),
            model_type,
            conceptual_params,
            architecture: "transformer".to_string(),
            fractal_function: "harmonic_sine".to_string(),
            precision: "fp32".to_string(),
            checksum: 0,
        }
    }

    /// Validate the header.
    pub fn validate(&self) -> Result<(), String> {
        if self.magic != NEX_MAGIC {
            return Err(format!(
                "Invalid magic number: expected 0x{:08X}, got 0x{:08X}",
                NEX_MAGIC, self.magic
            ));
        }
        if self.founder != FOUNDER_NAME {
            return Err(format!(
                "Invalid founder: expected '{}', got '{}'",
                FOUNDER_NAME, self.founder
            ));
        }
        Ok(())
    }
}