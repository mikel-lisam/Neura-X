// ==========================================================
// Neura-X: Intelligence Without Limits.
// Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
//
// Founder's Lock — Cryptographic signature for .nex files
// ==========================================================

//! The Founder's Lock: an immutable cryptographic signature
//! embedded in every .nex file. Derived from the Founder's name
//! using SHA-256. Removal or alteration is prohibited.

use sha2::{Sha256, Digest};
use crate::FOUNDER_NAME;

/// The Founder's Lock.
///
/// Generates a cryptographic signature from the Founder's identity.
/// This signature is embedded in every .nex file and cannot be
/// removed without invalidating the file.
pub struct FoundersLock {
    signature: Vec<u8>,
    signature_hex: String,
}

impl FoundersLock {
    /// Create the Founder's Lock.
    pub fn new() -> Self {
        let mut hasher = Sha256::new();
        hasher.update(FOUNDER_NAME.as_bytes());
        let result = hasher.finalize();

        let signature = result.to_vec();
        let signature_hex = hex::encode(&signature);

        FoundersLock {
            signature,
            signature_hex,
        }
    }

    /// Get the raw signature bytes.
    pub fn signature_bytes(&self) -> &[u8] {
        &self.signature
    }

    /// Get the hex-encoded signature.
    pub fn signature_hex(&self) -> &str {
        &self.signature_hex
    }

    /// Verify that a .nex file's signature matches the Founder's Lock.
    pub fn verify(&self, file_signature: &[u8]) -> bool {
        file_signature == self.signature.as_slice()
    }

    /// Generate a seed integer from the signature (for Fractal Tensors).
    pub fn fractal_seed(&self) -> u64 {
        let mut seed_bytes = [0u8; 8];
        seed_bytes.copy_from_slice(&self.signature[..8]);
        u64::from_le_bytes(seed_bytes)
    }

    /// Get the Founder's name.
    pub fn founder_name(&self) -> &'static str {
        FOUNDER_NAME
    }
}

impl Default for FoundersLock {
    fn default() -> Self {
        Self::new()
    }
}