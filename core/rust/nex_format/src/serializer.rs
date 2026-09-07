// ==========================================================
// Neura-X: Intelligence Without Limits.
// Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
//
// .nex Serializer — Writes .nex files
// ==========================================================

//! Serializer for the .nex file format.

use crate::{NexHeader, NEX_FORMAT_VERSION};
use serde::{Serialize, Deserialize};
use std::io::{self, Write};
use std::path::Path;

/// Fractal Seed data for serialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FractalSeedData {
    pub a: Vec<f32>,
    pub w1: Vec<f32>,
    pub w2: Vec<f32>,
    pub phi: Vec<f32>,
    pub k: usize,
}

/// Complete .nex model data.
#[derive(Debug, Serialize, Deserialize)]
pub struct NexModelData {
    pub header: NexHeader,
    pub fractal_seeds: Vec<FractalSeedData>,
    pub router_mask: Vec<u8>,
    pub holographic_memory: Vec<u8>,
    pub shadow_optimizer: Option<Vec<f32>>,
    pub skills: Vec<String>,
    pub tools: Vec<String>,
}

/// .nex file serializer.
pub struct NexSerializer {
    compression_level: i32,
}

impl NexSerializer {
    /// Create a new serializer.
    pub fn new() -> Self {
        NexSerializer {
            compression_level: 3, // Default zstd compression level
        }
    }

    /// Set compression level (0-21 for zstd).
    pub fn with_compression(mut self, level: i32) -> Self {
        self.compression_level = level.clamp(0, 21);
        self
    }

    /// Serialize model data to a .nex file.
    pub fn serialize_to_file(
        &self,
        data: &NexModelData,
        path: &Path,
    ) -> io::Result<()> {
        // Serialize to binary
        let encoded = bincode::serialize(data).map_err(|e| {
            io::Error::new(io::ErrorKind::Other, format!("Serialization error: {}", e))
        })?;

        // Compress with zstd
        let compressed = zstd::encode_all(&encoded[..], self.compression_level)
            .map_err(|e| {
                io::Error::new(io::ErrorKind::Other, format!("Compression error: {}", e))
            })?;

        // Write to file
        let mut file = std::fs::File::create(path)?;

        // Write magic number as raw ASCII bytes (human-readable "NEXX")
        file.write_all(b"NEXX")?;

        // Write format version as little-endian u32
        file.write_all(&NEX_FORMAT_VERSION.to_le_bytes())?;

        // Write compressed data size as little-endian u64
        let data_size = compressed.len() as u64;
        file.write_all(&data_size.to_le_bytes())?;

        // Write compressed data
        file.write_all(&compressed)?;

        file.flush()?;

        Ok(())
    }

    /// Serialize model data to bytes.
    pub fn serialize_to_bytes(&self, data: &NexModelData) -> io::Result<Vec<u8>> {
        let encoded = bincode::serialize(data).map_err(|e| {
            io::Error::new(io::ErrorKind::Other, format!("Serialization error: {}", e))
        })?;

        let compressed = zstd::encode_all(&encoded[..], self.compression_level)
            .map_err(|e| {
                io::Error::new(io::ErrorKind::Other, format!("Compression error: {}", e))
            })?;

        let mut output = Vec::new();

        // Magic number as raw bytes
        output.extend_from_slice(b"NEXX");

        // Format version
        output.extend_from_slice(&NEX_FORMAT_VERSION.to_le_bytes());

        // Data size
        output.extend_from_slice(&(compressed.len() as u64).to_le_bytes());

        // Compressed data
        output.extend_from_slice(&compressed);

        Ok(output)
    }

    /// Get the estimated file size for given model data.
    pub fn estimate_size(&self, data: &NexModelData) -> io::Result<usize> {
        let encoded = bincode::serialize(data).map_err(|e| {
            io::Error::new(io::ErrorKind::Other, format!("Serialization error: {}", e))
        })?;

        let compressed = zstd::encode_all(&encoded[..], self.compression_level)
            .map_err(|e| {
                io::Error::new(io::ErrorKind::Other, format!("Compression error: {}", e))
            })?;

        // 4 bytes magic + 4 bytes version + 8 bytes size + compressed data
        Ok(compressed.len() + 16)
    }
}