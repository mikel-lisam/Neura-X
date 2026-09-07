// ==========================================================
// Neura-X: Intelligence Without Limits.
// Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
//
// .nex Deserializer — Reads .nex files
// ==========================================================

//! Deserializer for the .nex file format.

use crate::{NexHeader, NEX_FORMAT_VERSION};
use crate::serializer::NexModelData;
use std::io::{self, Read};
use std::path::Path;

/// The magic bytes for .nex files.
const NEX_MAGIC_BYTES: &[u8; 4] = b"NEXX";

/// .nex file deserializer.
pub struct NexDeserializer;

impl NexDeserializer {
    /// Create a new deserializer.
    pub fn new() -> Self {
        NexDeserializer
    }

    /// Deserialize a .nex file.
    pub fn deserialize_from_file(path: &Path) -> io::Result<NexModelData> {
        let mut file = std::fs::File::open(path)?;
        Self::deserialize_from_reader(&mut file)
    }

    /// Deserialize from a reader.
    pub fn deserialize_from_reader<R: Read>(reader: &mut R) -> io::Result<NexModelData> {
        // Read magic number (4 bytes)
        let mut magic_bytes = [0u8; 4];
        reader.read_exact(&mut magic_bytes)?;

        if &magic_bytes != NEX_MAGIC_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Invalid magic number: expected 'NEXX', got '{}'",
                    String::from_utf8_lossy(&magic_bytes)
                ),
            ));
        }

        // Read format version (4 bytes, little-endian)
        let mut version_bytes = [0u8; 4];
        reader.read_exact(&mut version_bytes)?;
        let version = u32::from_le_bytes(version_bytes);

        if version > NEX_FORMAT_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported format version: {}", version),
            ));
        }

        // Read compressed data size (8 bytes, little-endian)
        let mut size_bytes = [0u8; 8];
        reader.read_exact(&mut size_bytes)?;
        let data_size = u64::from_le_bytes(size_bytes) as usize;

        // Read compressed data
        let mut compressed = vec![0u8; data_size];
        reader.read_exact(&mut compressed)?;

        // Decompress with zstd
        let decompressed = zstd::decode_all(&compressed[..]).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Decompression error: {}", e),
            )
        })?;

        // Deserialize with bincode
        let data: NexModelData = bincode::deserialize(&decompressed).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Deserialization error: {}", e),
            )
        })?;

        // Validate header
        data.header.validate().map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, e)
        })?;

        Ok(data)
    }

    /// Deserialize from bytes.
    pub fn deserialize_from_bytes(bytes: &[u8]) -> io::Result<NexModelData> {
        let mut cursor = io::Cursor::new(bytes);
        Self::deserialize_from_reader(&mut cursor)
    }

    /// Read only the header from a .nex file (fast).
    pub fn read_header(path: &Path) -> io::Result<NexHeader> {
        let data = Self::deserialize_from_file(path)?;
        Ok(data.header)
    }
}