// ==========================================================
// Neura-X: Intelligence Without Limits.
// Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
//
// PyO3 Bindings — Python-to-Rust interface
// ==========================================================

//! PyO3 bindings for the Neura-X core.

use pyo3::prelude::*;

/// Get the Neura-X version.
#[pyfunction]
fn get_version() -> &'static str {
    "1.0.0"
}

/// Get the founder's name.
#[pyfunction]
fn get_founder() -> &'static str {
    "Edusei Mikel Lisamba"
}

/// Get the tagline.
#[pyfunction]
fn get_tagline() -> &'static str {
    "Intelligence Without Limits."
}

/// Get CPU information as a dictionary.
#[pyfunction]
fn get_cpu_info() -> PyResult<String> {
    let info = format!(
        "Neura-X CPU Detection\n\
         Founder: Edusei Mikel Lisamba\n\
         Institution: Open University of Kenya"
    );
    Ok(info)
}

/// Create a memory pager.
#[pyfunction]
fn create_pager(ram_budget_mb: usize, backing_dir: &str) -> PyResult<u64> {
    use neura_x_pager::{MemoryPager, PagerConfig};
    use std::path::PathBuf;

    let config = PagerConfig {
        ram_budget_bytes: ram_budget_mb * 1024 * 1024,
        backing_dir: PathBuf::from(backing_dir),
        ..Default::default()
    };

    match MemoryPager::new(config) {
        Ok(_pager) => Ok(0), // Return pager ID
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to create pager: {}", e),
        )),
    }
}

/// Serialize a .nex file.
#[pyfunction]
fn serialize_nex(
    model_type: &str,
    conceptual_params: u64,
    output_path: &str,
) -> PyResult<()> {
    use neura_x_nex_format::{NexHeader, ModelType};
    use neura_x_nex_format::serializer::{NexSerializer, NexModelData};
    use std::path::Path;

    let mt = match model_type {
        "llm" => ModelType::LLM,
        "imagegen" => ModelType::ImageGen,
        "videogen" => ModelType::VideoGen,
        "audiogen" => ModelType::AudioGen,
        "moe" => ModelType::MoE,
        "predictive" => ModelType::Predictive,
        "agent" => ModelType::Agent,
        _ => ModelType::LLM,
    };

    let header = NexHeader::new(mt, conceptual_params);
    let data = NexModelData {
        header,
        fractal_seeds: vec![],
        router_mask: vec![],
        holographic_memory: vec![],
        shadow_optimizer: None,
        skills: vec![],
        tools: vec![],
    };

    let serializer = NexSerializer::new();
    serializer
        .serialize_to_file(&data, Path::new(output_path))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(
                format!("Serialization failed: {}", e),
            )
        })?;

    Ok(())
}

/// Get the Founder's Lock signature.
#[pyfunction]
fn get_founders_lock_signature() -> String {
    use neura_x_nex_format::founders_lock::FoundersLock;
    let lock = FoundersLock::new();
    lock.signature_hex().to_string()
}

/// Register all bindings with the module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_version, m)?)?;
    m.add_function(wrap_pyfunction!(get_founder, m)?)?;
    m.add_function(wrap_pyfunction!(get_tagline, m)?)?;
    m.add_function(wrap_pyfunction!(get_cpu_info, m)?)?;
    m.add_function(wrap_pyfunction!(create_pager, m)?)?;
    m.add_function(wrap_pyfunction!(serialize_nex, m)?)?;
    m.add_function(wrap_pyfunction!(get_founders_lock_signature, m)?)?;
    Ok(())
}