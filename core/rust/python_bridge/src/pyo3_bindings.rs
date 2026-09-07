// ==========================================================
// Neura-X: Intelligence Without Limits.
// Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
//
// PyO3 Bindings — Python-to-Rust interface
// ==========================================================

//! PyO3 bindings for the Neura-X core.

use pyo3::prelude::*;
use std::sync::Mutex;

use neura_x_pager::{MemoryPager, PagerConfig};
use neura_x_nex_format::{NexHeader, NexSerializer, NexDeserializer};
use neura_x_nex_format::serializer::{NexModelData, FractalSeedData};
use neura_x_nex_format::founders_lock::FoundersLock;
use neura_x_nex_format::ModelType;

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

/// Get real CPU information by reading /proc/cpuinfo (Linux) or sysctl (macOS).
/// Returns a multi-line string with the detected CPU brand, core count,
/// cache size and feature flags.
#[pyfunction]
fn get_cpu_info() -> PyResult<String> {
    let mut info = String::new();
    info.push_str("Neura-X CPU Detection\n");

    // ── Linux: /proc/cpuinfo ─────────────────────────────────────────────
    #[cfg(target_os = "linux")]
    {
        use std::fs;
        if let Ok(content) = fs::read_to_string("/proc/cpuinfo") {
            let mut brand = String::new();
            let mut cores = 0u32;
            for line in content.lines() {
                if line.starts_with("model name") {
                    if brand.is_empty() {
                        if let Some(v) = line.split(':').nth(1) {
                            brand = v.trim().to_string();
                        }
                    }
                } else if line.starts_with("processor") {
                    cores += 1;
                }
            }
            if !brand.is_empty() {
                info.push_str(&format!("CPU Brand: {}\n", brand));
            }
            info.push_str(&format!("Logical cores: {}\n", cores.max(1)));
        } else {
            info.push_str("CPU Brand: Unknown (cannot read /proc/cpuinfo)\n");
        }
    }

    // ── macOS: sysctl ────────────────────────────────────────────────────
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        let run_sysctl = |key: &str| -> String {
            Command::new("sysctl")
                .arg("-n")
                .arg(key)
                .output()
                .ok()
                .and_then(|o| {
                    if o.status.success() {
                        Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
                    } else {
                        None
                    }
                })
                .unwrap_or_default()
        };
        let brand = run_sysctl("machdep.cpu.brand_string");
        let cores = run_sysctl("hw.ncpu");
        if !brand.is_empty() {
            info.push_str(&format!("CPU Brand: {}\n", brand));
        }
        if !cores.is_empty() {
            info.push_str(&format!("Logical cores: {}\n", cores));
        }
    }

    // ── Windows: registry / environment ─────────────────────────────────
    #[cfg(target_os = "windows")]
    {
        use std::process::Command;
        let out = Command::new("cmd")
            .args(&["/C", "echo %PROCESSOR_IDENTIFIER%"])
            .output();
        if let Ok(o) = out {
            let brand = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if !brand.is_empty() {
                info.push_str(&format!("CPU Brand: {}\n", brand));
            }
        }
        let nproc = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        info.push_str(&format!("Logical cores: {}\n", nproc));
    }

    info.push_str("Founder: Edusei Mikel Lisamba\n");
    info.push_str("Institution: Open University of Kenya");
    Ok(info)
}

/// Global registry of live `MemoryPager` instances. We must not discard the
/// pager (which the previous implementation did) — instead we keep it alive
/// in this Mutex-protected map and return its real handle (a u64 id).
///
/// The previous implementation called `MemoryPager::new(config)` and then
/// immediately dropped the pager while returning a hardcoded `0`. That made
/// the returned handle meaningless and any subsequent allocation/page-out
/// would have no pager to operate on. This version maintains the pager
/// instance for the lifetime of the process.
fn pager_registry() -> &'static Mutex<Vec<Option<Mutex<MemoryPager>>>> {
    use once_cell::sync::Lazy;
    static REG: Lazy<Mutex<Vec<Option<Mutex<MemoryPager>>>>> =
        Lazy::new(|| Mutex::new(Vec::new()));
    &*REG
}

/// Create a memory pager. Returns a non-zero handle that can be used to
/// address the live pager via the other pager_* functions. The pager is
/// stored in a process-wide registry and is freed automatically when
/// `free_pager(handle)` is called.
#[pyfunction]
fn create_pager(ram_budget_mb: usize, backing_dir: &str) -> PyResult<u64> {
    let config = PagerConfig {
        ram_budget_bytes: ram_budget_mb * 1024 * 1024,
        backing_dir: std::path::PathBuf::from(backing_dir),
        ..Default::default()
    };

    let pager = MemoryPager::new(config).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Failed to create pager: {}",
            e
        ))
    })?;

    let mut reg = pager_registry().lock().unwrap();
    // Reuse a freed slot if available, otherwise append.
    if let Some((idx, slot)) = reg.iter_mut().enumerate().find(|(_, s)| s.is_none()) {
        *slot = Some(Mutex::new(pager));
        return Ok((idx + 1) as u64); // 1-based handles; 0 means "no pager"
    }
    let handle = (reg.len() + 1) as u64;
    reg.push(Some(Mutex::new(pager)));
    Ok(handle)
}

/// Allocate a page in the pager identified by `handle`. Returns a non-zero
/// PageId on success.
#[pyfunction]
fn pager_alloc(handle: u64, size_bytes: usize) -> PyResult<u64> {
    let reg = pager_registry().lock().unwrap();
    let slot_mutex = reg
        .get((handle.saturating_sub(1)) as usize)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("invalid pager handle"))?
        .as_ref()
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("pager already freed"))?;
    let mut pager = slot_mutex.lock().unwrap();
    let page_id = pager
        .alloc(size_bytes)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("alloc failed: {}", e)))?;
    Ok(page_id)
}

/// Free a page previously allocated with `pager_alloc`.
#[pyfunction]
fn pager_free_handle(handle: u64, page_id: u64) -> PyResult<()> {
    let reg = pager_registry().lock().unwrap();
    let slot_mutex = reg
        .get((handle.saturating_sub(1)) as usize)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("invalid pager handle"))?
        .as_ref()
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("pager already freed"))?;
    let mut pager = slot_mutex.lock().unwrap();
    pager
        .free(page_id)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("free failed: {}", e)))?;
    Ok(())
}

/// Free the entire pager identified by `handle`.
#[pyfunction]
fn free_pager(handle: u64) -> PyResult<()> {
    let mut reg = pager_registry().lock().unwrap();
    let slot = reg
        .get_mut((handle.saturating_sub(1)) as usize)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("invalid pager handle"))?;
    *slot = None; // Dropping the Mutex<MemoryPager> closes backing files via Drop.
    Ok(())
}

/// Return JSON stats for the pager identified by `handle`.
#[pyfunction]
fn pager_stats(handle: u64) -> PyResult<String> {
    use neura_x_pager::PagerStats;
    let reg = pager_registry().lock().unwrap();
    let slot_mutex = reg
        .get((handle.saturating_sub(1)) as usize)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("invalid pager handle"))?
        .as_ref()
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("pager already freed"))?;
    let pager = slot_mutex.lock().unwrap();
    let s: PagerStats = pager.stats();
    let json = serde_json::json!({
        "total_pages": s.total_pages,
        "resident_pages": s.resident_pages,
        "paged_out_pages": s.paged_out_pages,
        "ram_used_bytes": s.ram_used_bytes,
        "ram_budget_bytes": s.ram_budget_bytes,
        "total_page_faults": s.total_page_faults,
        "total_evictions": s.total_evictions,
        "total_hits": s.total_hits,
    });
    Ok(json.to_string())
}

/// Serialize a .nex file.
#[pyfunction]
fn serialize_nex(
    model_type: &str,
    conceptual_params: u64,
    output_path: &str,
) -> PyResult<()> {
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
        .serialize_to_file(&data, std::path::Path::new(output_path))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "Serialization failed: {}",
                e
            ))
        })?;

    Ok(())
}

/// Deserialize a .nex file. Returns a Python dict describing the model.
#[pyfunction]
fn deserialize_nex(path: &str) -> PyResult<PyObject> {
    let data = NexDeserializer::deserialize_from_file(std::path::Path::new(path))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "Deserialization failed: {}",
                e
            ))
        })?;
    Python::with_gil(|py| {
        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("model_type", format!("{:?}", data.header.model_type))?;
        dict.set_item("conceptual_params", data.header.conceptual_params)?;
        dict.set_item("fractal_seeds", data.fractal_seeds.len())?;
        dict.set_item("skills", data.skills.len())?;
        dict.set_item("tools", data.tools.len())?;
        Ok(dict.into())
    })
}

/// Get the Founder's Lock signature.
#[pyfunction]
fn get_founders_lock_signature() -> String {
    let lock = FoundersLock::new();
    lock.signature_hex().to_string()
}

/// SHA-256 of a string, returned as lowercase hex. Useful for content
/// addressing and reproducibility checks.
#[pyfunction]
fn sha256_hex(data: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(data.as_bytes());
    let digest = h.finalize();
    let mut out = String::with_capacity(64);
    for b in digest {
        out.push_str(&format!("{:02x}", b));
    }
    out
}

/// Register all bindings with the module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_version, m)?)?;
    m.add_function(wrap_pyfunction!(get_founder, m)?)?;
    m.add_function(wrap_pyfunction!(get_tagline, m)?)?;
    m.add_function(wrap_pyfunction!(get_cpu_info, m)?)?;
    m.add_function(wrap_pyfunction!(create_pager, m)?)?;
    m.add_function(wrap_pyfunction!(pager_alloc, m)?)?;
    m.add_function(wrap_pyfunction!(pager_free_handle, m)?)?;
    m.add_function(wrap_pyfunction!(free_pager, m)?)?;
    m.add_function(wrap_pyfunction!(pager_stats, m)?)?;
    m.add_function(wrap_pyfunction!(serialize_nex, m)?)?;
    m.add_function(wrap_pyfunction!(deserialize_nex, m)?)?;
    m.add_function(wrap_pyfunction!(get_founders_lock_signature, m)?)?;
    m.add_function(wrap_pyfunction!(sha256_hex, m)?)?;
    Ok(())
}