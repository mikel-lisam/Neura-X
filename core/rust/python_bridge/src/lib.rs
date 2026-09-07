// ==========================================================
// Neura-X: Intelligence Without Limits.
// Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
//
// Neura-X Python Bridge — Library root
// ==========================================================

//! # Neura-X Python Bridge
//!
//! PyO3 bindings that expose the Neura-X Rust core to Python.

mod pyo3_bindings;

use pyo3::prelude::*;

/// The Neura-X Python module.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register the module functions and classes
    pyo3_bindings::register(m)?;

    // Set module attributes
    m.add("__version__", "1.0.0")?;
    m.add("__author__", "Edusei Mikel Lisamba")?;
    m.add("__tagline__", "Intelligence Without Limits.")?;
    m.add("__institution__", "Open University of Kenya")?;

    Ok(())
}