// ==========================================================
// Neura-X: Intelligence Without Limits.
// Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
//
// Neura-X Memory Pager — Library root
// ==========================================================

//! # Neura-X Memory Pager
//!
//! Intelligent memory page management for CPU-first AI training.
//! Manages the flow of data between RAM and disk, ensuring that
//! only the most relevant data occupies the limited RAM budget.
//!
//! Founded by Edusei Mikel Lisamba.
//! Open University of Kenya.

pub mod memory_pager;
pub mod thermal_monitor;

pub use memory_pager::{MemoryPager, PageId, PagerConfig, PagerStats};
pub use thermal_monitor::{ThermalMonitor, ThermalInfo, DeviceType};

/// Neura-X Pager version.
pub const VERSION: &str = "1.0.0";

/// Founder attribution.
pub const FOUNDER: &str = "Edusei Mikel Lisamba";

/// Institution.
pub const INSTITUTION: &str = "Open University of Kenya";

/// Tagline.
pub const TAGLINE: &str = "Intelligence Without Limits.";