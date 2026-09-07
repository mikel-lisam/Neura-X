// ==========================================================
// Neura-X: Intelligence Without Limits.
// Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
//
// Thermal Monitor — CPU temperature monitoring for laptops
// ==========================================================

//! Thermal monitoring for laptop-first thermal management.

use std::fs;
use std::path::Path;

/// Device form factor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    Unknown,
    Laptop,
    Desktop,
    Server,
    Embedded,
}

/// Thermal information.
#[derive(Debug, Clone, Default)]
pub struct ThermalInfo {
    pub current_temp_celsius: f64,
    pub throttle_threshold_celsius: f64,
    pub is_throttling: bool,
    pub fan_speed_rpm: u32,
    pub sensor_available: bool,
}

/// Thermal monitor for Neura-X.
///
/// Reads CPU temperature from system sensors and provides
/// thermal-aware scheduling recommendations.
pub struct ThermalMonitor {
    device_type: DeviceType,
    throttle_threshold: f64,
}

impl ThermalMonitor {
    /// Create a new thermal monitor.
    pub fn new() -> Self {
        ThermalMonitor {
            device_type: Self::detect_device_type(),
            throttle_threshold: 90.0, // Default throttle threshold
        }
    }

    /// Read current thermal information.
    pub fn read(&self) -> ThermalInfo {
        let mut info = ThermalInfo {
            throttle_threshold_celsius: self.throttle_threshold,
            ..Default::default()
        };

        // Try Linux thermal zone
        #[cfg(target_os = "linux")]
        {
            let thermal_path = "/sys/class/thermal/thermal_zone0/temp";
            if Path::new(thermal_path).exists() {
                if let Ok(content) = fs::read_to_string(thermal_path) {
                    if let Ok(temp) = content.trim().parse::<f64>() {
                        info.current_temp_celsius = temp / 1000.0;
                        info.sensor_available = true;
                        info.is_throttling = info.current_temp_celsius >= self.throttle_threshold;
                        return info;
                    }
                }
            }

            // Try hwmon
            for i in 0..10 {
                let hwmon_path = format!("/sys/class/hwmon/hwmon{}/temp1_input", i);
                if Path::new(&hwmon_path).exists() {
                    if let Ok(content) = fs::read_to_string(&hwmon_path) {
                        if let Ok(temp) = content.trim().parse::<f64>() {
                            info.current_temp_celsius = temp / 1000.0;
                            info.sensor_available = true;
                            info.is_throttling = info.current_temp_celsius >= self.throttle_threshold;
                            return info;
                        }
                    }
                }
            }
        }

        // Try macOS
        #[cfg(target_os = "macos")]
        {
            // macOS doesn't expose thermal sensors directly.
            // In production, use IOKit or the `powermetrics` command.
            info.sensor_available = false;
        }

        info
    }

    /// Detect the device form factor.
    fn detect_device_type() -> DeviceType {
        #[cfg(target_os = "linux")]
        {
            // Check for laptop indicators
            let chassis_path = "/sys/class/dmi/id/chassis_type";
            if let Ok(content) = fs::read_to_string(chassis_path) {
                let chassis: u32 = content.trim().parse().unwrap_or(0);
                return match chassis {
                    8 | 9 | 10 | 14 => DeviceType::Laptop,
                    3 | 4 | 6 | 7 => DeviceType::Desktop,
                    17 | 23 => DeviceType::Server,
                    _ => DeviceType::Unknown,
                };
            }

            // Check for battery (laptop indicator)
            let battery_path = "/sys/class/power_supply/BAT0";
            if Path::new(battery_path).exists() {
                return DeviceType::Laptop;
            }
        }

        #[cfg(target_os = "macos")]
        {
            // All Macs with batteries are laptops
            return DeviceType::Laptop; // Simplified
        }

        DeviceType::Unknown
    }

    /// Get the detected device type.
    pub fn device_type(&self) -> DeviceType {
        self.device_type
    }

    /// Check if the system should pause training due to heat.
    pub fn should_pause(&self) -> bool {
        let info = self.read();
        if !info.sensor_available {
            return false;
        }
        info.current_temp_celsius >= self.throttle_threshold
    }

    /// Get recommended pause duration in seconds.
    pub fn recommended_pause_seconds(&self) -> u32 {
        let info = self.read();
        if !info.is_throttling {
            return 0;
        }

        // Scale pause duration with temperature
        let excess = info.current_temp_celsius - self.throttle_threshold;
        if excess > 10.0 {
            10 // Very hot: pause 10 seconds
        } else if excess > 5.0 {
            5 // Hot: pause 5 seconds
        } else {
            2 // Warm: pause 2 seconds
        }
    }
}