// ==========================================================
// Neura-X: Intelligence Without Limits.
// Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
//
// Memory Pager — LRU page management between RAM and disk
// ==========================================================

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::time::Instant;

/// Unique identifier for a memory page.
pub type PageId = u64;

/// Configuration for the memory pager.
#[derive(Debug, Clone)]
pub struct PagerConfig {
    pub ram_budget_bytes: usize,
    pub backing_dir: PathBuf,
    pub page_size_bytes: usize,
    pub max_pages: usize,
}

impl Default for PagerConfig {
    fn default() -> Self {
        PagerConfig {
            ram_budget_bytes: 4 * 1024 * 1024 * 1024, // 4 GB
            backing_dir: PathBuf::from("/tmp/neura-x-pager"),
            page_size_bytes: 1024 * 1024, // 1 MB
            max_pages: 4096,
        }
    }
}

/// State of a memory page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PageState {
    Resident,
    PagedOut,
}

/// Internal page metadata.
#[allow(dead_code)]
struct PageMeta {
    id: PageId,
    size_bytes: usize,
    state: PageState,
    last_access: Instant,
    access_count: u64,
    dirty: bool,
    backing_file: Option<PathBuf>,
}

/// Pager statistics.
#[derive(Debug, Clone, Default)]
pub struct PagerStats {
    pub total_pages: usize,
    pub resident_pages: usize,
    pub paged_out_pages: usize,
    pub ram_used_bytes: usize,
    pub ram_budget_bytes: usize,
    pub total_page_faults: u64,
    pub total_evictions: u64,
    pub total_hits: u64,
}

/// The Neura-X Memory Pager.
pub struct MemoryPager {
    config: PagerConfig,
    pages: HashMap<PageId, PageMeta>,
    data: HashMap<PageId, Vec<u8>>,
    next_page_id: PageId,
    ram_used_bytes: usize,
    total_page_faults: u64,
    total_evictions: u64,
    total_hits: u64,
}

impl MemoryPager {
    pub fn new(config: PagerConfig) -> io::Result<Self> {
        fs::create_dir_all(&config.backing_dir)?;

        Ok(MemoryPager {
            config,
            pages: HashMap::new(),
            data: HashMap::new(),
            next_page_id: 0,
            ram_used_bytes: 0,
            total_page_faults: 0,
            total_evictions: 0,
            total_hits: 0,
        })
    }

    pub fn alloc(&mut self, size_bytes: usize) -> io::Result<PageId> {
        if self.pages.len() >= self.config.max_pages {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "Maximum page count reached",
            ));
        }

        while self.ram_used_bytes + size_bytes > self.config.ram_budget_bytes {
            if !self.evict_lru()? { break; }
        }

        let page_id = self.next_page_id;
        self.next_page_id += 1;

        let meta = PageMeta {
            id: page_id,
            size_bytes,
            state: PageState::Resident,
            last_access: Instant::now(),
            access_count: 0,
            dirty: false,
            backing_file: None,
        };

        let data = vec![0u8; size_bytes];
        self.ram_used_bytes += size_bytes;
        self.pages.insert(page_id, meta);
        self.data.insert(page_id, data);

        Ok(page_id)
    }

    pub fn access(&mut self, page_id: PageId) -> io::Result<&Vec<u8>> {
        let (state, size, backing_file) = {
            let meta = self.pages.get(&page_id).ok_or_else(|| {
                io::Error::new(io::ErrorKind::NotFound, "Page not found")
            })?;
            (meta.state, meta.size_bytes, meta.backing_file.clone())
        };

        if state == PageState::PagedOut {
            self.total_page_faults += 1;
            let backing_file = backing_file.ok_or_else(|| {
                io::Error::new(io::ErrorKind::NotFound, "No backing file")
            })?;

            while self.ram_used_bytes + size > self.config.ram_budget_bytes {
                if !self.evict_lru()? { break; }
            }

            let mut file = File::open(&backing_file)?;
            let mut data = vec![0u8; size];
            file.read_exact(&mut data)?;

            if let Some(meta) = self.pages.get_mut(&page_id) {
                meta.state = PageState::Resident;
                meta.backing_file = None;
                meta.last_access = Instant::now();
                meta.access_count += 1;
            }
            self.ram_used_bytes += size;
            self.data.insert(page_id, data);
        } else {
            self.total_hits += 1;
            if let Some(meta) = self.pages.get_mut(&page_id) {
                meta.last_access = Instant::now();
                meta.access_count += 1;
            }
        }

        self.data.get(&page_id).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, "Page data not found")
        })
    }

    pub fn access_mut(&mut self, page_id: PageId) -> io::Result<&mut Vec<u8>> {
        self.access(page_id)?;
        if let Some(meta) = self.pages.get_mut(&page_id) {
            meta.dirty = true;
        }
        self.data.get_mut(&page_id).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, "Page data not found")
        })
    }

    pub fn page_out(&mut self, page_id: PageId) -> io::Result<()> {
        let (state, size, dirty, backing_file_exists) = {
            let meta = self.pages.get(&page_id).ok_or_else(|| {
                io::Error::new(io::ErrorKind::NotFound, "Page not found")
            })?;
            (meta.state, meta.size_bytes, meta.dirty, meta.backing_file.is_some())
        };

        if state != PageState::Resident { return Ok(()); }

        if dirty || !backing_file_exists {
            let data = self.data.get(&page_id).ok_or_else(|| {
                io::Error::new(io::ErrorKind::NotFound, "Page data not found")
            })?;

            let backing_path = self.config.backing_dir.join(format!("page_{}.bin", page_id));
            let mut file = File::create(&backing_path)?;
            file.write_all(data)?;

            if let Some(meta) = self.pages.get_mut(&page_id) {
                meta.backing_file = Some(backing_path);
            }
        }

        if let Some(meta) = self.pages.get_mut(&page_id) {
            meta.state = PageState::PagedOut;
            meta.dirty = false;
        }

        self.ram_used_bytes -= size;
        self.data.remove(&page_id);
        Ok(())
    }

    pub fn free(&mut self, page_id: PageId) -> io::Result<()> {
        if let Some(meta) = self.pages.get(&page_id) {
            if let Some(ref backing_file) = meta.backing_file {
                let _ = fs::remove_file(backing_file);
            }
            if meta.state == PageState::Resident {
                self.ram_used_bytes -= meta.size_bytes;
            }
        }
        self.pages.remove(&page_id);
        self.data.remove(&page_id);
        Ok(())
    }

    fn evict_lru(&mut self) -> io::Result<bool> {
        let lru_id = self.pages.iter()
            .filter(|(_, meta)| meta.state == PageState::Resident)
            .min_by_key(|(_, meta)| meta.last_access)
            .map(|(id, _)| *id);

        match lru_id {
            Some(id) => {
                self.page_out(id)?;
                self.total_evictions += 1;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    pub fn stats(&self) -> PagerStats {
        let resident = self.pages.values()
            .filter(|m| m.state == PageState::Resident)
            .count();
        
        PagerStats {
            total_pages: self.pages.len(),
            resident_pages: resident,
            paged_out_pages: self.pages.len() - resident,
            ram_used_bytes: self.ram_used_bytes,
            ram_budget_bytes: self.config.ram_budget_bytes,
            total_page_faults: self.total_page_faults,
            total_evictions: self.total_evictions,
            total_hits: self.total_hits,
        }
    }

    pub fn ram_usage(&self) -> usize { self.ram_used_bytes }
    
    pub fn resident_count(&self) -> usize {
        self.pages.values().filter(|m| m.state == PageState::Resident).count()
    }

    pub fn flush(&mut self) -> io::Result<()> {
        let dirty_ids: Vec<PageId> = self.pages.iter()
            .filter(|(_, meta)| meta.dirty && meta.state == PageState::Resident)
            .map(|(id, _)| *id)
            .collect();

        for id in dirty_ids { self.page_out(id)?; }
        Ok(())
    }
}

impl Drop for MemoryPager {
    fn drop(&mut self) {
        for (_, meta) in &self.pages {
            if let Some(ref backing_file) = meta.backing_file {
                let _ = fs::remove_file(backing_file);
            }
        }
    }
}