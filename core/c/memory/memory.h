/* ==========================================================
 * Neura-X: Intelligence Without Limits.
 * Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
 *
 * Memory Management — Header
 * ========================================================== */

#ifndef NEURA_X_MEMORY_H
#define NEURA_X_MEMORY_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ──────────────────────────────────────────────────────────
 * Memory Page Structure
 * ────────────────────────────────────────────────────────── */

typedef enum {
    NEX_PAGE_RESIDENT = 0,   /* In RAM */
    NEX_PAGE_PAGED_OUT,      /* On disk */
    NEX_PAGE_LOADING,        /* Being loaded from disk */
    NEX_PAGE_EVICTING        /* Being written to disk */
} nex_page_state_t;

typedef struct {
    uint64_t        page_id;
    void           *data;
    size_t          size_bytes;
    nex_page_state_t state;
    uint64_t        last_access_time;
    uint32_t        access_count;
    bool            dirty;
    char            backing_file[256];
} nex_memory_page_t;

/* ──────────────────────────────────────────────────────────
 * Memory Pager
 * ────────────────────────────────────────────────────────── */

typedef struct nex_pager nex_pager_t;

/* Create a memory pager with the given RAM budget and backing directory. */
nex_pager_t *nex_pager_create(size_t ram_budget_bytes, const char *backing_dir);

/* Destroy the pager and free all resources. */
void nex_pager_destroy(nex_pager_t *pager);

/* Allocate a new page. Returns page_id or -1 on failure. */
int64_t nex_pager_alloc(nex_pager_t *pager, size_t size_bytes);

/* Access a page, loading it from disk if necessary. Returns pointer or NULL. */
void *nex_pager_access(nex_pager_t *pager, uint64_t page_id);

/* Mark a page as dirty (modified). */
void nex_pager_mark_dirty(nex_pager_t *pager, uint64_t page_id);

/* Explicitly page out a page to disk. */
int nex_pager_page_out(nex_pager_t *pager, uint64_t page_id);

/* Free a page. */
void nex_pager_free(nex_pager_t *pager, uint64_t page_id);

/* Get current RAM usage. */
size_t nex_pager_ram_usage(const nex_pager_t *pager);

/* Get number of resident pages. */
size_t nex_pager_resident_count(const nex_pager_t *pager);

/* Get total number of pages. */
size_t nex_pager_total_count(const nex_pager_t *pager);

/* ──────────────────────────────────────────────────────────
 * Platform-Specific Memory Allocation
 * ────────────────────────────────────────────────────────── */

/* Allocate memory using platform-native API. */
void *nex_platform_alloc(size_t size_bytes);

/* Free memory allocated with nex_platform_alloc. */
void nex_platform_free(void *ptr, size_t size_bytes);

/* Map a file into memory. Returns pointer or NULL on failure. */
void *nex_mmap_file(const char *path, size_t size_bytes, bool writable);

/* Unmap a previously mapped file. */
void nex_munmap_file(void *ptr, size_t size_bytes);

/* ──────────────────────────────────────────────────────────
 * Memory Statistics
 * ────────────────────────────────────────────────────────── */

typedef struct {
    size_t total_ram_bytes;
    size_t available_ram_bytes;
    size_t pager_ram_used_bytes;
    size_t pager_disk_used_bytes;
    size_t resident_pages;
    size_t paged_out_pages;
    uint64_t total_page_faults;
    uint64_t total_evictions;
} nex_memory_stats_t;

/* Get memory statistics. */
void nex_memory_get_stats(nex_memory_stats_t *stats);

#ifdef __cplusplus
}
#endif

#endif /* NEURA_X_MEMORY_H */