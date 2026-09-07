/* ==========================================================
 * Neura-X: Intelligence Without Limits.
 * Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
 *
 * Memory Pager — LRU page management between RAM and disk
 * ========================================================== */

#include "memory.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NEX_MAX_PAGES 4096
#define NEX_PAGE_FILE_PREFIX "nex_page_"
#define NEX_PATH_BUF_SIZE 512

/* ──────────────────────────────────────────────────────────
 * Internal Pager Structure
 * ────────────────────────────────────────────────────────── */

struct nex_pager {
    nex_memory_page_t *pages;
    size_t             max_pages;
    size_t             page_count;
    size_t             ram_budget_bytes;
    size_t             ram_used_bytes;
    char               backing_dir[NEX_PATH_BUF_SIZE];
    uint64_t           total_page_faults;
    uint64_t           total_evictions;
};

/* ──────────────────────────────────────────────────────────
 * Internal Helpers
 * ────────────────────────────────────────────────────────── */

static uint64_t nex_get_time_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000 + (uint64_t)ts.tv_nsec / 1000000;
}

static void nex_page_file_path(const nex_pager_t *pager, uint64_t page_id,
                               char *out, size_t out_size)
{
    int written = snprintf(out, out_size, "%s/%s%llu.dat",
                           pager->backing_dir, NEX_PAGE_FILE_PREFIX,
                           (unsigned long long)page_id);
    /* Ensure null termination even if truncated */
    if (written < 0 || (size_t)written >= out_size) {
        out[out_size - 1] = '\0';
    }
}

/* Find the LRU resident page for eviction */
static int64_t nex_find_lru_page(const nex_pager_t *pager)
{
    int64_t lru_index = -1;
    uint64_t oldest_time = UINT64_MAX;

    for (size_t i = 0; i < pager->max_pages; i++) {
        if (pager->pages[i].data == NULL) continue;
        if (pager->pages[i].state != NEX_PAGE_RESIDENT) continue;
        if (pager->pages[i].last_access_time < oldest_time) {
            oldest_time = pager->pages[i].last_access_time;
            lru_index = (int64_t)i;
        }
    }

    return lru_index;
}

/* Write a page to disk */
static int nex_write_page_to_disk(nex_pager_t *pager, nex_memory_page_t *page)
{
    char path[NEX_PATH_BUF_SIZE];
    nex_page_file_path(pager, page->page_id, path, sizeof(path));

    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    size_t written = fwrite(page->data, 1, page->size_bytes, f);
    fclose(f);

    if (written != page->size_bytes) return -1;

    strncpy(page->backing_file, path, sizeof(page->backing_file) - 1);
    page->backing_file[sizeof(page->backing_file) - 1] = '\0';

    return 0;
}

/* Read a page from disk */
static int nex_read_page_from_disk(nex_pager_t *pager, nex_memory_page_t *page)
{
    (void)pager; /* backing_file path is stored in page, not pager */

    if (page->backing_file[0] == '\0') return -1;

    FILE *f = fopen(page->backing_file, "rb");
    if (!f) return -1;

    size_t bytes_read = fread(page->data, 1, page->size_bytes, f);
    fclose(f);

    if (bytes_read != page->size_bytes) return -1;

    return 0;
}

/* Evict a page to make room */
static int nex_evict_page(nex_pager_t *pager)
{
    int64_t lru = nex_find_lru_page(pager);
    if (lru < 0) return -1;

    nex_memory_page_t *page = &pager->pages[lru];

    /* Write to disk if dirty */
    if (page->dirty) {
        if (nex_write_page_to_disk(pager, page) != 0) return -1;
    }

    /* Free RAM */
    nex_platform_free(page->data, page->size_bytes);
    page->data = NULL;
    page->state = NEX_PAGE_PAGED_OUT;
    pager->ram_used_bytes -= page->size_bytes;
    pager->total_evictions++;

    return 0;
}

/* ──────────────────────────────────────────────────────────
 * Public API
 * ────────────────────────────────────────────────────────── */

nex_pager_t *nex_pager_create(size_t ram_budget_bytes, const char *backing_dir)
{
    nex_pager_t *pager = (nex_pager_t *)calloc(1, sizeof(nex_pager_t));
    if (!pager) return NULL;

    pager->pages = (nex_memory_page_t *)calloc(NEX_MAX_PAGES,
                                                sizeof(nex_memory_page_t));
    if (!pager->pages) {
        free(pager);
        return NULL;
    }

    pager->max_pages = NEX_MAX_PAGES;
    pager->page_count = 0;
    pager->ram_budget_bytes = ram_budget_bytes;
    pager->ram_used_bytes = 0;
    pager->total_page_faults = 0;
    pager->total_evictions = 0;

    if (backing_dir) {
        strncpy(pager->backing_dir, backing_dir, sizeof(pager->backing_dir) - 1);
        pager->backing_dir[sizeof(pager->backing_dir) - 1] = '\0';
    } else {
        strncpy(pager->backing_dir, "/tmp/neura-x", sizeof(pager->backing_dir) - 1);
        pager->backing_dir[sizeof(pager->backing_dir) - 1] = '\0';
    }

    return pager;
}

void nex_pager_destroy(nex_pager_t *pager)
{
    if (!pager) return;

    for (size_t i = 0; i < pager->max_pages; i++) {
        nex_memory_page_t *page = &pager->pages[i];
        if (page->data) {
            if (page->dirty) {
                nex_write_page_to_disk(pager, page);
            }
            nex_platform_free(page->data, page->size_bytes);
        }
        /* Clean up backing file */
        if (page->backing_file[0] != '\0') {
            remove(page->backing_file);
        }
    }

    free(pager->pages);
    free(pager);
}

int64_t nex_pager_alloc(nex_pager_t *pager, size_t size_bytes)
{
    if (!pager || size_bytes == 0) return -1;

    /* Find a free slot */
    int64_t slot = -1;
    for (size_t i = 0; i < pager->max_pages; i++) {
        if (pager->pages[i].data == NULL && pager->pages[i].size_bytes == 0) {
            slot = (int64_t)i;
            break;
        }
    }

    if (slot < 0) return -1;

    /* Check RAM budget and evict if needed */
    while (pager->ram_used_bytes + size_bytes > pager->ram_budget_bytes) {
        if (nex_evict_page(pager) != 0) break;
    }

    /* Allocate memory */
    void *data = nex_platform_alloc(size_bytes);
    if (!data) return -1;
    memset(data, 0, size_bytes);

    /* Initialize page */
    nex_memory_page_t *page = &pager->pages[slot];
    page->page_id = (uint64_t)slot;
    page->data = data;
    page->size_bytes = size_bytes;
    page->state = NEX_PAGE_RESIDENT;
    page->last_access_time = nex_get_time_ms();
    page->access_count = 0;
    page->dirty = false;
    page->backing_file[0] = '\0';

    pager->ram_used_bytes += size_bytes;
    pager->page_count++;

    return slot;
}

void *nex_pager_access(nex_pager_t *pager, uint64_t page_id)
{
    if (!pager || page_id >= pager->max_pages) return NULL;

    nex_memory_page_t *page = &pager->pages[page_id];

    if (page->state == NEX_PAGE_PAGED_OUT) {
        /* Page fault: load from disk */
        pager->total_page_faults++;

        /* Allocate RAM for the page */
        while (pager->ram_used_bytes + page->size_bytes > pager->ram_budget_bytes) {
            if (nex_evict_page(pager) != 0) return NULL;
        }

        page->data = nex_platform_alloc(page->size_bytes);
        if (!page->data) return NULL;

        if (nex_read_page_from_disk(pager, page) != 0) {
            nex_platform_free(page->data, page->size_bytes);
            page->data = NULL;
            return NULL;
        }

        page->state = NEX_PAGE_RESIDENT;
        pager->ram_used_bytes += page->size_bytes;
    }

    page->last_access_time = nex_get_time_ms();
    page->access_count++;

    return page->data;
}

void nex_pager_mark_dirty(nex_pager_t *pager, uint64_t page_id)
{
    if (!pager || page_id >= pager->max_pages) return;
    pager->pages[page_id].dirty = true;
}

int nex_pager_page_out(nex_pager_t *pager, uint64_t page_id)
{
    if (!pager || page_id >= pager->max_pages) return -1;

    nex_memory_page_t *page = &pager->pages[page_id];
    if (page->state != NEX_PAGE_RESIDENT || !page->data) return -1;

    if (nex_write_page_to_disk(pager, page) != 0) return -1;

    nex_platform_free(page->data, page->size_bytes);
    page->data = NULL;
    page->state = NEX_PAGE_PAGED_OUT;
    pager->ram_used_bytes -= page->size_bytes;

    return 0;
}

void nex_pager_free(nex_pager_t *pager, uint64_t page_id)
{
    if (!pager || page_id >= pager->max_pages) return;

    nex_memory_page_t *page = &pager->pages[page_id];

    if (page->data) {
        nex_platform_free(page->data, page->size_bytes);
        pager->ram_used_bytes -= page->size_bytes;
    }

    if (page->backing_file[0] != '\0') {
        remove(page->backing_file);
    }

    memset(page, 0, sizeof(nex_memory_page_t));
    if (pager->page_count > 0) {
        pager->page_count--;
    }
}

size_t nex_pager_ram_usage(const nex_pager_t *pager)
{
    return pager ? pager->ram_used_bytes : 0;
}

size_t nex_pager_resident_count(const nex_pager_t *pager)
{
    if (!pager) return 0;
    size_t count = 0;
    for (size_t i = 0; i < pager->max_pages; i++) {
        if (pager->pages[i].state == NEX_PAGE_RESIDENT &&
            pager->pages[i].data != NULL) {
            count++;
        }
    }
    return count;
}

size_t nex_pager_total_count(const nex_pager_t *pager)
{
    return pager ? pager->page_count : 0;
}

void nex_memory_get_stats(nex_memory_stats_t *stats)
{
    if (!stats) return;
    memset(stats, 0, sizeof(nex_memory_stats_t));

#if defined(__linux__)
    FILE *f = fopen("/proc/meminfo", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            unsigned long long value = 0;
            if (sscanf(line, "MemTotal: %llu kB", &value) == 1) {
                stats->total_ram_bytes = (size_t)value * 1024;
            }
            if (sscanf(line, "MemAvailable: %llu kB", &value) == 1) {
                stats->available_ram_bytes = (size_t)value * 1024;
            }
        }
        fclose(f);
    }
#elif defined(__APPLE__)
    /* Use sysctl on macOS */
    #include <sys/sysctl.h>
    int mib[2] = {CTL_HW, HW_MEMSIZE};
    uint64_t memsize = 0;
    size_t len = sizeof(memsize);
    if (sysctl(mib, 2, &memsize, &len, NULL, 0) == 0) {
        stats->total_ram_bytes = (size_t)memsize;
    }
    stats->available_ram_bytes = stats->total_ram_bytes;
#endif
}