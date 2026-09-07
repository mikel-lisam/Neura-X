/* ==========================================================
 * Neura-X: Intelligence Without Limits.
 * Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
 *
 * Windows Virtual Memory — VirtualAlloc/VirtualFree wrappers
 * ========================================================== */

#include "memory.h"

#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <stdio.h>

void *nex_platform_alloc(size_t size_bytes)
{
    if (size_bytes == 0) return NULL;

    /* Use VirtualAlloc for large allocations */
    if (size_bytes >= 4096) {
        void *ptr = VirtualAlloc(NULL, size_bytes,
                                 MEM_COMMIT | MEM_RESERVE,
                                 PAGE_READWRITE);
        return ptr;
    }

    /* Use HeapAlloc for small allocations */
    return HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, size_bytes);
}

void nex_platform_free(void *ptr, size_t size_bytes)
{
    if (!ptr) return;

    if (size_bytes >= 4096) {
        VirtualFree(ptr, 0, MEM_RELEASE);
    } else {
        HeapFree(GetProcessHeap(), 0, ptr);
    }
}

void *nex_mmap_file(const char *path, size_t size_bytes, bool writable)
{
    if (!path) return NULL;

    DWORD access = GENERIC_READ;
    DWORD share = FILE_SHARE_READ;
    DWORD creation = OPEN_EXISTING;
    DWORD protect = PAGE_READONLY;
    DWORD map_access = FILE_MAP_READ;

    if (writable) {
        access = GENERIC_READ | GENERIC_WRITE;
        protect = PAGE_READWRITE;
        map_access = FILE_MAP_ALL_ACCESS;
    }

    HANDLE file = CreateFileA(path, access, share, NULL,
                              creation, FILE_ATTRIBUTE_NORMAL, NULL);
    if (file == INVALID_HANDLE_VALUE) return NULL;

    /* Get file size if not specified */
    if (size_bytes == 0) {
        LARGE_INTEGER file_size;
        if (GetFileSizeEx(file, &file_size)) {
            size_bytes = (size_t)file_size.QuadPart;
        }
    }

    if (size_bytes == 0) {
        CloseHandle(file);
        return NULL;
    }

    HANDLE mapping = CreateFileMappingA(file, NULL, protect,
                                        (DWORD)(size_bytes >> 32),
                                        (DWORD)(size_bytes & 0xFFFFFFFF),
                                        NULL);
    if (!mapping) {
        CloseHandle(file);
        return NULL;
    }

    void *ptr = MapViewOfFile(mapping, map_access, 0, 0, size_bytes);

    CloseHandle(mapping);
    CloseHandle(file);

    return ptr;
}

void nex_munmap_file(void *ptr, size_t size_bytes)
{
    (void)size_bytes;
    if (ptr) {
        UnmapViewOfFile(ptr);
    }
}

#endif /* _WIN32 */