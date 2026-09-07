/* ==========================================================
 * Neura-X: Intelligence Without Limits.
 * Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
 *
 * Linux Memory Mapping — mmap/munmap wrappers
 * ========================================================== */

#include "memory.h"

#ifdef __linux__

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

void *nex_platform_alloc(size_t size_bytes)
{
    if (size_bytes == 0) return NULL;

    /* Use mmap for large allocations (better for paging) */
    if (size_bytes >= 4096) {
        void *ptr = mmap(NULL, size_bytes,
                         PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS,
                         -1, 0);
        if (ptr == MAP_FAILED) return NULL;

        /* Advise the kernel that this memory will be accessed sequentially */
        madvise(ptr, size_bytes, MADV_SEQUENTIAL);

        return ptr;
    }

    /* Use malloc for small allocations */
    return calloc(1, size_bytes);
}

void nex_platform_free(void *ptr, size_t size_bytes)
{
    if (!ptr) return;

    if (size_bytes >= 4096) {
        munmap(ptr, size_bytes);
    } else {
        free(ptr);
    }
}

void *nex_mmap_file(const char *path, size_t size_bytes, bool writable)
{
    if (!path || size_bytes == 0) return NULL;

    int flags = O_RDONLY;
    int prot = PROT_READ;

    if (writable) {
        flags = O_RDWR;
        prot = PROT_READ | PROT_WRITE;
    }

    int fd = open(path, flags);
    if (fd < 0) return NULL;

    /* Get actual file size if size_bytes is 0 */
    if (size_bytes == 0) {
        struct stat st;
        if (fstat(fd, &st) == 0) {
            size_bytes = (size_t)st.st_size;
        }
    }

    if (size_bytes == 0) {
        close(fd);
        return NULL;
    }

    void *ptr = mmap(NULL, size_bytes, prot, MAP_SHARED, fd, 0);
    close(fd); /* fd can be closed after mmap */

    if (ptr == MAP_FAILED) return NULL;

    /* Advise random access for model files */
    madvise(ptr, size_bytes, MADV_RANDOM);

    return ptr;
}

void nex_munmap_file(void *ptr, size_t size_bytes)
{
    if (ptr && size_bytes > 0) {
        munmap(ptr, size_bytes);
    }
}

#else /* Non-Linux fallback */

void *nex_platform_alloc(size_t size_bytes)
{
    if (size_bytes == 0) return NULL;
    return calloc(1, size_bytes);
}

void nex_platform_free(void *ptr, size_t size_bytes)
{
    (void)size_bytes;
    free(ptr);
}

void *nex_mmap_file(const char *path, size_t size_bytes, bool writable)
{
    if (!path) return NULL;

    const char *mode = writable ? "rb+" : "rb";
    FILE *f = fopen(path, mode);
    if (!f) {
        if (writable) f = fopen(path, "wb+");
        if (!f) return NULL;
    }

    if (size_bytes == 0) {
        fseek(f, 0, SEEK_END);
        size_bytes = (size_t)ftell(f);
        fseek(f, 0, SEEK_SET);
    }

    void *ptr = malloc(size_bytes);
    if (!ptr) {
        fclose(f);
        return NULL;
    }

    size_t bytes_read = fread(ptr, 1, size_bytes, f);
    fclose(f);

    if (bytes_read != size_bytes) {
        free(ptr);
        return NULL;
    }

    return ptr;
}

void nex_munmap_file(void *ptr, size_t size_bytes)
{
    (void)size_bytes;
    free(ptr);
}

#endif /* __linux__ */