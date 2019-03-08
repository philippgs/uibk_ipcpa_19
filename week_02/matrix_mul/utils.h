#pragma once

#include <time.h>

// add a pseudo-bool type
typedef int bool;
#define true  (0==0)
#define false (0!=0)


// a small wrapper for convenient time measurements

typedef double timestamp;

timestamp now() {
    struct timespec spec;
    timespec_get(&spec, TIME_UTC);
    return spec.tv_sec + spec.tv_nsec / (1e9);
}
