#ifndef SCOPED_TIMER_H
#define SCOPED_TIMER_H

#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>

#ifdef ENABLE_TIMING

class ScopedTimer {
public:
    explicit ScopedTimer(const std::string& label)
        : label_(label), start_(std::chrono::high_resolution_clock::now()) {}

    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start_).count();
        // Format locally to avoid polluting cout's global format state.
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.2f", ms);
        std::cout << "  [TIME] " << label_ << ": " << buf << " ms" << std::endl;
    }

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    std::string label_;
    std::chrono::high_resolution_clock::time_point start_;
};

#define PROFILE_SCOPE(name) ScopedTimer _timer_##__LINE__(name)

#else

#define PROFILE_SCOPE(name)

#endif // ENABLE_TIMING

#endif // SCOPED_TIMER_H
