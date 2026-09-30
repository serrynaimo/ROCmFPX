#pragma once

// --gpu-keepalive-ms N: keep the GPUs this server uses from idling into their powered-down state.
//
// Some Windows GPU drivers (AMD 32.0.31007..31041, incl. PRO 26.Q3) evict a device's whole VRAM into system RAM seconds
// after it has no work and no display to drive - headless, display off, or an RDP session rendered on the CPU. The model
// then runs out of (or thrashes) system RAM. A tiny device memset every N ms keeps each device busy enough to stay up.
//
// Uses only the generic backend API (buffer clear), so it works for any GPU backend. The CUDA/HIP clear runs on the
// per-thread default stream and graph capture is relaxed, so it never syncs with or disturbs the model's own streams.

#include "ggml-backend.h"
#include "log.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

struct server_gpu_keepalive {
    void start(const std::vector<ggml_backend_dev_t> & requested, int interval_ms) {
        if (interval_ms <= 0) { return; }
        std::vector<ggml_backend_dev_t> devs = requested;
        if (devs.empty()) {   // no -dev given: every GPU this process can see
            for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                ggml_backend_dev_t d = ggml_backend_dev_get(i);
                if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) { devs.push_back(d); }
            }
        }
        for (auto * d : devs) {
            if (!d || ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_CPU) { continue; }
            ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(d), 64 * 1024);
            if (buf) {
                bufs.push_back(buf);
                LOG_INF("gpu keep-alive: %s every %d ms\n", ggml_backend_dev_name(d), interval_ms);
            }
        }
        if (bufs.empty()) { return; }
        th = std::thread([this, interval_ms] {
            uint8_t v = 0;
            std::unique_lock<std::mutex> lock(mtx);
            while (!cv.wait_for(lock, std::chrono::milliseconds(interval_ms), [this] { return stopping.load(); })) {
                for (auto * b : bufs) { ggml_backend_buffer_clear(b, v); }
                v++;
            }
        });
    }

    // must run before the backends are freed
    void stop() {
        { std::lock_guard<std::mutex> lock(mtx); stopping = true; }
        cv.notify_all();
        if (th.joinable()) { th.join(); }
        for (auto * b : bufs) { ggml_backend_buffer_free(b); }
        bufs.clear();
    }

    ~server_gpu_keepalive() { stop(); }

private:
    std::vector<ggml_backend_buffer_t> bufs;
    std::thread th;
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<bool> stopping { false };
};
