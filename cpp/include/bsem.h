#include <mutex>


// https://stackoverflow.com/a/4793662
class bsem { // Binary Semaphore
    std::mutex mutex_;
    std::condition_variable condition_;
    bool free;
public:
    bsem() : free(false) {}
    bsem(bool free) : free(free) {}
    void release() {
        std::lock_guard<decltype(mutex_)> lock(mutex_);
        free = true;
        condition_.notify_one();
    }

    void acquire() {
        std::unique_lock<decltype(mutex_)> lock(mutex_);
        while (!free) // Handle spurious wake-ups.
            condition_.wait(lock);
        free = false;
    }

    bool try_acquire() {
        std::lock_guard<decltype(mutex_)> lock(mutex_);
        if (free) {
            free = false;
            return true;
        }
        return false;
    }
};