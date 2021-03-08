// SpscFragmentedArrayQueue<T> is a unbounded multi-producer, single-consumer queue.

#pragma once

#include <atomic>
#include <limits>
#include <memory>
#include <type_traits>
#include <variant>

#include "flow/Platform.h"

inline size_t pow2Rank(size_t n) {
    return std::numeric_limits<size_t>::digits - 1 - clzll(n);
}

template <typename T>
class SpscFragmentedArrayQueue  {
    struct jump {
        using element_type = std::variant<T, jump>;
        element_type* to;
    };

    using element_type = typename jump::element_type;

    // Layout taking into account cache line size and prefetching.
    struct alignas(CACHE_LINE_SIZE * 2) {
        std::atomic<size_t> index;
        size_t mask;
        element_type* buffer;
        // Cached exclusive index until which we known we can push items,
        // without needing to load the consumer index.
        size_t coordFreeLimit;
    } producer;

    struct alignas(CACHE_LINE_SIZE * 2) {
        std::atomic<size_t> index;
        size_t mask;
        element_type* buffer;
    } consumer;

private:
    void doPush(T&& data, element_type* buffer, size_t idx, size_t offset) {
        new (buffer + offset) element_type(std::forward<T>(data));
        producer.index.store(idx + 1, std::memory_order_release);
    }

    // Slow path requiring coordination with the consumer. There's at least one slot left.
    void pushSlowPath(T&& data, element_type* buffer, size_t idx, size_t mask, size_t offset) {
        auto consumerIdx = consumer.index.load(std::memory_order_acquire);
        auto ahead = idx - consumerIdx;
        if (ahead < mask) {
            producer.coordFreeLimit = idx + mask - ahead;
            doPush(std::forward<T>(data), buffer, idx, offset);
        } else {
            // There's a single slot left. Link to a new buffer of the same size.
            // The consumer will cleanup the old buffer.
            auto newBuffer = new element_type[mask + 1];
            producer.buffer = newBuffer;
            new (buffer + offset) element_type(jump{newBuffer});
            producer.coordFreeLimit = idx + mask;
            doPush(std::forward<T>(data), newBuffer, idx, offset);
        }
    }

public:
    SpscFragmentedArrayQueue(size_t initialSize = 1024) {
        auto p2initialSize = 1 << pow2Rank(initialSize);
        auto mask = p2initialSize - 1;
        auto buffer = new element_type[p2initialSize];

        // Each index is private.
        producer.index = 0;
        consumer.index = 0;

        // The mask is immutable, but hold copies to fit evertyhing onto the same producer or consumer cache line.
        producer.mask = mask;
        consumer.mask = mask;

        // The producer may be some amount of buffers ahead of the consumer.
        producer.buffer = buffer;
        consumer.buffer = buffer;

        // The last index is the position that will hold the link to the next buffer.
        producer.coordFreeLimit = mask;
    }

    SpscFragmentedArrayQueue(const SpscFragmentedArrayQueue&) = delete;
	SpscFragmentedArrayQueue& operator=(const SpscFragmentedArrayQueue&) = delete;

    void push(T&& data) {
        auto idx = producer.index.load(std::memory_order_relaxed); // Only written from this thread.
        auto mask = producer.mask;
        auto offset = idx & mask;
        if (idx < producer.coordFreeLimit) {
            doPush(std::forward<T>(data), producer.buffer, idx, offset);
        } else {
            pushSlowPath(std::forward<T>(data), producer.buffer, idx, mask, offset);
        }
    }

    template <typename Func>
    int pop(Func f) {
        auto buf = consumer.buffer;
        auto initialIdx = consumer.index.load(std::memory_order_relaxed); // Only written from this thread.
        auto idx = initialIdx;
        auto mask = consumer.mask;
        auto producerIdx = producer.index.load(std::memory_order_acquire);
        while (idx < producerIdx) {
            std::visit([&] (auto&& v) {
                using V = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<V, jump>) {
                    buf = v.to;
                    delete consumer.buffer;
                    consumer.buffer = buf;
                } else {
                    f(std::move(v));
                    ++idx;
                }
            }, buf[idx & mask]);
        }
        consumer.index.store(producerIdx, std::memory_order_release);
        return producerIdx - initialIdx;
    }
};
