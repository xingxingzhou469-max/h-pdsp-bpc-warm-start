#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using V = std::int64_t;
using Clock = std::chrono::steady_clock;
constexpr V missing = -(V(1) << 61);
constexpr std::size_t budget_cells = 18500000;

static int compressed(int mask, int own) {
    return (mask & (own - 1)) | ((mask >> 1) & ~(own - 1));
}

struct Move {
    int j, time, bit;
};

struct Mask {
    std::uint64_t lo = 0, hi = 0;
    bool operator==(const Mask& other) const {
        return lo == other.lo && hi == other.hi;
    }
    bool contains(int i) const {
        return ((i < 64 ? lo : hi) >> (i % 64)) & 1;
    }
    Mask with(int i) const {
        auto copy = *this;
        (i < 64 ? copy.lo : copy.hi) |= std::uint64_t(1) << (i % 64);
        return copy;
    }
};

struct MaskHash {
    std::size_t operator()(const Mask& mask) const {
        return mask.lo * 0x9e3779b185ebca87ULL ^ mask.hi * 0xc2b2ae3d27d4eb4fULL;
    }
};

struct Position {
    Mask mask;
    int tail;
    bool operator==(const Position& other) const { return tail == other.tail && mask == other.mask; }
};

struct PositionHash {
    std::size_t operator()(const Position& pos) const {
        return MaskHash{}(pos.mask) ^ (std::uint64_t(pos.tail) * 0x165667b19e3779f9ULL);
    }
};

extern "C" int zone_screen(
    int n, int h, int layered, V time_ns, const int* arcs, const int* start,
    const int* finish, const int* region, const V* rewards, V threshold, int max_masks,
    int* count, std::uint64_t* output, std::uint64_t* explored
) {
    if (n <= 0 || n > 128 || h < 0 || h > 640) return 3;
    const auto deadline = Clock::now() + std::chrono::nanoseconds(time_ns);
    const int layers = layered ? n : 1;
    std::vector<std::vector<int>> groups(n);
    for (int i = 0; i < n; ++i) groups[region[i]].push_back(i);
    std::vector<int> offsets(n + 1), own(n), width(n);
    std::vector<std::vector<Move>> inner(n), outer(n);
    for (int i = 0; i < n; ++i) {
        const auto& group = groups[region[i]];
        if (group.size() > 12) return 3;
        width[i] = 1 << (group.size() - 1);
        offsets[i + 1] = offsets[i] + width[i];
        for (unsigned b = 0; b < group.size(); ++b) {
            const int j = group[b], bit = 1 << b;
            if (j == i) own[i] = bit;
            else if (arcs[i * n + j] <= h) inner[i].push_back({j, arcs[i * n + j], bit});
        }
        for (int j = 0; j < n; ++j)
            if (region[j] != region[i] && arcs[i * n + j] <= h)
                outer[i].push_back({j, arcs[i * n + j], 0});
        std::sort(outer[i].begin(), outer[i].end(), [](const Move& a, const Move& b) {
            return a.time < b.time;
        });
    }
    const std::size_t frame = offsets[n];
    const std::size_t rows = std::size_t(layers) * (h + 1);
    if (rows * (frame + 2 * n) > budget_cells) return 3;
    std::vector<V> vals(rows * frame, missing), other(rows * n, missing);
    std::vector<int> excluded(rows * n, -1);
    const auto index = [h](int left, int time) { return std::size_t(left) * (h + 1) + time; };
    const auto singleton = [&](std::size_t row, int j, int prev) {
        return prev >= 0 && excluded[row * n + j] == prev
            ? other[row * n + j] : vals[row * frame + offsets[j]];
    };
    for (int left = 0; left < layers; ++left) {
        for (int time = 0; time <= h; ++time) {
            if (Clock::now() >= deadline) return 2;
            const auto row = index(left, time);
            for (int i = 0; i < n; ++i) {
                const V terminal = finish[i] <= time ? 0 : missing;
                V largest = terminal, second = missing;
                int first = -1;
                V* const dp = vals.data() + row * frame + offsets[i];
                if (layered && !left) {
                    std::fill_n(dp, width[i], terminal);
                    other[row * n + i] = terminal;
                    continue;
                }
                for (const Move& a : outer[i]) {
                    if (a.time > time) break;
                    auto value = singleton(index(left - layered, time - a.time), a.j, i);
                    if (value == missing) continue;
                    value += rewards[a.j];
                    if (value > largest) {
                        second = largest;
                        largest = value;
                        first = a.j;
                    } else if (value > second) {
                        second = value;
                    }
                }
                const int high = own[i], lower = high - 1;
                V solo_inner = missing;
                for (int compact = 0; compact < width[i]; ++compact) {
                    const int mask = ((compact & ~lower) << 1) | (compact & lower) | high;
                    V inside = missing;
                    for (const Move& a : inner[i]) {
                        if (a.time > time || (mask & a.bit)) continue;
                        const auto sub = index(left - layered, time - a.time);
                        const auto value = vals[sub * frame + offsets[a.j]
                            + compressed(mask | a.bit, own[a.j])];
                        if (value != missing) inside = std::max(inside, value + rewards[a.j]);
                    }
                    dp[compact] = std::max(largest, inside);
                    if (!compact) solo_inner = inside;
                }
                other[row * n + i] = std::max(second, solo_inner);
                excluded[row * n + i] = first;
            }
        }
    }
    std::unordered_set<Mask, MaskHash> masks;
    std::unordered_map<Position, int, PositionHash> dominance;
    *count = 0;
    *explored = 0;
    int status = 0;
    auto keep = [&](Mask mask) {
        if (!masks.insert(mask).second) return;
        if (*count == max_masks) {
            status = 3;
            return;
        }
        output[2 * *count] = mask.lo;
        output[2 * *count + 1] = mask.hi;
        ++*count;
    };
    if (threshold <= 0) keep(Mask{});
    auto search = [&](auto&& self, int i, int prev, int memory, int left,
                      int time, Mask visited, V score) -> void {
        if (status) return;
        if (((++*explored) & 1023) == 0 && Clock::now() >= deadline) {
            status = 2;
            return;
        }
        const auto row = index(left, time);
        const int state = compressed(memory, own[i]);
        V suffix = vals[row * frame + offsets[i] + state];
        if (!state && prev >= 0 && region[prev] != region[i]
                && excluded[row * n + i] == prev) {
            suffix = other[row * n + i];
        }
        if (suffix == missing || score + suffix < threshold) return;
        const Position key{visited, i};
        auto prior = dominance.find(key);
        if (prior != dominance.end()) {
            if (prior->second >= time) return;
            prior->second = time;
        } else if (dominance.size() < 2500000) {
            dominance.emplace(key, time);
        }
        if (finish[i] <= time && score >= threshold) keep(visited);
        if (status || (layered && !left)) return;
        for (const auto& move : inner[i]) {
            if (move.time <= time && !visited.contains(move.j)) {
                self(self, move.j, i, memory | move.bit, left - layered, time - move.time,
                     visited.with(move.j), score + rewards[move.j]);
            }
        }
        for (const auto& move : outer[i]) {
            if (move.time > time) break;
            if (!visited.contains(move.j)) {
                self(self, move.j, i, own[move.j], left - layered, time - move.time,
                     visited.with(move.j), score + rewards[move.j]);
            }
        }
    };
    for (int root = 0; root < n; ++root) {
        if (status || Clock::now() >= deadline) return status ? status : 2;
        if (start[root] > h) continue;
        search(search, root, -1, own[root], layers - 1, h - start[root],
               Mask{}.with(root), rewards[root]);
    }
    return status;
}
