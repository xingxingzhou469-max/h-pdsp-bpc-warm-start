#include <algorithm>
#include <chrono>
#include <cstdint>
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

extern "C" int zone_price(
    int n, int h, int layered, V time_ns, const int* arcs, const int* start,
    const int* finish, const int* region, const V* rewards, V* upper, int* lengths, int* paths
) {
    if (n <= 0 || n > 256 || h < 0 || h > 640) return 3;
    const auto deadline = Clock::now() + std::chrono::nanoseconds(time_ns);
    const int layers = layered ? n : 1;
    const int steps = layered ? n : h + 1;
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
    *upper = 0;
    std::fill_n(lengths, n, 0);
    for (int root = 0; root < n; ++root) {
        if (start[root] > h) continue;
        int time = h - start[root];
        int left = layers - 1, i = root, memory = own[i], prev = -1;
        V target = singleton(index(left, time), root, -1);
        if (target == missing) continue;
        *upper = std::max(*upper, rewards[root] + target);
        int* path = paths + root * steps;
        path[lengths[root]++] = i;
        while (true) {
            if (finish[i] <= time && target == 0) break;
            if (layered && !left) return 4;
            bool found = false;
            for (const Move& a : inner[i]) {
                if (a.time > time || (memory & a.bit)) continue;
                const auto sub = index(left - layered, time - a.time);
                const int next = memory | a.bit;
                const auto v = vals[sub * frame + offsets[a.j] + compressed(next, own[a.j])];
                if (v == missing || rewards[a.j] + v != target) continue;
                prev = i;
                i = a.j;
                memory = next;
                time -= a.time;
                target = v;
                found = true;
                break;
            }
            if (!found) {
                for (const Move& a : outer[i]) {
                    if (a.time > time) break;
                    if (a.j == prev) continue;
                    const auto v = singleton(index(left - layered, time - a.time), a.j, i);
                    if (v == missing || rewards[a.j] + v != target) continue;
                    prev = i;
                    i = a.j;
                    memory = own[i];
                    time -= a.time;
                    target = v;
                    found = true;
                    break;
                }
            }
            if (!found || lengths[root] == steps) return 4;
            left -= layered;
            path[lengths[root]++] = i;
        }
    }
    return 0;
}
