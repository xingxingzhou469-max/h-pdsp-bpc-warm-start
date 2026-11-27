#include <algorithm>
#include <chrono>
#include <cstdint>
#include <vector>

using Val = std::int64_t;
using Clock = std::chrono::steady_clock;
constexpr Val missing = -(Val(1) << 61);
constexpr std::size_t limit_cells = 18000000;

struct Arc { int to, cost, bit; };

static int drop(int mask, int own) {
    return (mask & (own - 1)) | ((mask >> 1) & ~(own - 1));
}

extern "C" int mask_price(
    int n, int h, int layered, int marked, Val nanos, const int* arcs,
    const int* start, const int* finish, const int* region,
    const int* tag, const Val* reward, Val* maxima, int* lengths, int* paths
) {
    if (n <= 0 || n > 256 || h < 0 || h > 640 || marked < 1 || marked > 4) return 3;
    const auto deadline = Clock::now() + std::chrono::nanoseconds(nanos);
    const int count = 1 << marked;
    const int layers = layered ? n : 1, steps = layered ? n : h + 1;
    std::vector<std::vector<int>> groups(n);
    for (int i = 0; i < n; ++i) groups[region[i]].push_back(i);
    std::vector<int> own(n), offsets(n + 1);
    std::vector<std::vector<Arc>> in(n), out(n);
    for (int i = 0; i < n; ++i) {
        const auto& group = groups[region[i]];
        if (group.size() > 12) return 3;
        offsets[i + 1] = offsets[i] + (1 << (group.size() - 1));
        for (unsigned b = 0; b < group.size(); ++b) {
            const int j = group[b], bit = 1 << b;
            const int cost = arcs[i * n + j];
            if (i == j) own[i] = bit;
            else if (cost <= h) in[i].push_back({j, cost, bit});
        }
        for (int j = 0; j < n; ++j)
            if (region[j] != region[i] && arcs[i * n + j] <= h)
                out[i].push_back({j, arcs[i * n + j], 0});
        std::sort(out[i].begin(), out[i].end(), [](const Arc& a, const Arc& b) {
            return a.cost < b.cost;
        });
    }
    const std::size_t frame = offsets[n], rows = std::size_t(layers) * (h + 1) * count;
    if (rows * (frame + 2 * n) > limit_cells) return 3;
    std::vector<Val> best(rows * frame, missing), alt(rows * n, missing);
    std::vector<int> blocked(rows * n, -1);
    const auto clock = [h, count](int layer, int remaining, int mask) {
        return (std::size_t(layer) * (h + 1) + remaining) * count + mask;
    };
    const auto singleton = [&](std::size_t row, int i, int prev) {
        return prev >= 0 && blocked[row * n + i] == prev
            ? alt[row * n + i] : best[row * frame + offsets[i]];
    };
    for (int layer = 0; layer < layers; ++layer) {
        for (int t = 0; t <= h; ++t) {
            if (Clock::now() >= deadline) return 2;
            for (int mask = 0; mask < count; ++mask) {
                const auto row = clock(layer, t, mask);
                for (int i = 0; i < n; ++i) {
                    if (tag[i] && !(mask & tag[i])) continue;
                    const int rest = mask ^ tag[i];
                    Val first = !rest && finish[i] <= t ? 0 : missing, second = missing;
                    int excluded = -1;
                    auto* values = best.data() + row * frame + offsets[i];
                    const int width = offsets[i + 1] - offsets[i];
                    if (layered && !layer) {
                        std::fill_n(values, width, first);
                        alt[row * n + i] = first;
                        continue;
                    }
                    for (const Arc& a : out[i]) {
                        if (a.cost > t) break;
                        Val v = singleton(clock(layer - layered, t - a.cost, rest), a.to, i);
                        if (v == missing) continue;
                        v += reward[a.to];
                        if (v > first) {
                            second = first;
                            first = v;
                            excluded = a.to;
                        } else if (v > second) {
                            second = v;
                        }
                    }
                    const int low = own[i] - 1;
                    for (int s = 0; s < width; ++s) {
                        const int local = (s & low) | ((s & ~low) << 1) | own[i];
                        Val inside = missing;
                        for (const Arc& a : in[i]) {
                            if (a.cost > t || (local & a.bit)) continue;
                            const auto pos = clock(layer - layered, t - a.cost, rest);
                            const auto v = best[pos * frame + offsets[a.to]
                                + drop(local | a.bit, own[a.to])];
                            if (v != missing) inside = std::max(inside, v + reward[a.to]);
                        }
                        values[s] = std::max(inside, first);
                        if (!s) alt[row * n + i] = std::max(inside, second);
                    }
                    blocked[row * n + i] = excluded;
                }
            }
        }
    }
    for (int target = 0; target < count; ++target) {
        int root = -1;
        Val top = target == 0 ? 0 : missing;
        for (int i = 0; i < n; ++i) {
            if (start[i] > h) continue;
            const Val v = singleton(clock(layers - 1, h - start[i], target), i, -1);
            if (v != missing && reward[i] + v > top) {
                top = reward[i] + v;
                root = i;
            }
        }
        maxima[target] = top;
        lengths[target] = 0;
        if (root < 0) continue;
        int time = h - start[root], layer = layers - 1, mask = target;
        int at = root, prev = -1, local = own[root];
        Val value = top - reward[root];
        int* path = paths + target * steps;
        path[lengths[target]++] = root;
        while (true) {
            const int rest = mask ^ tag[at];
            if (!rest && finish[at] <= time && value == 0) break;
            if (layered && !layer) return 4;
            bool found = false;
            for (const Arc& a : in[at]) {
                if (a.cost > time || (local & a.bit)) continue;
                const auto row = clock(layer - layered, time - a.cost, rest);
                const int next = local | a.bit;
                const Val v = best[row * frame + offsets[a.to] + drop(next, own[a.to])];
                if (v == missing || reward[a.to] + v != value) continue;
                prev = at;
                at = a.to;
                local = next;
                time -= a.cost;
                value = v;
                found = true;
                break;
            }
            if (!found) {
                for (const Arc& a : out[at]) {
                    if (a.cost > time) break;
                    if (a.to == prev) continue;
                    const Val v = singleton(clock(layer - layered, time - a.cost, rest), a.to, at);
                    if (v == missing || reward[a.to] + v != value) continue;
                    prev = at;
                    at = a.to;
                    local = own[at];
                    time -= a.cost;
                    value = v;
                    found = true;
                    break;
                }
            }
            if (!found || lengths[target] >= steps) return 4;
            mask = rest;
            layer -= layered;
            path[lengths[target]++] = at;
        }
    }
    return 0;
}
