#include <algorithm>
#include <chrono>
#include <cstdint>
#include <vector>

using Integer = std::int64_t;
using Clock = std::chrono::steady_clock;
constexpr Integer missing = -(Integer(1) << 61);

struct Arc { int to, time; };
int compact(int mask, int bit) {
    return (mask & (bit - 1)) | ((mask >> 1) & ~(bit - 1));
}

// A backward route price remembering the required suppliers, current cell
// visits, and the immediately preceding node across cell boundaries.
extern "C" int sparse_price(int n, int h, int group, Integer time_ns, const int* cost,
                            const int* supplier, const int* region, const Integer* rewards,
                            const Integer* prefix, Integer* bound, int* length, int* path) {
    const auto stop = Clock::now() + std::chrono::nanoseconds(time_ns);
    std::vector<std::vector<int>> cells(n);
    std::vector<bool> allowed(n);
    for (int i = 1; i < n; ++i) {
        allowed[i] = !supplier[i] || (supplier[i] & group);
        if (allowed[i]) cells[region[i]].push_back(i);
    }
    std::vector<std::vector<Arc>> in(n), out(n);
    std::vector<int> bit(n), width(n), slot(n), mask_index(group + 1);
    unsigned states = 0;
    for (int mask = 0; mask <= group; ++mask)
        if (!(mask & ~group)) mask_index[mask] = states++;
    int frame = 0;
    for (int i = 1; i < n; ++i) {
        if (!allowed[i]) continue;
        const auto& cell = cells[region[i]];
        if (cell.size() > 6) return 3;
        bit[i] = 1 << (std::find(cell.begin(), cell.end(), i) - cell.begin());
        width[i] = 1 << (cell.size() - 1);
        slot[i] = frame;
        frame += width[i];
        for (int j = 1; j < n; ++j) {
            const int time = cost[i * n + j];
            if (i == j || !allowed[j] || time > h) continue;
            if (!time) return 3;
            (region[i] == region[j] ? in[i] : out[i]).push_back({j, time});
        }
        auto by_time = [](Arc a, Arc b) { return a.time < b.time; };
        std::sort(in[i].begin(), in[i].end(), by_time);
        std::sort(out[i].begin(), out[i].end(), by_time);
    }
    const auto row = [&](int time, int mask) { return std::size_t(time) * states + mask_index[mask]; };
    const std::size_t rows = (h + 1) * states;
    if (rows * (frame + 2 * n) > 10'000'000) return 3;
    std::vector<Integer> value(rows * frame, missing), alternate(rows * n, missing);
    std::vector<int> blocked(rows * n, -1);

    for (int t = 0; t <= h; ++t) {
        if (Clock::now() >= stop) return 2;
        for (int mask = 0; mask <= group; ++mask) {
            if (mask & ~group) continue;
            const auto r = row(t, mask);
            const auto* credit = mask == group ? prefix : rewards;
            for (int i = 1; i < n; ++i) {
                if (!allowed[i] || (mask & supplier[i])) continue;
                Integer first = !mask && cost[i * n] <= t ? 0 : missing;
                Integer second = missing;
                int forbidden = -1;
                for (const auto [j, duration] : out[i]) {
                    if (duration > t) break;
                    if (supplier[j] && !(mask & supplier[j])) continue;
                    const auto next = row(t - duration, mask ^ supplier[j]);
                    auto v = blocked[next * n + j] == i
                        ? alternate[next * n + j] : value[next * frame + slot[j]];
                    if (v == missing) continue;
                    v += credit[j];
                    if (v > first) {
                        second = first;
                        first = v;
                        forbidden = j;
                    } else second = std::max(second, v);
                }
                const int lower = bit[i] - 1;
                for (int local = 0; local < width[i]; ++local) {
                    const int seen = (local & lower) | ((local & ~lower) << 1) | bit[i];
                    Integer best_inside = missing;
                    for (const auto [j, duration] : in[i]) {
                        if (duration > t) break;
                        if (seen & bit[j]) continue;
                        const auto next = row(t - duration, mask);
                        const auto v = value[next * frame + slot[j] + compact(seen | bit[j], bit[j])];
                        if (v != missing) best_inside = std::max(best_inside, v + credit[j]);
                    }
                    value[r * frame + slot[i] + local] = std::max(best_inside, first);
                    if (!local) alternate[r * n + i] = std::max(second, best_inside);
                }
                blocked[r * n + i] = forbidden;
            }
        }
    }

    *bound = 0;
    *length = 0;
    int i = 0, time = 0, mask = group;
    for (int j = 1; j < n; ++j) {
        if (!allowed[j] || cost[j] > h) continue;
        const auto r = row(h - cost[j], group ^ supplier[j]);
        const auto v = value[r * frame + slot[j]];
        const auto prize = prefix[j];
        if (v != missing && v + prize > *bound) {
            *bound = v + prize;
            i = j;
            time = h - cost[j];
            mask = group ^ supplier[j];
        }
    }
    if (!i) return 0;
    int seen = bit[i], previous = -1;
    auto target = *bound - prefix[i];
    path[(*length)++] = i;
    while (mask || cost[i * n] > time || target != 0) {
        const auto* credit = mask == group ? prefix : rewards;
        bool found = false;
        for (const auto [j, duration] : in[i]) {
            if (duration > time) break;
            if (seen & bit[j]) continue;
            const auto child = row(time - duration, mask);
            const auto v = value[child * frame + slot[j] + compact(seen | bit[j], bit[j])];
            if (v == missing || v + credit[j] != target) continue;
            previous = i;
            i = j;
            seen |= bit[j];
            target = v;
            time -= duration;
            found = true;
            break;
        }
        if (!found) {
            for (const auto [j, duration] : out[i]) {
                if (duration > time) break;
                if (j == previous || (supplier[j] && !(mask & supplier[j]))) continue;
                const auto child = row(time - duration, mask ^ supplier[j]);
                const auto v = blocked[child * n + j] == i
                    ? alternate[child * n + j] : value[child * frame + slot[j]];
                if (v == missing || v + credit[j] != target) continue;
                previous = i;
                i = j;
                seen = bit[j];
                target = v;
                mask ^= supplier[j];
                time -= duration;
                found = true;
                break;
            }
        }
        if (!found || *length > h) return 4;
        path[(*length)++] = i;
    }
    return 0;
}
