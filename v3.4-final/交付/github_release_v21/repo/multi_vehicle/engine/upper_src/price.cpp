#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

using I = std::int64_t;
using Mask = std::uint64_t;
constexpr I missing = std::numeric_limits<I>::min() / 4;

struct Label {
    I value = missing;
    int from = -1;
    int trace = -1;
    int incoming_load = 0;
};

struct Best {
    Label first, second;

    void add(Label candidate, bool cycle2) {
        if (candidate.value > first.value) {
            if (cycle2 && candidate.from != first.from) second = first;
            first = candidate;
        } else if (cycle2 && candidate.from != first.from && candidate.value > second.value) {
            second = candidate;
        }
    }

    void merge(const Best& other, bool cycle2) {
        add(other.first, cycle2);
        if (cycle2) add(other.second, true);
    }

    void shift(I delta) {
        if (first.value != missing) first.value += delta;
        if (second.value != missing) second.value += delta;
    }

    bool covers(const Label& candidate, I clock_price, bool cycle2) const {
        if (candidate.value == missing) return true;
        const I net = candidate.value - clock_price;
        return first.value >= net &&
            (!cycle2 || first.from == candidate.from || second.value >= net);
    }

    void remember(Label candidate, I clock_price, bool cycle2) {
        if (candidate.value == missing) return;
        candidate.value -= clock_price;
        add(candidate, cycle2);
    }
};

struct Trail {
    int previous, node, amount;
};

struct Group {
    Mask memory;
    int next;
    int history;
};

struct Arc {
    int target;
    int duration;
    int completion;
};

struct Key {
    Mask memory;
    int base;

    bool operator==(const Key& other) const {
        return memory == other.memory && base == other.base;
    }
};

struct KeyHash {
    std::size_t operator()(const Key& key) const {
        std::uint64_t x = key.memory + 0x9e3779b97f4a7c15ULL * (1ULL + key.base);
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        return x ^ (x >> 31);
    }
};

// 0: exact optimum, 1: state cap, 2: deadline. Only status 0 may be certified.
// One exact best column per terminal agency, with the idle route implicit.
extern "C" int price(
    int n, int horizon, int m, int layered, int cycle2, int sparse, int state_cap, I remaining_ns,
    const int* arcs, const int* home, const int* pickups, const I* reward,
    I load_reward, I time_price, const Mask* neighborhoods, const Mask* own_bit,
    I* values, int* lengths, int* used_time, int* stops, int* amounts, int* reached_states,
    Mask marked_bits, int class_count, I* class_values
) {
    using Clock = std::chrono::steady_clock;
    const auto deadline = Clock::now() + std::chrono::nanoseconds(remaining_ns);
    const int width = m + 1;
    const int plane = (horizon + 1) * n;
    const int bases = (layered ? n + 1 : 1) * plane;
    const int max_steps = layered ? n : horizon + 1;
    if (bases > state_cap || (!sparse && bases > state_cap / width)) return 1;

    // Shortest completion to any final agency, ignoring loads and forbidden visits.
    std::vector<int> exit_cost(n, horizon + 1);
    std::vector<bool> settled(n, false);
    for (int i = 0; i < n; ++i) {
        if (pickups[i] < 0) exit_cost[i] = home[i];
    }
    for (int step = 0; step < n; ++step) {
        int target = -1;
        for (int i = 0; i < n; ++i) {
            if (!settled[i] && (target < 0 || exit_cost[i] < exit_cost[target])) target = i;
        }
        settled[target] = true;
        for (int i = 0; i < n; ++i) {
            exit_cost[i] = std::min(exit_cost[i], arcs[i * n + target] + exit_cost[target]);
        }
    }
    std::vector<std::vector<Arc>> successors(n);
    for (int node = 0; node < n; ++node) {
        auto& outgoing = successors[node];
        for (int next = 0; next < n; ++next) {
            if (next == node) continue;
            const int duration = arcs[node * n + next];
            const int completion = duration + exit_cost[next];
            if (completion <= horizon) outgoing.push_back({next, duration, completion});
        }
        std::sort(outgoing.begin(), outgoing.end(), [](const Arc& a, const Arc& b) {
            if (a.completion != b.completion) return a.completion < b.completion;
            return a.target < b.target;
        });
    }

    std::vector<Best> arrival(sparse ? 0 : bases * width);
    const int stations = (layered ? n + 1 : 1) * n;
    std::vector<Best> history(sparse ? 0 : stations * width);
    std::vector<int> heads(sparse ? bases : 0, -1);
    std::vector<Group> groups;
    std::unordered_map<Key, int, KeyHash> group_index;
    std::unordered_map<Key, int, KeyHash> history_index;
    std::vector<Trail> trails;
    std::vector<int> endpoints(n, -1);
    std::vector<Best> after(width), prefix(width), suffix(width);
    std::fill(values, values + n, missing);
    std::fill(lengths, lengths + n, 0);
    if (class_count) {
        std::fill(class_values, class_values + class_count, missing);
        class_values[0] = 0; // The idle route has no visited markers.
    }

    const auto group = [&](int base, Mask memory) -> int {
        if (!sparse) return base;
        const Key key{memory, base};
        const auto found = group_index.find(key);
        if (found != group_index.end()) return found->second;
        if (static_cast<int>(groups.size()) == state_cap / width) return -1;
        const int id = static_cast<int>(groups.size());
        const int station = (base / plane) * n + base % n;
        const auto [history_cell, inserted] = history_index.try_emplace(
            Key{memory, station}, static_cast<int>(history_index.size()));
        if (inserted) history.resize(history_index.size() * width);
        group_index.emplace(key, id);
        groups.push_back({memory, heads[base], history_cell->second});
        heads[base] = id;
        arrival.resize((id + 1) * width);
        return id;
    };
    for (int node = 0; node < n; ++node) {
        const int time = arcs[n * n + node];
        if (pickups[node] < 0 || time + exit_cost[node] > horizon) continue;
        const int id = group(layered * plane + time * n + node, own_bit[node]);
        if (id < 0) return 1;
        arrival[id * width].first = {0, -1, -1, 0};
    }

    // Process a DAG of projected clocks, or visit layers when zero arcs exist.
    for (int layer = layered; layer <= (layered ? n : 0); ++layer) {
        for (int time = 0; time <= horizon; ++time) {
            for (int node = 0; node < n; ++node) {
                const int base = layer * plane + time * n + node;
                int id = sparse ? heads[base] : base;
                while (id >= 0) {
                    if (Clock::now() >= deadline) return 2;
                    const Mask memory = sparse ? groups[id].memory : 0;
                    const Best* before = &arrival[id * width];
                    if (m == 0) {
                        after[0] = before[0];
                    } else if (pickups[node] < 0) {
                        Best best;
                        for (int load = m; load >= 0; --load) {
                            Best candidate = before[load];
                            candidate.shift(load_reward * load);
                            best.merge(candidate, cycle2);
                            after[load] = best;
                            after[load].shift(-load_reward * load);
                        }
                    } else if (pickups[node] == 0) {
                        std::copy(before, before + width, after.begin());
                    } else {
                        // Fixed-width sliding interval [load-pickup, load].
                        // Block prefix/suffix aggregates merge top two distinct predecessors.
                        const int block = pickups[node] + 1;
                        for (int load = 0; load <= m; ++load) {
                            prefix[load] = before[load];
                            if (load % block != 0) prefix[load].merge(prefix[load - 1], cycle2);
                        }
                        for (int load = m; load >= 0; --load) {
                            suffix[load] = before[load];
                            if (load < m && (load + 1) % block != 0) {
                                suffix[load].merge(suffix[load + 1], cycle2);
                            }
                        }
                        for (int load = 0; load <= m; ++load) {
                            after[load] = prefix[load];
                            if (load >= block) after[load].merge(suffix[load - block + 1], cycle2);
                        }
                    }

                    Best available;
                    for (int load = 0; load <= m; ++load) {
                        Best& best = after[load];
                        const int history_row = sparse ? groups[id].history : layer * n + node;
                        Best& earlier = history[history_row * width + load];
                        const I clock_price = time_price * time;
                        const bool drop_first = earlier.covers(best.first, clock_price, cycle2);
                        const bool drop_second = earlier.covers(best.second, clock_price, cycle2);
                        if (!drop_first) earlier.remember(best.first, clock_price, cycle2);
                        if (!drop_second) earlier.remember(best.second, clock_price, cycle2);
                        if (drop_first) best.first = drop_second ? Label{} : best.second;
                        if (drop_first || drop_second) best.second = {};
                        best.shift(reward[node]);
                        for (Label* label : {&best.first, &best.second}) {
                            if (label->value == missing) continue;
                            const int amount = std::abs(load - label->incoming_load);
                            trails.push_back({label->trace, node, amount});
                            label->trace = static_cast<int>(trails.size() - 1);
                        }
                        available.merge(best, cycle2);
                    }
                    if (available.first.value != missing) {
                        const int done = time + home[node];
                        if (pickups[node] < 0 && done <= horizon && after[0].first.value != missing) {
                            const I candidate = after[0].first.value - time_price * done;
                            if (class_count) {
                                int signature = 0, output_bit = 1;
                                for (Mask bits = marked_bits; bits; bits &= bits - 1) {
                                    if (memory & (bits & -bits)) signature |= output_bit;
                                    output_bit <<= 1;
                                }
                                class_values[signature] = std::max(class_values[signature], candidate);
                            }
                            if (candidate > values[node]) {
                                values[node] = candidate;
                                endpoints[node] = after[0].first.trace;
                                used_time[node] = done;
                            }
                        }
                        if (!layered || layer < n) {
                            for (const Arc& arc : successors[node]) {
                                if (time + arc.completion > horizon) break;
                                const int next = arc.target;
                                if (own_bit[next] & memory) continue;
                                if (cycle2 && available.first.from == next &&
                                    available.second.value == missing) continue;
                                const int when = time + arc.duration;
                                const int target = group((layer + layered) * plane + when * n + next,
                                                         (memory & neighborhoods[next]) | own_bit[next]);
                                if (target < 0) return 1;
                                for (int load = 0; load <= m; ++load) {
                                    const auto& best = after[load];
                                    const Label& source = cycle2 && best.first.from == next
                                        ? best.second : best.first;
                                    if (source.value == missing) continue;
                                    arrival[target * width + load].add(
                                        {source.value, node, source.trace, load}, cycle2);
                                }
                            }
                        }
                    }
                    id = sparse ? groups[id].next : -1;
                }
            }
        }
    }

    *reached_states = sparse ? static_cast<int>(arrival.size()) : bases * width;
    for (int end = 0; end < n; ++end) {
        for (int at = endpoints[end]; at >= 0; at = trails[at].previous) {
            const int slot = end * max_steps + lengths[end]++;
            stops[slot] = trails[at].node;
            amounts[slot] = trails[at].amount;
        }
    }
    return 0;
}
