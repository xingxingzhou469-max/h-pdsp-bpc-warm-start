#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <queue>
#include <unordered_map>
#include <vector>

using Value = std::int64_t;
using Clock = std::chrono::steady_clock;
constexpr Value missing = -(Value(1) << 61);

struct Choice {
    Value value = missing;
    int next = -2;
};

struct Tail {
    Choice first, second;

    Choice best_except(int blocked) const {
        return blocked >= 0 && blocked == first.next ? second : first;
    }

    void add(Value value, int next) {
        if (value > first.value) {
            second = first;
            first = {value, next};
        } else if (value > second.value) {
            second = {value, next};
        }
    }
};

struct StateKey {
    std::array<std::uint64_t, 4> visited{};
    int end = -1;

    bool operator==(const StateKey& other) const {
        return end == other.end && visited == other.visited;
    }

    bool contains(int i) const {
        return visited[i / 64] & (std::uint64_t(1) << (i % 64));
    }

    void insert(int i) {
        visited[i / 64] |= std::uint64_t(1) << (i % 64);
        end = i;
    }
};

struct StateHash {
    std::size_t operator()(const StateKey& key) const {
        std::uint64_t hash = key.end + 1;
        for (auto word : key.visited) {
            word ^= word >> 30;
            word *= 0xbf58476d1ce4e5b9ULL;
            word ^= word >> 27;
            word *= 0x94d049bb133111ebULL;
            hash ^= word ^ (word >> 31);
            hash = (hash << 19) | (hash >> 45);
        }
        return hash;
    }
};

struct Prefix {
    StateKey key;
    Value reward;
    int previous;
    int remaining;
    int depth;
    int parent;
};

// tails only use projected clocks and omit repetitions beyond the two-cycle
// rule. The optional elementary phase builds a covering frontier within them.
extern "C" int route_price(
    int n, int horizon, int layered, int state_limit, Value time_ns,
    const int* arcs, const int* start, const int* finish, const Value* reward,
    Value* upper, int* lengths, int* paths, int* expanded
) {
    const auto deadline = Clock::now() + std::chrono::nanoseconds(time_ns);
    if (n <= 0 || n > 256) return 3;
    const int layers = layered ? n : 1;
    const int width = layered ? n : horizon + 1;
    const std::size_t plane = std::size_t(n) * (horizon + 1);
    if (plane * layers > 7500000) return 3;
    std::vector<Tail> tails(plane * layers);
    const auto cell = [&](int left, int remaining, int i) -> Tail& {
        return tails[plane * left + std::size_t(remaining) * n + i];
    };
    std::vector<std::vector<int>> outgoing(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            if (arcs[i * n + j] <= horizon) outgoing[i].push_back(j);
        std::sort(outgoing[i].begin(), outgoing[i].end(), [&](int a, int b) {
            return arcs[i * n + a] < arcs[i * n + b];
        });
    }
    for (int left = 0; left < layers; ++left) {
        const int after = layered ? left - 1 : 0;
        for (int t = 0; t <= horizon; ++t) {
            if (Clock::now() >= deadline) return 2;
            for (int i = 0; i < n; ++i) {
                Tail& tail = cell(left, t, i);
                if (finish[i] <= t) tail.add(0, -1);
                if (after < 0) continue;
                for (int j : outgoing[i]) {
                    const int rest = t - arcs[i * n + j];
                    if (rest < 0) break;
                    const Value child = cell(after, rest, j).best_except(i).value;
                    if (child != missing) tail.add(reward[j] + child, j);
                }
            }
        }
    }
    std::fill(lengths, lengths + n + 1, 0);
    *upper = 0;
    *expanded = 0;
    Value incumbent = 0;
    for (int i = 0; i < n; ++i) {
        if (start[i] > horizon) continue;
        int left = layers - 1;
        int remaining = horizon - start[i];
        Choice tail = cell(left, remaining, i).best_except(-1);
        if (tail.value == missing) continue;
        const Value value = reward[i] + tail.value;
        *upper = std::max(*upper, value);
        int* path = paths + i * width;
        path[lengths[i]++] = i;
        StateKey visited;
        visited.insert(i);
        bool simple = true;
        for (int node = i; tail.next >= 0; ) {
            const int next = tail.next;
            remaining -= arcs[node * n + next];
            left -= layered;
            tail = cell(left, remaining, next).best_except(node);
            node = next;
            path[lengths[i]++] = node;
            simple &= !visited.contains(node);
            visited.insert(node);
        }
        if (simple && value > incumbent) {
            incumbent = value;
            lengths[n] = lengths[i];
            std::copy_n(path, lengths[i], paths + n * width);
        }
    }
    if (!state_limit || incumbent == *upper) return 0;

    std::vector<Prefix> prefixes;
    std::priority_queue<std::pair<Value, int>> frontier;
    std::unordered_map<StateKey, int, StateHash> earliest;
    const auto push = [&](const Prefix& prefix) {
        const int i = prefix.key.end;
        const int left = layered ? n - prefix.depth : 0;
        const Value tail = cell(left, prefix.remaining, i).best_except(prefix.previous).value;
        if (tail == missing) return;
        const Value bound = prefix.reward + tail;
        if (bound <= incumbent) return;
        auto [known, new_key] = earliest.try_emplace(prefix.key, prefix.remaining);
        if (!new_key && prefix.remaining <= known->second) return;
        known->second = prefix.remaining;
        frontier.emplace(bound, static_cast<int>(prefixes.size()));
        prefixes.push_back(prefix);
    };
    for (int i = 0; i < n; ++i) {
        if (start[i] > horizon) continue;
        Prefix root{StateKey{}, reward[i], -1, horizon - start[i], 1, -1};
        root.key.insert(i);
        push(root);
    }

    while (!frontier.empty() && frontier.top().first > incumbent
           && static_cast<int>(prefixes.size()) < state_limit && Clock::now() < deadline) {
        const int label = frontier.top().second;
        frontier.pop();
        const Prefix prefix = prefixes[label];
        if (earliest.at(prefix.key) > prefix.remaining) continue;
        ++*expanded;
        const int i = prefix.key.end;
        if (finish[i] <= prefix.remaining && prefix.reward > incumbent) {
            incumbent = prefix.reward;
            lengths[n] = prefix.depth;
            int* path = paths + n * width;
            int id = label;
            for (int pos = prefix.depth - 1; pos >= 0; --pos) {
                path[pos] = prefixes[id].key.end;
                id = prefixes[id].parent;
            }
        }
        if (prefix.depth >= n) continue;
        for (int j : outgoing[i]) {
            const int rest = prefix.remaining - arcs[i * n + j];
            if (rest < 0) break;
            if (prefix.key.contains(j)) continue;
            Prefix next{prefix.key, prefix.reward + reward[j], i, rest, prefix.depth + 1, label};
            next.key.insert(j);
            push(next);
        }
    }
    *upper = frontier.empty() ? incumbent : std::max(incumbent, frontier.top().first);
    return 0;
}
