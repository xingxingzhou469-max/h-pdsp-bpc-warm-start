// Rebuild-first fleet search. Double quantities rank proposals; the caller
// reconstructs canonical rational quantities and verifies each witness.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <random>
#include <sstream>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using Route = std::vector<int>;
using Fleet = std::vector<Route>;
using Publish = void (*)(const char*, const char*);
constexpr int max_nodes = 126;
using Visited = std::array<bool, max_nodes + 1>;
constexpr double eps = 1e-9;

struct Level { double rate, people; };
struct Run { double supply, people; };
struct Rank { double objective, late, elapsed; int broken; };
struct Column {
    std::uint32_t mask;
    Route route;
    std::vector<Level> profile;
    double score, root;
};

struct Search {
    int n, vehicles;
    double capacity, limit, population = 0, supply_scale = 0;
    std::vector<double> supply, people, home;
    std::vector<std::vector<double>> arc;
    std::vector<int> suppliers, agencies, active;
    Clock::time_point deadline;
    std::mt19937 rng;
    Fleet best;
    double best_score = 0;
    Publish callback;

    int pick(int count) { return static_cast<int>(rng() % count); }
    bool done(Clock::time_point end) const { return Clock::now() >= end; }
    double edge(int from, int to) const { return to ? arc[from][to] : home[from]; }

    double time_of(const Route& route) const {
        double time = 0;
        int previous = 0;
        for (int node : route) {
            time += arc[previous][node];
            previous = node;
        }
        return time + home[previous];
    }

    bool shorten(Route& route, Clock::time_point end, bool preserve_profile) const {
        const int size = route.size();
        if (size < 2) return false;
        bool shortened = false;
        while (!done(end)) {
            std::array<double, max_nodes> reversal{};
            std::array<int, max_nodes> block{};
            for (int i = 1; i < size; ++i) {
                // Reversing an internal directed arc can change its cost.
                reversal[i] = reversal[i - 1] + edge(route[i], route[i - 1])
                                              - edge(route[i - 1], route[i]);
                block[i] = block[i - 1] + (bool(supply[route[i]]) != bool(supply[route[i - 1]]));
            }
            double best_delta = -eps;
            int reverse_first = -1, reverse_last = -1;
            int move_first = -1, move_after = -1, move_gap = -1;
            for (int first = 0; first + 1 < size; ++first) {
                const int before = first ? route[first - 1] : 0;
                for (int last = first + 1; last < size; ++last) {
                    if (preserve_profile && block[first] != block[last]) break;
                    if ((!first && !supply[route[last]]) || (last == size - 1 && !people[route[first]]))
                        continue;
                    const int after = last + 1 < size ? route[last + 1] : 0;
                    const double delta = edge(before, route[last]) + edge(route[first], after)
                                       - edge(before, route[first]) - edge(route[last], after)
                                       + reversal[last] - reversal[first];
                    if (delta < best_delta) {
                        best_delta = delta;
                        reverse_first = first;
                        reverse_last = last;
                    }
                }
            }
            for (int length = 1; length <= 3 && length < size; ++length) {
                for (int first = 0; first + length <= size; ++first) {
                    const int after = first + length;
                    const int left = first ? route[first - 1] : 0;
                    const int right = after < size ? route[after] : 0;
                    const int head = route[first], tail = route[after - 1];
                    const double removal = edge(left, right) - edge(left, head) - edge(tail, right);
                    for (int gap = 0; gap <= size; ++gap) {
                        if (gap >= first && gap <= after) continue;
                        // Moving within a contiguous same-type block keeps each
                        // pickup and delivery run, and hence the allocation profile.
                        if (preserve_profile && block[std::min(gap, first)]
                            != block[std::max(gap, after) - 1]) continue;
                        const int new_front = gap == 0 ? head : (first == 0 ? right : route.front());
                        const int new_back = gap == size ? tail : (after == size ? left : route.back());
                        if (!supply[new_front] || !people[new_back]) continue;
                        const int before = gap ? route[gap - 1] : 0;
                        const int next = gap < size ? route[gap] : 0;
                        const double delta = removal + edge(before, head) + edge(tail, next)
                                          - edge(before, next);
                        if (delta < best_delta) {
                            best_delta = delta;
                            move_first = first;
                            move_after = after;
                            move_gap = gap;
                        }
                    }
                }
            }
            if (move_first >= 0) {
                if (move_gap < move_first)
                    std::rotate(route.begin() + move_gap, route.begin() + move_first, route.begin() + move_after);
                else
                    std::rotate(route.begin() + move_first, route.begin() + move_after, route.begin() + move_gap);
            } else if (reverse_first >= 0) {
                std::reverse(route.begin() + reverse_first, route.begin() + reverse_last + 1);
            } else {
                break;
            }
            shortened = true;
        }
        return shortened;
    }

    // One two-state chord cut for the fixed-route weighted polymatroid.
    // This is the floating-point proposal counterpart of allocation.py.
    static std::uint64_t chord(const Run* runs, int count, double capacity,
                               std::uint64_t fixed, std::uint64_t allowed,
                               double rate, double target) {
        double outside = 0, inside = 0;
        std::uint64_t out_mask = 0, in_mask = 0;
        for (int i = 0; i < count; ++i) {
            const std::uint64_t bit = std::uint64_t{1} << i;
            for (int sink = 0; sink < 2; ++sink) {
                double next = outside;
                std::uint64_t mask = out_mask;
                if (inside + capacity < next) {
                    next = inside + capacity;
                    mask = in_mask;
                }
                if (outside < inside) {
                    inside = outside;
                    in_mask = out_mask;
                }
                if (sink && (fixed & bit)) inside = INFINITY;
                else if (sink && (allowed & bit)) inside += rate * runs[i].people;
                outside = next + (sink ? 0 : runs[i].supply);
                out_mask = mask | (sink ? bit & allowed : 0);
            }
        }
        return std::min(outside, inside) < target - eps
                   ? (outside <= inside ? out_mask : in_mask) : allowed;
    }

    void split(const Run* runs, int size, std::uint64_t left, std::uint64_t right,
               double q_left, double q_right, double p_left, double p_right,
               Level* levels, int& count) const {
        const double persons = p_right - p_left;
        const double rate = (q_right - q_left) / persons;
        const auto block = left ^ right;
        if (block & (block - 1)) {
            const auto middle = chord(runs, size, capacity, left, right, rate, q_right);
            if (middle != left && middle != right) {
                double quantity = 0, people_mid = 0, load = 0;
                for (int i = 0; i < size; ++i) {
                    load = std::min(capacity, load + runs[i].supply);
                    if ((middle >> i) & 1) {
                        quantity += load;
                        people_mid += runs[i].people;
                        load = 0;
                    }
                }
                split(runs, size, left, middle, q_left, quantity,
                      p_left, people_mid, levels, count);
                split(runs, size, middle, right, quantity, q_right,
                      people_mid, p_right, levels, count);
                return;
            }
        }
        levels[count++] = {rate, persons};
    }

    void profile(const Route& route, Level* levels, int& count) const {
        std::array<Run, max_nodes / 2> runs;
        int size = 0;
        double pending = 0, quantity = 0, persons = 0;
        for (int node : route) {
            if (supply[node]) {
                pending += supply[node];
                continue;
            }
            if (pending) {
                const double loaded = std::min(capacity, pending);
                runs[size++] = {loaded, 0};
                quantity += loaded;
                pending = 0;
            }
            if (size) {
                persons += people[node];
                runs[size - 1].people += people[node];
            }
        }
        if (!size) return;
        if (size == 1) {
            levels[count++] = {quantity / persons, persons};
            return;
        }
        if (quantity <= capacity) {
            // Capacity cannot bind between runs. Convex prefix envelope.
            std::array<double, max_nodes / 2 + 1> x{}, y{};
            int last = 0;
            double p = 0, q = 0;
            for (int i = 0; i < size; ++i) {
                p += runs[i].people;
                q += runs[i].supply;
                while (last && (y[last] - y[last - 1]) * (p - x[last])
                                   >= (q - y[last]) * (x[last] - x[last - 1])) --last;
                x[++last] = p;
                y[last] = q;
            }
            for (int i = 1; i <= last; ++i)
                levels[count++] = {(y[i] - y[i - 1]) / (x[i] - x[i - 1]), x[i] - x[i - 1]};
            return;
        }
        split(runs.data(), size, 0, (std::uint64_t{1} << size) - 1,
              0, quantity, 0, persons, levels, count);
    }

    double objective(Level* levels, int count) const {
        std::sort(levels, levels + count, [](Level a, Level b) { return a.rate < b.rate; });
        double remaining = 0, result = 0;
        for (int i = 0; i < count; ++i) remaining += levels[i].people;
        for (int i = 0; i < count; ++i) {
            const auto [rate, persons] = levels[i];
            remaining -= persons;
            result += rate * persons * (persons + 2 * remaining);
        }
        return population > 0 ? result / population : 0;
    }

    Rank evaluate(const Fleet& fleet) const {
        std::array<Level, max_nodes> levels;
        Rank rank{};
        int count = 0;
        for (const auto& route : fleet) {
            if (route.empty()) continue;
            if (!supply[route.front()] || !people[route.back()]) ++rank.broken;
            const double elapsed = time_of(route);
            rank.elapsed += elapsed;
            rank.late += std::max(0.0, elapsed - limit);
            profile(route, levels.data(), count);
        }
        rank.objective = objective(levels.data(), count);
        return rank;
    }

    void publish(const Fleet& fleet, double score, const char* source) {
        if (score <= best_score + eps || done(deadline)) return;
        best = fleet;
        best_score = score;
        std::ostringstream stops;
        for (const auto& route : fleet) {
            stops << ' ' << route.size();
            for (int site : route) stops << ' ' << site;
        }
        callback(source, stops.str().c_str());
    }

    double insert_missing(Fleet& fleet, int slot, Visited used, double value, Clock::time_point end) {
        auto& route = fleet[slot];
        for (int node : route) used[node] = true;
        while (!done(end)) {
            const int length = route.size();
            const double elapsed = time_of(route);
            double next_value = value;
            int selected = 0, position = 0;
            for (int node : active) {
                if (used[node]) continue;
                if (done(end)) break;
                for (int j = 0; j <= length; ++j) {
                    if ((!j && !supply[node]) || (j == length && !people[node])) continue;
                    const int prev = j ? route[j - 1] : 0;
                    const int next = j < length ? route[j] : 0;
                    if (elapsed + edge(prev, node) + edge(node, next) - edge(prev, next) > limit + eps)
                        continue;
                    route.insert(route.begin() + j, node);
                    const double score = evaluate(fleet).objective;
                    route.erase(route.begin() + j);
                    if (score > next_value + eps) {
                        selected = node;
                        position = j;
                        next_value = score;
                    }
                }
            }
            if (!selected) break;
            route.insert(route.begin() + position, selected);
            used[selected] = true;
            value = next_value;
        }
        return value;
    }

    void polish(Fleet& fleet, Clock::time_point end) {
        double value = evaluate(fleet).objective;
        for (int slot = 0; slot < vehicles && !done(end); ++slot) {
            auto& route = fleet[slot];
            if (route.empty()) continue;
            Visited blocked{};
            for (int v = 0; v < vehicles; ++v)
                if (v != slot) for (int node : fleet[v]) blocked[node] = true;
            shorten(route, end, true);
            value = insert_missing(fleet, slot, blocked, value, end);
            Route previous = route;
            const double previous_time = time_of(previous);
            if (shorten(route, end, false)) {
                double candidate = insert_missing(fleet, slot, blocked, evaluate(fleet).objective, end);
                if (candidate > value + eps ||
                    (std::abs(candidate - value) <= eps && time_of(route) < previous_time - eps)) {
                    value = candidate;
                } else {
                    route = std::move(previous);
                }
            }
        }
        const auto rank = evaluate(fleet);
        if (!rank.broken && rank.late <= eps) publish(fleet, rank.objective, "polish");
    }

    void rebuild(Fleet& fleet, int slot, Clock::time_point end, bool diversify) {
        Visited blocked{};
        for (int v = 0; v < vehicles; ++v) {
            if (v != slot) for (int node : fleet[v]) blocked[node] = true;
        }
        Route candidate = fleet[slot];
        double best_value = evaluate(fleet).objective;
        struct Seed { double value; int source, agency; };
        std::vector<Seed> seeds;
        for (int source : suppliers) {
            if (blocked[source] || done(end)) continue;
            for (int agency : agencies) {
                if (blocked[agency]) continue;
                fleet[slot] = {source, agency};
                if (time_of(fleet[slot]) <= limit + eps)
                    seeds.push_back({evaluate(fleet).objective, source, agency});
            }
        }
        std::sort(seeds.begin(), seeds.end(), [](const Seed& a, const Seed& b) {
            return a.value > b.value;
        });
        if (diversify && seeds.size() > 1) {
            const int offset = pick(std::min<int>(24, seeds.size()));
            std::swap(seeds[0], seeds[offset]);
        }
        const int starts = std::min<int>(diversify ? 3 : 6, seeds.size());
        for (int i = 0; i < starts && !done(end); ++i) {
            const auto seed = seeds[i];
            auto& route = fleet[slot];
            route = {seed.source, seed.agency};
            double value = insert_missing(fleet, slot, blocked, seed.value, end);
            while (shorten(route, end, true))
                value = insert_missing(fleet, slot, blocked, value, end);
            if (value > best_value + eps) {
                best_value = value;
                candidate = route;
            }
            publish(fleet, value, "rebuild");
        }
        fleet[slot] = std::move(candidate);
    }

    void propose(Fleet& fleet) {
        const int src = pick(vehicles);
        const int dst = vehicles > 1 ? (src + 1 + pick(vehicles - 1)) % vehicles : src;
        const int move = pick(15);
        Route& a = fleet[src];
        Route& b = fleet[dst];
        if (move <= 3) {
            if (active.empty()) return;
            const int node = active[pick(active.size())];
            for (const auto& route : fleet)
                if (std::find(route.begin(), route.end(), node) != route.end()) return;
            if (move <= 1 || a.empty()) a.insert(a.begin() + pick(a.size() + 1), node);
            else a[pick(a.size())] = node;
            return;
        }
        if (a.empty()) return;
        const int start = pick(a.size());
        if (move <= 8) {
            const int count = move <= 5 ? 1 : std::min<int>(a.size() - start, 1 + pick(3));
            const Route segment(a.begin() + start, a.begin() + start + count);
            a.erase(a.begin() + start, a.begin() + start + count);
            if (move == 4) return;
            Route& target = move % 2 == 0 ? a : b;
            target.insert(target.begin() + pick(target.size() + 1), segment.begin(), segment.end());
        } else if (move == 9) {
            int end = pick(a.size());
            std::reverse(a.begin() + std::min(start, end), a.begin() + 1 + std::max(start, end));
        } else if (move == 10 || dst == src) {
            std::swap(a[start], a[pick(a.size())]);
        } else if (move == 11 && !b.empty()) {
            std::swap(a[start], b[pick(b.size())]);
        } else {
            const int offset = pick(b.size() + 1);
            const int left = move == 12 ? a.size() - start : std::min<int>(a.size() - start, 1 + pick(3));
            const int right = move == 12 ? b.size() - offset : std::min<int>(b.size() - offset, 1 + pick(3));
            const Route segment(a.begin() + start, a.begin() + start + left);
            a.erase(a.begin() + start, a.begin() + start + left);
            a.insert(a.begin() + start, b.begin() + offset, b.begin() + offset + right);
            b.erase(b.begin() + offset, b.begin() + offset + right);
            b.insert(b.begin() + offset, segment.begin(), segment.end());
        }
    }

    void repartition(Fleet& fleet, Rank& score, std::uint64_t step) {
        // A second trajectory searches over the sets of stops. Re-sequencing
        // after a transfer opens moves that are late in their inherited order.
        Fleet trial = fleet;
        propose(trial);
        for (int v = 0; v < vehicles; ++v) {
            Route& route = trial[v];
            if (route == fleet[v] || route.empty()) continue;
            if (!supply[route.front()]) {
                const auto supplier = std::find_if(route.begin(), route.end(), [&](int node) { return supply[node]; });
                if (supplier != route.end()) std::rotate(route.begin(), supplier, supplier + 1);
            }
            if (!people[route.back()]) {
                const auto agency = std::find_if(route.rbegin(), route.rend(), [&](int node) { return people[node]; });
                if (agency != route.rend()) std::rotate(agency.base() - 1, agency.base(), route.end());
            }
            if (supply[route.front()] && people[route.back()]) {
                const bool preserve = time_of(route) <= limit || step % 3 != 0;
                shorten(route, deadline, preserve);
                if (preserve && time_of(route) > limit) shorten(route, deadline, false);
            }
        }
        const auto result = evaluate(trial);
        if (!result.broken && result.late <= eps) publish(trial, result.objective, "repartition");
        const double reference = std::max({1.0, best_score, 0.12 * supply_scale});
        const double delta = (result.objective - score.objective) / reference
                           - 4 * (result.late - score.late) / std::max(limit, 1.0)
                           - 0.18 * (result.broken - score.broken);
        const double remaining = 1 - static_cast<double>(step % 4096) / 4096;
        const double heat = 0.00005 + 0.006 * remaining * remaining;
        if (delta >= 0 || std::generate_canonical<double, 32>(rng) < std::exp(delta / heat)) {
            fleet = std::move(trial);
            score = result;
        }
    }

    void anneal() {
        const auto cycle = static_cast<std::uint64_t>(n <= 20 ? 90'000 : 180'000);
        Fleet current = best;
        Rank score = evaluate(current);
        Fleet partition = best;
        Rank partition_rank = score;
        std::uniform_real_distribution<double> uniform(0, 1);
        for (std::uint64_t step = 0; ; ++step) {
            if (step % 128 == 0 && done(deadline)) return;
            if (step % (128 * 4096) == 0) {
                partition = best;
                polish(partition, deadline);
                partition_rank = evaluate(partition);
            }
            if (step % 128 == 0) repartition(partition, partition_rank, step / 128);
            const auto round = step / cycle;
            if (step % cycle == 0) {
                current = best;
                if (round % 5 == 4) {
                    const int slot = pick(vehicles);
                    current[slot].clear();
                    rebuild(current, slot, std::min(deadline, Clock::now() + std::chrono::milliseconds(45)), true);
                }
                polish(current, deadline);
                score = evaluate(current);
            }
            auto trial = current;
            propose(trial);
            auto result = evaluate(trial);
            if (!result.broken && result.late > eps && result.objective > best_score + eps && step % 32 == 0) {
                // Keep the promising allocation and first try to repair only
                // the time violation on the changed routes.
                for (auto& route : trial)
                    if (time_of(route) > limit + eps) shorten(route, deadline, true);
                result = evaluate(trial);
            }
            if (!result.broken && result.late <= eps)
                publish(trial, result.objective, "joint");
            const double progress = static_cast<double>(step % cycle) / cycle;
            const double reference = std::max({1.0, best_score, 0.12 * supply_scale});
            const double penalty = 2.0 + (round % 4) + 3.0 * progress;
            const double delta = (result.objective - score.objective) / reference
                - (penalty * (result.late - score.late)
                   + 0.004 * (result.elapsed - score.elapsed)) / std::max(limit, 1.0)
                - 0.18 * (result.broken - score.broken);
            const double heat = 0.00012 + (0.021 + 0.011 * (round % 4)) * (1 - progress) * (1 - progress);
            if (delta >= 0 || uniform(rng) < std::exp(delta / heat)) {
                current = std::move(trial);
                score = result;
            }
        }
    }

    // Small-map route pool. Keep distinct allocation profiles per mask and
    // combine disjoint columns against the full fleet score.
    void route_pool(Clock::time_point end) {
        if (vehicles > 3 || n > 20 || done(end)) return;
        std::vector<double> tail(n + 1, INFINITY);
        for (int node : agencies) tail[node] = home[node];
        for (int pass = 0; pass < n; ++pass) {
            for (int i : active) for (int j : active) {
                if (i != j) tail[i] = std::min(tail[i], arc[i][j] + tail[j]);
            }
        }
        std::vector<std::vector<std::size_t>> profiles(std::size_t{1} << n);
        std::vector<Column> columns;
        Route route;
        std::uint64_t work = 0;
        const auto enumerate_end = Clock::now() + (end - Clock::now()) / 2;
        const auto visit = [&](auto&& self, int last, std::uint32_t mask, double elapsed) -> void {
            if ((++work & 255) == 0 && (done(enumerate_end) || work >= 2'000'000))
                work = 2'000'000;
            if (work >= 2'000'000) return;
            if (people[last] && elapsed + home[last] <= limit + eps) {
                std::array<Level, max_nodes> levels;
                int count = 0;
                profile(route, levels.data(), count);
                double score = objective(levels.data(), count);
                auto& indices = profiles[mask];
                const auto equal = [&](std::size_t i) {
                    const auto& old = columns[i].profile;
                    return old.size() == static_cast<std::size_t>(count)
                        && std::equal(old.begin(), old.end(), levels.begin(), [](Level a, Level b) {
                            return a.rate == b.rate && a.people == b.people;
                        });
                };
                if (score > eps && std::none_of(indices.begin(), indices.end(), equal)) {
                    const auto weakest = std::min_element(indices.begin(), indices.end(), [&](auto a, auto b) {
                        return columns[a].score < columns[b].score;
                    });
                    if (indices.size() < 12 || columns[*weakest].score < score) {
                        Column column{mask, route, {levels.begin(), levels.begin() + count},
                                      score, std::sqrt(score)};
                        if (indices.size() < 12) {
                            indices.push_back(columns.size());
                            columns.push_back(std::move(column));
                        } else {
                            columns[*weakest] = std::move(column);
                        }
                    }
                }
            }
            for (int node : active) {
                if (work >= 2'000'000) break;
                const auto bit = std::uint32_t{1} << (node - 1);
                const auto next = elapsed + arc[last][node];
                if ((mask & bit) || next + tail[node] > limit + eps) continue;
                route.push_back(node);
                self(self, node, mask | bit, next);
                route.pop_back();
            }
        };
        for (int source : suppliers) {
            if (work >= 2'000'000) break;
            if (arc[0][source] + tail[source] > limit + eps) continue;
            route = {source};
            visit(visit, source, std::uint32_t{1} << (source - 1), arc[0][source]);
        }
        if (columns.empty() || done(end)) return;
        std::sort(columns.begin(), columns.end(), [](const Column& a, const Column& b) {
            return a.score > b.score;
        });
        std::uint32_t supply_mask = 0;
        for (int source : suppliers) supply_mask |= std::uint32_t{1} << (source - 1);
        std::vector<unsigned> per_suppliers(std::size_t{1} << n);
        std::vector<const Column*> selected;
        for (const auto& column : columns) {
            if (per_suppliers[column.mask & supply_mask]++ < 640) selected.push_back(&column);
            if (selected.size() == 7'500) break;
        }
        const auto bound_can_improve = [&](double root) {
            return root * root > best_score + eps;
        };
        Fleet fleet(vehicles);
        std::array<Level, max_nodes> levels, combined;
        std::uint64_t pairs = 0;
        for (std::size_t i = 0; i < selected.size() && !done(end); ++i) {
            const auto& a = *selected[i];
            if (!bound_can_improve(vehicles * a.root)) break;
            fleet[0] = a.route;
            publish(fleet, a.score, "pool");
            if (vehicles == 1) break;
            for (std::size_t j = i + 1; j < selected.size(); ++j) {
                if ((++pairs & 127) == 0 && done(end)) return;
                const auto& b = *selected[j];
                if (!bound_can_improve(a.root + (vehicles - 1) * b.root)) break;
                if (a.mask & b.mask) continue;
                const auto finish = std::copy(b.profile.begin(), b.profile.end(),
                                              std::copy(a.profile.begin(), a.profile.end(), levels.begin()));
                const auto pair = objective(levels.data(), finish - levels.begin());
                fleet[1] = b.route;
                publish(fleet, pair, "pool");
                if (vehicles == 3) {
                    const auto pair_root = std::sqrt(pair);
                    for (std::size_t k = j + 1; k < selected.size(); ++k) {
                        if ((++pairs & 127) == 0 && done(end)) return;
                        const auto& c = *selected[k];
                        if (!bound_can_improve(pair_root + c.root)) break;
                        if ((a.mask | b.mask) & c.mask) continue;
                        const auto last = std::copy(c.profile.begin(), c.profile.end(),
                                                    std::copy(levels.begin(), finish, combined.begin()));
                        fleet[2] = c.route;
                        publish(fleet, objective(combined.data(), last - combined.begin()), "pool");
                    }
                    fleet[2].clear();
                }
            }
            fleet[1].clear();
        }
    }
};
}  // namespace

extern "C" int rebuild_fleet(const char* payload, Publish callback) {
    std::istringstream input(payload);
    Search search{};
    search.callback = callback;
    unsigned seed;
    double seconds;
    if (!(input >> search.n >> search.vehicles >> seed >> seconds >> search.capacity >> search.limit)
        || search.n < 1 || search.n > max_nodes || search.vehicles < 1) return 2;
    const int size = search.n + 1;
    search.supply.resize(size);
    search.people.resize(size);
    search.home.resize(size);
    search.arc.assign(size, std::vector<double>(size));
    for (auto* row : {&search.supply, &search.people, &search.home})
        for (double& value : *row) input >> value;
    for (auto& row : search.arc) for (double& value : row) input >> value;
    if (!input) return 2;
    for (int node = 1; node < size; ++node) {
        if (search.supply[node]) search.suppliers.push_back(node);
        if (search.people[node]) search.agencies.push_back(node);
        if (search.supply[node] || search.people[node]) search.active.push_back(node);
        search.population += search.people[node];
        search.supply_scale += std::min(search.supply[node], search.capacity);
    }
    search.rng.seed(seed);
    const auto start = Clock::now();
    const auto budget = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
    search.deadline = start + budget;
    search.best.resize(search.vehicles);
    Fleet rebuilt = search.best;
    const auto rebuild_end = start + std::min(budget / 6,
                            std::chrono::duration_cast<Clock::duration>(std::chrono::seconds(2)));
    for (int v = 0; v < search.vehicles && !search.done(rebuild_end); ++v)
        search.rebuild(rebuilt, v, rebuild_end, false);
    search.route_pool(start + budget * 2 / 5);
    search.anneal();
    return 0;
}
