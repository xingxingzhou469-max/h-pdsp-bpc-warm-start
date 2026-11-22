#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using Route = std::vector<int>;
using Time = std::int64_t;
using Publish = int (*)(const int *, int, double);
constexpr int MAX_NODES = 128;

struct Search {
    int n;
    Time limit;
    const Time *arc;
    const double *supply, *pop;
    double capacity, total_pop;
    std::mt19937_64 rng;
    Clock::time_point deadline;
    Time dist[MAX_NODES][MAX_NODES];
    std::int64_t *counters;

    Time edge(int i, int j) const { return arc[i * n + j]; }
    bool pickup(int i) const { return supply[i] > 0; }
    bool expired() const { return Clock::now() >= deadline; }
    int sample(int k) {
        return std::uniform_int_distribution<int>(0, k - 1)(rng);
    }

    Time time(const Route &route) const {
        Time t = 0;
        int previous = 0;
        for (int v : route) {
            t += edge(previous, v);
            previous = v;
        }
        return t + edge(previous, 0);
    }

    bool legal(const Route &route) const {
        return route.size() >= 2 && pickup(route.front()) && !pickup(route.back());
    }

    double score(const Route &route) const {
        std::array<double, MAX_NODES / 2> stock{}, people{};
        int groups = 0;
        double load = 0, sum_people = 0;
        for (int v : route) {
            if (pickup(v)) {
                load += supply[v];
            } else {
                if (load > 0) {
                    stock[groups++] = std::min(load, capacity);
                    load = 0;
                }
                people[groups - 1] += pop[v];
                sum_people += pop[v];
            }
        }

        double stock_visited = std::accumulate(stock.begin(), stock.begin() + groups, 0.0);
        if (stock_visited <= capacity) {
            // With carry capacity inactive, pool decreasing adjacent wealths.
            int runs = 0;
            for (int j = 0; j < groups; ++j) {
                stock[runs] = stock[j];
                people[runs] = people[j];
                ++runs;
                while (runs > 1 && stock[runs - 2] * people[runs - 1] >
                                    stock[runs - 1] * people[runs - 2]) {
                    --runs;
                    stock[runs - 1] += stock[runs];
                    people[runs - 1] += people[runs];
                }
            }
            double value = 0;
            for (int j = 0; j < runs; ++j) {
                sum_people -= people[j];
                value += stock[j] * (people[j] + 2 * sum_people);
            }
            return value / total_pop;
        }

        // A two-state cut over alternating pickup and delivery runs.
        using Mask = std::uint64_t;
        struct Block { Mask left, right; double rl, rr, pl, pr; };
        const Mask all = groups == 64 ? ~Mask(0) : (Mask(1) << groups) - 1;
        std::array<Block, MAX_NODES / 2> pending;
        int count = 1;
        pending[0] = {0, all, 0, stock_visited, 0, sum_people};
        double value = 0;
        while (count) {
            auto [left, right, rl, rr, pl, pr] = pending[--count];
            Mask allowed = right ^ left;
            const double rate = (rr - rl) / (pr - pl);
            if (allowed & (allowed - 1)) {
                double outside = 0, inside = 0;
                Mask out_mask = 0, in_mask = 0;
                for (int j = 0; j < groups; ++j) {
                    for (bool delivery : {false, true}) {
                        Mask bit = delivery ? Mask(1) << j : 0;
                        const bool leave = inside + capacity < outside;
                        double next_out = (leave ? inside + capacity : outside)
                                          + (delivery ? 0 : stock[j]);
                        Mask next_mask = (leave ? in_mask : out_mask) | (bit & allowed);
                        if (outside < inside) {
                            inside = outside;
                            in_mask = out_mask;
                        }
                        if (bit & left) inside = std::numeric_limits<double>::infinity();
                        if (bit & allowed) inside += rate * people[j];
                        outside = next_out;
                        out_mask = next_mask;
                    }
                }
                if (inside < outside) {
                    outside = inside;
                    out_mask = in_mask;
                }
                if (out_mask && out_mask != allowed && outside < rr - 1e-10) {
                    const Mask middle = left | out_mask;
                    double flow = 0, rank = 0, population = 0;
                    for (int j = 0; j < groups; ++j) {
                        flow = std::min(capacity, flow + stock[j]);
                        if (middle & (Mask(1) << j)) {
                            rank += flow;
                            population += people[j];
                            flow = 0;
                        }
                    }
                    pending[count++] = {middle, right, rank, rr, population, pr};
                    pending[count++] = {left, middle, rl, rank, pl, population};
                    continue;
                }
            }
            sum_people -= pr - pl;
            value += (rr - rl) * (pr - pl + 2 * sum_people);
        }
        return value / total_pop;
    }

    int weighted(const Route &nodes, bool inserting) {
        auto attractiveness = [&](int v) {
            const double weight = pickup(v) ? supply[v] / capacity : pop[v] / total_pop;
            return inserting ? weight : 1 / weight;
        };
        double sum = 0;
        for (int v : nodes) sum += attractiveness(v);
        double x = std::generate_canonical<double, 53>(rng) * sum;
        for (int i = 0; i + 1 < int(nodes.size()); ++i) {
            x -= attractiveness(nodes[i]);
            if (x <= 0) return i;
        }
        return int(nodes.size()) - 1;
    }

    std::vector<int> breaks(const Route &route) const {
        std::vector<int> points{0};
        for (int i = 1; i < int(route.size()); ++i) {
            if (pickup(route[i]) && !pickup(route[i - 1])) points.push_back(i);
        }
        points.push_back(int(route.size()));
        return points;
    }

    Route move(const Route &route, const Route &unused, const std::vector<int> &cuts) {
        Route result = route;
        const int k = int(route.size());
        const int i = sample(k), j = sample(k), l = std::min(i, j), r = std::max(i, j);
        switch (sample(9)) {
        case 0:
            if (!unused.empty()) {
                result.insert(result.begin() + sample(k + 1), unused[weighted(unused, true)]);
            }
            break;
        case 1:
            if (k > 2) result.erase(result.begin() + weighted(route, false));
            break;
        case 2:
            result.erase(result.begin() + i);
            result.insert(result.begin() + j, route[i]);
            break;
        case 3:
            std::swap(result[i], result[j]);
            break;
        case 4:
            if (!unused.empty()) result[weighted(route, false)] = unused[weighted(unused, true)];
            break;
        case 5:
            std::reverse(result.begin() + l, result.begin() + r + 1);
            break;
        case 6: {
            const auto begin = std::upper_bound(cuts.begin(), cuts.end(), i);
            const int a = *std::prev(begin), b = *begin;
            if (j < a) {
                std::rotate(result.begin() + j, result.begin() + a, result.begin() + b);
            } else if (j >= b) {
                std::rotate(result.begin() + a, result.begin() + b, result.begin() + j + 1);
            }
            break;
        }
        case 7:
            if (k > 2) result.erase(result.begin() + i);
            if (!unused.empty()) {
                result.insert(result.begin() + sample(int(result.size()) + 1),
                              unused[weighted(unused, true)]);
            }
            break;
        case 8:
            if (r > l) {
                std::rotate(result.begin() + l, result.begin() + l + 1, result.begin() + r + 1);
            }
            break;
        }
        return result;
    }

    void polish(Route &route) const {
        // Same-role changes leave the feasible aggregate allocation unchanged.
        Time best = time(route);
        const int k = int(route.size());
        for (int i = 0; i < k; ++i) {
            for (int j = i + 1; j < k && pickup(route[i]) == pickup(route[j]); ++j) {
                std::swap(route[i], route[j]);
                Time candidate = time(route);
                if (candidate < best) best = candidate;
                else std::swap(route[i], route[j]);
            }
        }
    }

    Route bridge(const Route &route, double current_score) {
        constexpr int WIDTH = 12;
        constexpr int MAX_PATH = 9;
        struct Label {
            std::array<int, MAX_PATH> path;
            int end;
            Time time;
            double rank;
        };
        const int k = int(route.size());
        const int removed = std::min(k, 2 + sample(6));
        const int left = sample(k - removed + 1), right = left + removed;
        const int a = left ? route[left - 1] : 0, b = right < k ? route[right] : 0;
        std::array<bool, MAX_NODES> fixed{};
        Time outside = 0;
        for (int i = 0, prev = 0; i < left; ++i) {
            fixed[route[i]] = true;
            outside += edge(prev, route[i]);
            prev = route[i];
        }
        for (int i = right; i < k; ++i) {
            fixed[route[i]] = true;
            outside += edge(route[i], i + 1 < k ? route[i + 1] : 0);
        }
        const Time available = limit - outside;
        if (available < 0) return {};
        Route work;
        work.reserve(n);
        work.insert(work.end(), route.begin(), route.begin() + left);
        const auto suffix = route.begin() + right;
        std::vector<Label> beam{{{}, a, 0, 0}}, next;
        next.reserve(WIDTH * n);

        Route best;
        double best_rank = -std::numeric_limits<double>::infinity();
        // Vary the tradeoff to allow an improved successor to be reachable
        // through a short, slightly lower-value bridge.
        const double slack = std::max(1.0, double(available));
        const double price = current_score * (0.005 + 0.08 *
                             std::generate_canonical<double, 53>(rng)) / slack;
        for (int length = 0; length < std::min(MAX_PATH, removed + 2); ++length) {
            if (expired()) break;
            for (const Label &label : beam) {
                for (int v = 1; v < n; ++v) {
                    if (fixed[v] || (!left && !length && !pickup(v))) continue;
                    if (std::find(label.path.begin(), label.path.begin() + length, v)
                            != label.path.begin() + length) continue;
                    const Time spent = label.time + edge(label.end, v);
                    const Time relaxed = spent + dist[v][b];
                    if (relaxed > available) continue;
                    work.resize(left);
                    work.insert(work.end(), label.path.begin(), label.path.begin() + length);
                    work.push_back(v);
                    work.insert(work.end(), suffix, route.end());
                    // A final pickup has no present delivery. Score the preceding
                    // path as a proxy, while still allowing this label to extend.
                    double value = 0;
                    if (legal(work)) {
                        value = score(work);
                        const Time direct = spent + edge(v, b);
                        const double rank = value - price * double(direct);
                        if (direct <= available && work != route && rank > best_rank) {
                            best_rank = rank;
                            best = work;
                        }
                    } else {
                        // Preserve a high-stock partial path that can make a drop.
                        auto last_drop = std::find_if(work.rbegin(), work.rend(),
                                                      [&](int u) { return !pickup(u); });
                        if (last_drop != work.rend()) {
                            Route completed(work.begin(), last_drop.base());
                            value = score(completed);
                            for (auto p = last_drop.base(); p != work.end(); ++p) {
                                value += 0.25 * supply[*p];
                            }
                        } else {
                            for (int p : work) value += 0.25 * supply[p];
                        }
                    }
                    Label child = label;
                    child.path[length] = v;
                    child.end = v;
                    child.time = spent;
                    child.rank = value - price * double(relaxed);
                    next.push_back(child);
                }
            }
            if (next.empty()) break;
            auto better = [](const Label &x, const Label &y) {
                if (x.rank != y.rank) return x.rank > y.rank;
                if (x.time != y.time) return x.time < y.time;
                return x.path < y.path;
            };
            if (next.size() > WIDTH) {
                std::nth_element(next.begin(), next.begin() + WIDTH, next.end(), better);
                next.resize(WIDTH);
            }
            beam.swap(next);
            next.clear();
        }
        return best;
    }

    void run(Route incumbent, bool rebuild, Publish publish) {
        Route current = incumbent;
        double best = score(incumbent), value = best;
        Time best_time = time(incumbent);
        std::vector<Route> elite{incumbent};
        int no_improvement = 0, bad = 0;
        while (!expired()) {
            ++counters[0];
            Route unused;
            std::array<bool, MAX_NODES> visited{};
            for (int v : current) visited[v] = true;
            for (int v = 1; v < n; ++v) {
                if (!visited[v]) unused.push_back(v);
            }
            auto points = breaks(current);
            Route chosen;
            double chosen_value = -1;
            Time chosen_time = limit;
            for (int sample = 0; sample < 10; ++sample) {
                Route next = move(current, unused, points);
                if (next == current || !legal(next)) continue;
                auto spent = time(next);
                if (spent > limit) continue;
                ++counters[1];
                double z = score(next);
                if (z > chosen_value || (z == chosen_value && spent < chosen_time)) {
                    chosen = std::move(next);
                    chosen_value = z;
                    chosen_time = spent;
                }
            }
            if (rebuild && counters[0] % 13 == 0) {
                ++counters[2];
                Route next = bridge(current, value);
                if (!next.empty()) {
                    ++counters[3];
                    const double z = score(next);
                    const Time t = time(next);
                    if (z > chosen_value || (z == chosen_value && t < chosen_time)) {
                        chosen = std::move(next);
                        chosen_value = z;
                        chosen_time = t;
                    }
                }
            }
            if (expired()) break;
            ++no_improvement;
            if (chosen.empty() || chosen_value < 0.8 * best) ++bad;
            else bad = 0;
            if (!chosen.empty()) {
                // Prefer quality, while permitting smaller departures from a
                // local maximum and periodically starting another trajectory.
                const double heat = std::max(1e-10, best * (0.019 - 0.016 *
                                                    (no_improvement / 1000.0)));
                const double draw = std::generate_canonical<double, 53>(rng);
                if (chosen_value >= value || draw < std::exp((chosen_value - value) / heat)) {
                    current = chosen;
                    value = chosen_value;
                }
                if (chosen_value > best + 1e-10 ||
                        (std::abs(chosen_value - best) <= 1e-10 && chosen_time < best_time)) {
                    polish(chosen);
                    if (chosen_value > best + 1e-10) {
                        if (expired()) break;
                        if (!publish(chosen.data(), int(chosen.size()), chosen_value)) return;
                        no_improvement = 0;
                        best = chosen_value;
                        elite.push_back(chosen);
                        if (elite.size() > 12) elite.erase(elite.begin() + 1);
                    }
                    incumbent = chosen;
                    best_time = time(chosen);
                    current = incumbent;
                    value = best;
                }
            }
            if (no_improvement >= 1000 || bad >= 50) {
                ++counters[4];
                current = counters[4] % 3 == 0 ? elite[sample(int(elite.size()))] : incumbent;
                value = score(current);
                no_improvement = bad = 0;
            }
        }
    }
};
}

extern "C" void route_search(int n, Time limit, const Time *arc, const double *supply,
                             const double *pop, double capacity, const int *seed,
                             int seed_len, std::uint64_t key, int rebuild, double seconds,
                             Publish publish, std::int64_t *counters) {
    Search s{n, limit, arc, supply, pop, capacity, std::accumulate(pop, pop + n, 0.0),
             std::mt19937_64(key), Clock::now() +
                 std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds)),
             {}, counters};
    if (rebuild) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) s.dist[i][j] = i == j ? 0 : s.edge(i, j);
        }
        for (int k = 0; k < n; ++k) {
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    s.dist[i][j] = std::min(s.dist[i][j], s.dist[i][k] + s.dist[k][j]);
                }
            }
        }
    }
    s.run(Route(seed, seed + seed_len), rebuild, publish);
}

extern "C" double route_score(int n, const double *supply, const double *pop,
                              double capacity, const int *route, int size) {
    Search s{n, 0, nullptr, supply, pop, capacity, std::accumulate(pop, pop + n, 0.0),
             std::mt19937_64(), {}, {}, nullptr};
    return s.score(Route(route, route + size));
}
