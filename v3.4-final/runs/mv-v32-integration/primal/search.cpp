// Derived from the isolated v4 search used by the previous v3.1b scheduler.
// Origin and changes are recorded in README.md.
#include "primal_profile.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace v3 {

using Clock = std::chrono::steady_clock;
using Tick = Clock::time_point;
using Publish = int (*)(const char*, const int*, int, double);

struct Column {
    Route route;
    Levels levels;
    double elapsed = 0;
    double self = 0;
};

using Fleet = std::vector<Column>;

struct Move {
    double key = 0;
    double second = 0;
    int slot = -1;
    Route route;
};

struct RouteHash {
    std::size_t operator()(const Route& route) const {
        std::size_t key = 0;
        for (int site : route) key = key * 1021 + static_cast<unsigned>(site);
        return key;
    }
};

constexpr double eps = 1e-10;

// Instrumentation only; never affects candidate generation or scoring.
unsigned long long g_iterations = 0;
unsigned long long g_repair_us = 0, g_walk_us = 0, g_polish_us = 0;
unsigned long long g_master_us = 0, g_destroy_us = 0, g_deep_us = 0;

class Search {
    int n, k, stride;
    double capacity, limit, total_people, quantity_scale = 1;
    const double* arc;
    std::vector<double> supply, people;
    std::vector<int> suppliers, agencies, by_people;
    Profile profile;
    Tick deadline;
    std::mt19937 rng;
    Publish publish;
    Fleet best;
    double best_value = 0;
    bool callback_failure = false;
    std::vector<double> destroy_weight = std::vector<double>(7, 1.0);
    // Constructed below with lower initial weight for the two role-focused ops.
    std::vector<double> repair_weight = std::vector<double>(2, 1.0);

    struct Seed { int source, agency; double time, rate; };
    std::vector<Seed> seeds;
    std::vector<Route> sparse_arcs;

    struct PoolEntry { Route route; double priority, elapsed; };
    std::unordered_map<Route, PoolEntry, RouteHash> pool;
    std::unordered_set<Route, RouteHash> rejected;
    std::uint64_t iterations = 0;
    std::uint64_t deep_calls = 0;

    double edge(int from, int to) const { return arc[from * stride + to]; }
    bool pickup(int site) const { return supply[site] > 0; }
    bool done() const { return callback_failure || Clock::now() >= deadline; }
    int pick(int count) { return static_cast<int>(rng() % static_cast<unsigned>(count)); }
    double uniform() { return std::generate_canonical<double, 32>(rng); }
    double time_tolerance() const { return 1e-12 * limit; }
    bool fits(double time) const { return time <= limit + time_tolerance(); }
    int choose(const std::vector<double>& weights) {
        double key = uniform() * std::accumulate(weights.begin(), weights.end(), 0.0);
        for (std::size_t i = 0; i < weights.size(); ++i)
            if ((key -= weights[i]) <= 0) return static_cast<int>(i);
        return static_cast<int>(weights.size()) - 1;
    }

    double route_time(const Route& route) const {
        if (route.empty()) return 0;
        int prev = 0;
        double time = 0;
        for (int site : route) {
            time += edge(prev, site);
            prev = site;
        }
        return time + edge(prev, 0);
    }

    bool endpoints(const Route& route) const {
        return route.empty() || (pickup(route.front()) && people[route.back()] > 0);
    }

    void update(Column& col) {
        col.elapsed = route_time(col.route);
        profile(col.route, col.levels);
        col.self = lorentz(col.levels, total_people);
    }

    double measure(const Fleet& fleet) const {
        Levels all;
        for (const auto& col : fleet)
            all.insert(all.end(), col.levels.begin(), col.levels.end());
        std::sort(all.begin(), all.end(), rate_order);
        return lorentz(all, total_people);
    }

    Background background(const Fleet& fleet, int a, int b = -1) const {
        Background bg;
        for (int v = 0; v < k; ++v) {
            if (v == a || v == b) continue;
            const auto& levels = fleet[v].levels;
            bg.levels.insert(bg.levels.end(), levels.begin(), levels.end());
        }
        bg.finish();
        return bg;
    }

    double contribution(const Column& col, const Background& bg) const {
        return col.self + bg.cross(col.levels, total_people);
    }

    double increment(const Route& route, const Background& bg, Levels& work) {
        profile(route, work);
        return lorentz(work, total_people) + bg.cross(work, total_people);
    }

    std::vector<int> occupancy(const Fleet& fleet) const {
        std::vector<int> used(stride, -1);
        for (int v = 0; v < k; ++v)
            for (int node : fleet[v].route) used[node] = v;
        return used;
    }

    static std::vector<int> pack(const Fleet& fleet) {
        std::vector<int> result{0};
        for (const auto& col : fleet) {
            if (col.route.empty()) continue;
            ++result[0];
            result.push_back(static_cast<int>(col.route.size()));
            result.insert(result.end(), col.route.begin(), col.route.end());
        }
        return result;
    }

    bool record(const Fleet& fleet, double value, const char* source) {
        const double tolerance = 2e-13 * std::abs(best_value);
        if (done() || value <= best_value + tolerance) return false;
        for (const auto& col : fleet)
            if (!fits(col.elapsed) || !endpoints(col.route)) return false;
        auto proposal = pack(fleet);
        if (rejected.contains(proposal)) return false;
        const int status = publish(source, proposal.data(), static_cast<int>(proposal.size()), value * quantity_scale);
        callback_failure = status < 0;
        if (status <= 0) {
            if (rejected.size() >= 1024) rejected.clear();
            if (status == 0) rejected.insert(std::move(proposal));
            return false;
        }
        best_value = value;
        best = fleet;
        return true;
    }

    // Directed 2-opt and Or-opt are scored in constant time with a reversal
    // prefix. The preserving pass moves only within a contiguous role run.
    bool shorten(Route& route, bool preserve, int passes) {
        const int size = static_cast<int>(route.size());
        if (size < 3) return false;
        std::vector<double> reversal(size);
        std::vector<int> block(size);
        bool changed = false;
        for (int pass = 0; pass < passes && !done(); ++pass) {
            reversal[0] = 0;
            block[0] = 0;
            for (int i = 1; i < size; ++i) {
                reversal[i] = reversal[i - 1] + edge(route[i], route[i - 1]) - edge(route[i - 1], route[i]);
                block[i] = block[i - 1] + (pickup(route[i]) != pickup(route[i - 1]));
            }
            const double threshold = 1e-13 * std::max(limit, route_time(route));
            double delta = -threshold;
            int a = -1, b = -1, gap = -1;
            for (int i = 0; i < size - 1 && !done(); ++i) {
                const int previous = i ? route[i - 1] : 0;
                for (int j = i + 1; j < size; ++j) {
                    if (preserve && block[i] != block[j]) break;
                    if ((!i && !pickup(route[j])) || (j == size - 1 && pickup(route[i]))) continue;
                    const int next = j + 1 < size ? route[j + 1] : 0;
                    const double change = edge(previous, route[j]) + edge(route[i], next)
                                        - edge(previous, route[i]) - edge(route[j], next)
                                        + reversal[j] - reversal[i];
                    if (change < delta) {
                        delta = change;
                        a = i;
                        b = j + 1;
                    }
                }
            }
            for (int length = 1; length <= 3 && length < size; ++length) {
                for (int i = 0; i + length <= size && !done(); ++i) {
                    const int j = i + length;
                    const int left = i ? route[i - 1] : 0, right = j < size ? route[j] : 0;
                    const int head = route[i], tail = route[j - 1];
                    const double removal = edge(left, right) - edge(left, head) - edge(tail, right);
                    for (int pos = 0; pos <= size; ++pos) {
                        if (pos >= i && pos <= j) continue;
                        if (preserve && block[std::min(pos, i)] != block[std::max(pos, j) - 1]) continue;
                        const int first = pos == 0 ? head : (i == 0 ? right : route.front());
                        const int last = pos == size ? tail : (j == size ? left : route.back());
                        if (!pickup(first) || pickup(last)) continue;
                        const int prev = pos ? route[pos - 1] : 0, next = pos < size ? route[pos] : 0;
                        const double change = removal + edge(prev, head) + edge(tail, next) - edge(prev, next);
                        if (change < delta) {
                            delta = change;
                            a = i;
                            b = j;
                            gap = pos;
                        }
                    }
                }
            }
            if (a < 0) break;
            if (gap < 0) std::reverse(route.begin() + a, route.begin() + b);
            else if (gap < a) std::rotate(route.begin() + gap, route.begin() + a, route.begin() + b);
            else std::rotate(route.begin() + a, route.begin() + b, route.begin() + gap);
            changed = true;
        }
        return changed;
    }

    void setup_seeds() {
        for (int source : suppliers)
            for (int agency : agencies) {
                const double cost = edge(0, source) + edge(source, agency) + edge(agency, 0);
                if (fits(cost))
                    seeds.push_back({source, agency, cost, std::min(supply[source], capacity) / people[agency]});
            }
    }

    void setup_sparse_arcs() {
        if (n < 64) return;
        sparse_arcs.resize(stride);
        int count = 0;
        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j <= n; ++j) {
                if (i == j || !fits(edge(i, j))) continue;
                if (++count > 6 * n) {
                    sparse_arcs.clear();
                    return;
                }
                sparse_arcs[i].push_back(j);
            }
        }
    }

    // On sparse directed graphs a feasible route can require adding a pickup
    // and its downstream deliveries together. Grow a path forward and evaluate
    // its return at every agency, even if intermediate returns are infeasible.
    void fill_sparse_routes(Fleet& fleet, Tick stop) {
        if (sparse_arcs.empty() || Clock::now() >= stop) return;
        auto owner = occupancy(fleet);
        for (int v = 0; v < k && Clock::now() < stop && !done(); ++v) {
            if (!fleet[v].route.empty()) continue;
            const auto bg = background(fleet, v);
            Route longest;
            double merit = 0, best_time = 0;
            Levels scratch;
            auto used = owner;
            for (int start : suppliers) {
                if (Clock::now() >= stop || done()) break;
                if (owner[start] >= 0 || !fits(edge(0, start))) continue;
                Route trial{start};
                used[start] = v;
                int tail = start;
                double elapsed = edge(0, start);
                while (!done() && Clock::now() < stop) {
                    if (!pickup(tail)) {
                        const double return_time = elapsed + edge(tail, 0);
                        if (fits(return_time)) {
                            const double objective = increment(trial, bg, scratch);
                            if (better(objective, -return_time, merit, -best_time)) {
                                longest = trial;
                                merit = objective;
                                best_time = return_time;
                            }
                        }
                    }
                    int next = 0;
                    double best_edge = std::numeric_limits<double>::infinity();
                    for (int node : sparse_arcs[tail]) {
                        const double cost = edge(tail, node);
                        if (used[node] >= 0 || !fits(elapsed + cost)) continue;
                        if (cost < best_edge ||
                            (cost == best_edge && people[node] > people[next])) {
                            next = node;
                            best_edge = cost;
                        }
                    }
                    if (!next) break;
                    used[next] = v;
                    trial.push_back(next);
                    tail = next;
                    elapsed += best_edge;
                }
                for (int node : trial) used[node] = -1;
            }
            if (longest.empty()) return;
            auto& column = fleet[v];
            column.route = std::move(longest);
            for (int node : column.route) owner[node] = v;
            update(column);
            record(fleet, measure(fleet), "directed-chain");
        }
    }

    static std::vector<int> gap_classes(const Route& route, bool is_pickup,
                                        const std::vector<double>& supply) {
        std::vector<int> classes(route.size() + 1);
        for (std::size_t i = 1; i <= route.size(); ++i)
            classes[i] = classes[i - 1] + ((supply[route[i - 1]] > 0) != is_pickup);
        return classes;
    }

    static void offer(Move& move, double key, int slot, const Route& route) {
        if (key > move.key) {
            move.second = move.key;
            move.key = key;
            move.slot = slot;
            move.route = route;
        } else {
            move.second = std::max(move.second, key);
        }
    }

    void add_seed(int slot, const Background& bg, const std::vector<int>& used,
                  double ratio, const std::vector<double>& priority, std::vector<Move>& proposals) {
        Levels level(1);
        Route route(2);
        for (const auto& s : seeds) {
            if (used[s.source] >= 0 || used[s.agency] >= 0) continue;
            level[0] = {s.rate, people[s.agency]};
            const double score = (s.rate * people[s.agency] * people[s.agency] / total_people
                                  + bg.cross(level, total_people));
            const double cost = limit > 0 ? s.time / limit : 0;
            const double merit = score / (1 + ratio * cost) * priority[s.agency] * priority[s.source];
            route[0] = s.source;
            route[1] = s.agency;
            offer(proposals[s.agency], merit, slot, route);
        }
    }

    void insertions(const Column& col, int slot, const Background& bg,
                    const std::vector<int>& used, double ratio,
                    const std::vector<double>& priority, std::vector<Move>& proposals) {
        const auto& route = col.route;
        const int size = static_cast<int>(route.size());
        const auto pick_class = gap_classes(route, true, supply);
        const auto drop_class = gap_classes(route, false, supply);
        std::vector<double> deltas(size + 1), best_delta(size + 1);
        std::vector<int> before(size + 1), after(size + 1), position(size + 1);
        for (int i = 0; i <= size; ++i) {
            before[i] = i ? route[i - 1] : 0;
            after[i] = i < size ? route[i] : 0;
            deltas[i] = edge(before[i], after[i]);
        }
        Route trial;
        trial.reserve(size + 1);
        Levels scratch;
        const double old = contribution(col, bg);
        for (int node = 1; node <= n && !done(); ++node) {
            if (used[node] >= 0) continue;
            const bool source = pickup(node);
            const auto& classes = source ? pick_class : drop_class;
            const int first = source ? 0 : 1, last = source ? size - 1 : size;
            std::fill(position.begin(), position.end(), -1);
            for (int i = first; i <= last; ++i) {
                const double delta = edge(before[i], node) + edge(node, after[i]) - deltas[i];
                const int group = classes[i];
                if (fits(col.elapsed + delta) && (position[group] < 0 || delta < best_delta[group])) {
                    position[group] = i;
                    best_delta[group] = delta;
                }
            }
            for (int group = 0; group <= classes.back(); ++group) {
                const int pos = position[group];
                if (pos < 0) continue;
                trial.assign(route.begin(), route.end());
                trial.insert(trial.begin() + pos, node);
                const double gain = increment(trial, bg, scratch) - old;
                const double delta = best_delta[group];
                if (gain <= eps * old && delta >= -time_tolerance()) continue;
                const double cost = limit > 0 ? delta / limit : 0;
                const double merit = (std::max(0.0, gain) + old * std::max(0.0, -cost) * 0.01)
                                   * priority[node] / (1 + ratio * std::max(0.0, cost));
                offer(proposals[node], merit, slot, trial);
            }
        }
    }

    void repair(Fleet& fleet, bool regret, double ratio, double noise, int target = 0) {
        std::vector<double> priority(stride, 1.0);
        if (noise > 0)
            for (int i = 1; i <= n; ++i) priority[i] += noise * (2 * uniform() - 1);
        if (target) priority[target] *= 1.55;
        auto used = occupancy(fleet);
        while (!done()) {
            std::vector<Move> moves(stride);
            int idle = -1;
            for (int v = 0; v < k && !done(); ++v) {
                if (fleet[v].route.empty() && idle >= 0) continue;
                const auto bg = background(fleet, v);
                if (fleet[v].route.empty()) {
                    idle = v;
                    add_seed(v, bg, used, ratio, priority, moves);
                } else {
                    insertions(fleet[v], v, bg, used, ratio, priority, moves);
                }
            }
            double merit = 0;
            int selected = 0;
            for (int node = 1; node <= n; ++node) {
                const auto& move = moves[node];
                const double key = move.key + (regret ? 0.75 * (move.key - move.second) : 0);
                if (key > merit) {
                    selected = node;
                    merit = key;
                }
            }
            if (!selected) return;
            auto& move = moves[selected];
            auto& col = fleet[move.slot];
            col.route = std::move(move.route);
            for (int site : col.route) used[site] = move.slot;
            update(col);
        }
    }

    void restore(Column& col) {
        auto& route = col.route;
        if (route.empty()) {
            update(col);
            return;
        }
        const auto first = std::find_if(route.begin(), route.end(), [&](int node) { return pickup(node); });
        if (first == route.end()) route.clear();
        else if (first != route.begin()) std::rotate(route.begin(), first, first + 1);
        const auto last = std::find_if(route.rbegin(), route.rend(), [&](int node) { return !pickup(node); });
        if (last == route.rend()) route.clear();
        else if (last != route.rbegin()) std::rotate(last.base() - 1, last.base(), route.end());
        if (!fits(route_time(route))) shorten(route, false, 4);
        if (!fits(route_time(route))) route.clear();
        update(col);
    }

    int destroy(Fleet& fleet, int op, bool large) {
        const auto owner = occupancy(fleet);
        Route visited, active_routes;
        for (int site = 1; site <= n; ++site)
            if (owner[site] >= 0) visited.push_back(site);
        for (int v = 0; v < k; ++v)
            if (!fleet[v].route.empty()) active_routes.push_back(v);
        if (visited.empty()) return 0;
        const auto size = static_cast<int>(visited.size());
        const int remove_count = std::min(size, 2 + pick(std::max(1, std::min(16, size / (large ? 3 : 7)))));
        std::vector<unsigned char> removed(stride);
        int target = 0;

        if (op == 5 || op == 6) {
            // Role-focused removal: free time by dropping the weakest
            // agencies (op 5) so suppliers can enter, or the weakest
            // suppliers (op 6) so population coverage can rise. The repair
            // merit then decides whether the replacement is profitable.
            std::vector<std::pair<double, int>> ranked;
            for (int site : visited) {
                if (op == 5 && pickup(site)) continue;
                if (op == 6 && !pickup(site)) continue;
                const double value = op == 5 ? people[site] : std::min(capacity, supply[site]);
                ranked.emplace_back(value * (0.70 + 0.60 * uniform()), site);
            }
            std::sort(ranked.begin(), ranked.end());
            const int count = std::min<int>(remove_count, ranked.size());
            for (int i = 0; i < count; ++i) removed[ranked[i].second] = 1;
        } else if (op == 2) {
            const auto slot = active_routes[pick(static_cast<int>(active_routes.size()))];
            for (int site : fleet[slot].route) removed[site] = 1;
        } else if (op == 3) {
            const int a = active_routes[pick(static_cast<int>(active_routes.size()))];
            const int b = (a + 1 + pick(std::max(1, k - 1))) % k;
            for (int v : {a, b}) {
                const auto& route = fleet[v].route;
                Route sources;
                for (int site : route)
                    if (pickup(site)) sources.push_back(site);
                std::shuffle(sources.begin(), sources.end(), rng);
                const int count = std::min<int>(sources.size(), 1 + static_cast<int>(sources.size()) / 2);
                for (int i = 0; i < count; ++i) removed[sources[i]] = 1;
                if (route.size() > 3 && uniform() < 0.6)
                    removed[route[1 + pick(static_cast<int>(route.size()) - 2)]] = 1;
            }
        } else {
            int center = visited[pick(size)];
            if (op == 4) {
                Route missed;
                for (int node : by_people) {
                    if (owner[node] < 0) missed.push_back(node);
                    if (missed.size() == 9) break;
                }
                if (!missed.empty()) target = center = missed[pick(static_cast<int>(missed.size()))];
            }

            std::vector<std::pair<double, int>> ranking;
            if (op == 1) {
                Levels scratch;
                for (int v : active_routes) {
                    if (done()) break;
                    const auto& col = fleet[v];
                    const auto bg = background(fleet, v);
                    const double old = contribution(col, bg);
                    for (std::size_t i = 0; i < col.route.size(); ++i) {
                        Route reduced = col.route;
                        const int site = reduced[i];
                        reduced.erase(reduced.begin() + i);
                        const double value = endpoints(reduced) ? increment(reduced, bg, scratch) : 0;
                        const double saved = col.elapsed - route_time(reduced);
                        const double time = limit > 0 ? saved / limit : 0;
                        ranking.emplace_back((old - value) / (1 + 15 * std::max(0.0, time))
                                               * (0.55 + uniform()), site);
                    }
                }
            } else {
                const double scale = limit > 0 ? limit : 1;
                int nearby = owner[center];
                if (nearby < 0) {
                    const auto close = std::min_element(visited.begin(), visited.end(), [&](int a, int b) {
                        return edge(a, center) + edge(center, a) < edge(b, center) + edge(center, b);
                    });
                    nearby = owner[*close];
                }
                for (int site : visited) {
                    const double distance = (edge(center, site) + edge(site, center)) / scale;
                    const double related = distance + (owner[site] == nearby ? 0 : 0.035)
                                           + (pickup(site) == pickup(center) ? 0 : 0.025);
                    ranking.emplace_back(related * (0.53 + uniform()), site);
                }
            }
            std::sort(ranking.begin(), ranking.end());
            for (int i = 0; i < std::min<int>(remove_count, ranking.size()); ++i)
                removed[ranking[i].second] = 1;
        }
        for (auto& col : fleet) {
            const auto count = std::erase_if(col.route, [&](int node) { return removed[node]; });
            if (count) restore(col);
        }
        return target;
    }

    double substitution_time(const Column& col, int pos, int node) const {
        const auto& route = col.route;
        const int a = pos ? route[pos - 1] : 0;
        const int b = pos + 1 < static_cast<int>(route.size()) ? route[pos + 1] : 0;
        const int site = route[pos];
        return col.elapsed + edge(a, node) + edge(node, b) - edge(a, site) - edge(site, b);
    }

    static bool better(double gain, double saving, double old, double old_saving) {
        const double tolerance = 1e-12 * std::max(std::abs(gain), std::abs(old));
        return gain > old + tolerance || (std::abs(gain - old) <= tolerance && saving > old_saving);
    }

    bool replace_unused(Fleet& fleet) {
        const auto owner = occupancy(fleet);
        double best_gain = 0, best_saving = time_tolerance();
        int selected = -1;
        Route replacement;
        for (int v = 0; v < k && !done(); ++v) {
            const auto& col = fleet[v];
            if (col.route.empty()) continue;
            const auto bg = background(fleet, v);
            const double old = contribution(col, bg);
            Route trial = col.route;
            Levels scratch;
            for (int i = 0; i < static_cast<int>(trial.size()) && !done(); ++i) {
                const int site = col.route[i];
                const auto& options = pickup(site) ? suppliers : agencies;
                for (int node : options) {
                    if (owner[node] >= 0) continue;
                    const auto time = substitution_time(col, i, node);
                    if (!fits(time)) continue;
                    trial[i] = node;
                    const double gain = increment(trial, bg, scratch) - old;
                    const double saving = col.elapsed - time;
                    if (better(gain, saving, best_gain, best_saving)) {
                        replacement = trial;
                        selected = v;
                        best_gain = gain;
                        best_saving = saving;
                    }
                }
                trial[i] = site;
            }
        }
        if (selected < 0) return false;
        fleet[selected].route = std::move(replacement);
        update(fleet[selected]);
        return true;
    }

    double pair_value(const Levels& a, const Levels& b, const Background& bg, Levels& work) const {
        work.clear();
        std::merge(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(work), rate_order);
        return lorentz(work, total_people) + bg.cross(work, total_people);
    }

    // Explicit cross-vehicle 1-1 swap and 2-opt*. Prefix clocks include the
    // return arc so each candidate passes a constant-time directed filter.
    bool cross_improve(Fleet& fleet, int a, int b) {
        auto& A = fleet[a];
        auto& B = fleet[b];
        const int na = static_cast<int>(A.route.size()), nb = static_cast<int>(B.route.size());
        if (!na || !nb) return false;
        const auto bg = background(fleet, a, b);
        Levels al, bl, merged;
        const double old = pair_value(A.levels, B.levels, bg, merged);
        double best_gain = 0, best_saving = time_tolerance();
        Route x = A.route, y = B.route, ax, by;
        const auto evaluate = [&](double elapsed_a, double elapsed_b) {
            if (!endpoints(x) || !endpoints(y)) return;
            profile(x, al);
            profile(y, bl);
            const double gain = pair_value(al, bl, bg, merged) - old;
            const double saving = A.elapsed + B.elapsed - elapsed_a - elapsed_b;
            if (better(gain, saving, best_gain, best_saving)) {
                best_gain = gain;
                best_saving = saving;
                ax = x;
                by = y;
            }
        };
        for (int i = 0; i < na && !done(); ++i) {
            for (int j = 0; j < nb; ++j) {
                const double ta = substitution_time(A, i, B.route[j]);
                const double tb = substitution_time(B, j, A.route[i]);
                if (!fits(ta) || !fits(tb)) continue;
                std::swap(x[i], y[j]);
                evaluate(ta, tb);
                std::swap(x[i], y[j]);
            }
        }
        std::vector<double> ap(na + 1), bp(nb + 1);
        auto prefix = [&](const Route& route, std::vector<double>& prefix_clock) {
            int site = 0;
            for (std::size_t i = 0; i < route.size(); ++i) {
                prefix_clock[i + 1] = prefix_clock[i] + edge(site, route[i]);
                site = route[i];
            }
        };
        prefix(A.route, ap);
        prefix(B.route, bp);
        for (int i = 1; i <= na && !done(); ++i) {
            for (int j = 1; j <= nb; ++j) {
                if (i == na && j == nb) continue;
                const int left = A.route[i - 1], right = B.route[j - 1];
                const int xnext = j < nb ? B.route[j] : 0;
                const int ynext = i < na ? A.route[i] : 0;
                const double ta = ap[i] + edge(left, xnext)
                                + (j < nb ? B.elapsed - bp[j + 1] : 0);
                const double tb = bp[j] + edge(right, ynext)
                                + (i < na ? A.elapsed - ap[i + 1] : 0);
                if (!fits(ta) || !fits(tb)) continue;
                x.assign(A.route.begin(), A.route.begin() + i);
                x.insert(x.end(), B.route.begin() + j, B.route.end());
                y.assign(B.route.begin(), B.route.begin() + j);
                y.insert(y.end(), A.route.begin() + i, A.route.end());
                evaluate(ta, tb);
            }
        }
        if (ax.empty()) return false;
        A.route = std::move(ax);
        B.route = std::move(by);
        update(A);
        update(B);
        return true;
    }

    // On a time-saturated fleet the ALNS repair cannot insert an unserved
    // station without a destroy. This pass scores exact fleet deltas for
    // one-for-one exchanges (drop a served stop, insert an unserved one) and
    // for pure insertions when slack exists, moving the selection toward
    // higher population or supply per unit of route time.
    bool exchange_unvisited(Fleet& fleet, int limit_candidates) {
        auto owner = occupancy(fleet);
        std::vector<int> unused;
        for (int node = 1; node <= n; ++node)
            if (owner[node] < 0) unused.push_back(node);
        if (unused.empty()) return false;
        const auto value = [&](int node) {
            return std::max(people[node], std::min(capacity, supply[node]));
        };
        std::sort(unused.begin(), unused.end(), [&](int a, int b) {
            return value(a) == value(b) ? a < b : value(a) > value(b);
        });
        if (static_cast<int>(unused.size()) > limit_candidates) unused.resize(limit_candidates);
        double best_gain = 0, best_saving = time_tolerance();
        int best_vehicle = -1;
        Route best_route;
        Levels scratch;
        for (int v = 0; v < k && !done(); ++v) {
            const auto& col = fleet[v];
            if (col.route.empty()) continue;
            const auto bg = background(fleet, v);
            const double old = contribution(col, bg);
            const auto& route = col.route;
            const int size = static_cast<int>(route.size());
            for (int node : unused) {
                if (done()) break;
                const bool source = pickup(node);
                const int first = source ? 0 : 1;
                const int last = source ? size - 1 : size;
                for (int pos = first; pos <= last; ++pos) {
                    Route trial = route;
                    trial.insert(trial.begin() + pos, node);
                    const double elapsed = route_time(trial);
                    if (!fits(elapsed)) continue;
                    const double gain = increment(trial, bg, scratch) - old;
                    const double saving = col.elapsed - elapsed;
                    if (better(gain, saving, best_gain, best_saving)) {
                        best_gain = gain;
                        best_saving = saving;
                        best_vehicle = v;
                        best_route = std::move(trial);
                    }
                }
                for (int i = 0; i < size && !done(); ++i) {
                    if (route[i] == node) continue;
                    Route reduced = route;
                    reduced.erase(reduced.begin() + i);
                    const int trimmed = size - 1;
                    const int head = source ? 0 : 1;
                    const int tail = source ? trimmed - 1 : trimmed;
                    if (tail < head) continue;
                    for (int pos = head; pos <= tail; ++pos) {
                        Route trial = reduced;
                        trial.insert(trial.begin() + pos, node);
                        if (!endpoints(trial)) continue;
                        const double elapsed = route_time(trial);
                        if (!fits(elapsed)) continue;
                        const double gain = increment(trial, bg, scratch) - old;
                        const double saving = col.elapsed - elapsed;
                        if (better(gain, saving, best_gain, best_saving)) {
                            best_gain = gain;
                            best_saving = saving;
                            best_vehicle = v;
                            best_route = std::move(trial);
                        }
                    }
                }
            }
        }
        if (best_vehicle < 0) return false;
        fleet[best_vehicle].route = std::move(best_route);
        update(fleet[best_vehicle]);
        return true;
    }

    // Move a single stop to another vehicle. Route-local allocation levels
    // show whether supply and population are balanced across the fleet; the
    // ALNS walk only proposes such relocations at random. This pass scores
    // every feasible directed relocation exactly and takes the best gain.
    bool relocate(Fleet& fleet) {
        if (k < 2) return false;
        double best_gain = 0, best_saving = time_tolerance();
        int best_a = -1, best_b = -1;
        Route best_a_route, best_b_route;
        Levels la, lb, merged;
        for (int a = 0; a < k && !done(); ++a) {
            if (fleet[a].route.size() < 2) continue;
            for (int b = 0; b < k; ++b) {
                if (a == b || fleet[b].route.empty()) continue;
                const auto bg = background(fleet, a, b);
                const double old = pair_value(fleet[a].levels, fleet[b].levels, bg, merged);
                const auto& ra = fleet[a].route;
                const auto& rb = fleet[b].route;
                for (int i = 0; i < static_cast<int>(ra.size()) && !done(); ++i) {
                    const int node = ra[i];
                    const bool source = pickup(node);
                    Route reduced = ra;
                    reduced.erase(reduced.begin() + i);
                    if (!endpoints(reduced)) continue;
                    const double ta = route_time(reduced);
                    if (!fits(ta)) continue;
                    profile(reduced, la);
                    const int head = source ? 0 : 1;
                    const int tail = source ? static_cast<int>(rb.size()) - 1
                                            : static_cast<int>(rb.size());
                    for (int pos = head; pos <= tail; ++pos) {
                        Route grown = rb;
                        grown.insert(grown.begin() + pos, node);
                        if (!endpoints(grown)) continue;
                        const double tb = route_time(grown);
                        if (!fits(tb)) continue;
                        profile(grown, lb);
                        const double gain = pair_value(la, lb, bg, merged) - old;
                        const double saving = fleet[a].elapsed + fleet[b].elapsed - ta - tb;
                        if (better(gain, saving, best_gain, best_saving)) {
                            best_gain = gain;
                            best_saving = saving;
                            best_a = a;
                            best_b = b;
                            best_a_route = reduced;
                            best_b_route = grown;
                        }
                    }
                }
            }
        }
        if (best_a < 0) return false;
        fleet[best_a].route = std::move(best_a_route);
        fleet[best_b].route = std::move(best_b_route);
        update(fleet[best_a]);
        update(fleet[best_b]);
        return true;
    }

    // Relocate across role runs when the allocation gain is worth the
    // travel cost. A time-only pass would miss these rebalancing moves.
    bool order_improve(Fleet& fleet, int slot) {
        auto& col = fleet[slot];
        const auto& route = col.route;
        const int size = static_cast<int>(route.size());
        if (size < 3) return false;
        const auto bg = background(fleet, slot);
        const double old = contribution(col, bg);
        double best_gain = 0, best_saving = time_tolerance();
        Route best_route;
        Levels scratch;
        for (int i = 0; i < size && !done(); ++i) {
            const int site = route[i], left = i ? route[i - 1] : 0;
            const int right = i + 1 < size ? route[i + 1] : 0;
            const double removal = edge(left, right) - edge(left, site) - edge(site, right);
            for (int pos = 0; pos <= size; ++pos) {
                if (pos == i || pos == i + 1) continue;
                const int prev = pos ? route[pos - 1] : 0;
                const int next = pos < size ? route[pos] : 0;
                const double delta = removal + edge(prev, site) + edge(site, next) - edge(prev, next);
                if (!fits(col.elapsed + delta)) continue;
                Route candidate = route;
                if (pos < i)
                    std::rotate(candidate.begin() + pos, candidate.begin() + i, candidate.begin() + i + 1);
                else
                    std::rotate(candidate.begin() + i, candidate.begin() + i + 1, candidate.begin() + pos);
                if (!endpoints(candidate)) continue;
                const double gain = increment(candidate, bg, scratch) - old;
                if (better(gain, -delta, best_gain, best_saving)) {
                    best_gain = gain;
                    best_saving = -delta;
                    best_route = std::move(candidate);
                }
            }
        }
        if (best_route.empty()) return false;
        col.route = std::move(best_route);
        update(col);
        return true;
    }

    void polish(Fleet& fleet, bool intensify) {
        bool changed = false;
        for (auto& col : fleet) {
            if (shorten(col.route, true, intensify ? 8 : 3)) {
                // Reordering within a role run preserves the allocation.
                col.elapsed = route_time(col.route);
                changed = true;
            }
            if (intensify && !done() && col.route.size() >= 3) {
                // A role-agnostic pass can free time that the preserving pass
                // cannot (v2.1 polish used the same two-stage pattern). Keep
                // it only when the route's own fair score does not fall.
                const Route saved = col.route;
                const double before = col.self;
                if (shorten(col.route, false, 4)) {
                    update(col);
                    if (col.self + 1e-12 * std::abs(before) < before) {
                        col.route = saved;
                        update(col);
                    } else {
                        changed = true;
                    }
                }
            }
        }
        if (intensify && !done()) {
            const auto a = pick(k);
            changed |= order_improve(fleet, a);
            changed |= replace_unused(fleet);
            if (k > 1) {
                for (int offset = 1; offset < k && !done(); ++offset)
                    if (cross_improve(fleet, a, (a + offset) % k)) {
                        changed = true;
                        break;
                    }
            }
            // Deep passes are the expensive part: they examine every served
            // stop for an exchange or a cross-vehicle relocation. Run them on
            // alternate intensifications so the ALNS keeps its throughput.
            if (n >= 64 && (++deep_calls & 1U) == 0) {
                const auto clock = Clock::now();
                changed |= exchange_unvisited(fleet, 24);
                changed |= relocate(fleet);
                g_deep_us += std::chrono::duration_cast<std::chrono::microseconds>(
                    Clock::now() - clock).count();
            }
        }
        if (changed && !done()) repair(fleet, false, 2.5, 0);
    }

    void archive(const Fleet& fleet, double value) {
        for (const auto& col : fleet) {
            if (col.route.empty() || !fits(col.elapsed)) continue;
            auto key = col.route;
            // Two routes with the same stops can have different fairness
            // profiles. Only coalesce permutations within the same role run.
            auto first = key.begin();
            while (first != key.end()) {
                const bool role = pickup(*first);
                const auto last = std::find_if(first, key.end(), [&](int site) {
                    return pickup(site) != role;
                });
                std::sort(first, last);
                first = last;
            }
            const auto found = pool.find(key);
            const double priority = value + 0.25 * col.self;
            if (found != pool.end()) {
                if (priority > found->second.priority ||
                    (priority == found->second.priority && col.elapsed < found->second.elapsed))
                    found->second = {col.route, priority, col.elapsed};
                continue;
            }
            if (pool.size() >= 360) {
                // Keep useful alternatives from different station sets.
                const auto worst = std::min_element(pool.begin(), pool.end(), [](const auto& a, const auto& b) {
                    return a.second.priority < b.second.priority;
                });
                if (priority < worst->second.priority && uniform() >= 0.08) continue;
                pool.erase(worst);
            }
            pool.emplace(std::move(key), PoolEntry{col.route, priority, col.elapsed});
        }
    }

#include "fair_search.inc"

    // A directed nonmetric graph can have a feasible longer route even if
    // no direct pickup/delivery pair fits. Find its shortest eligible chain.
    void directed_seeds(Fleet& fleet) {
        const double inf = std::numeric_limits<double>::infinity();
        std::vector<std::pair<int, int>> blocked;
        auto try_paths = [&](int blocked_from, int blocked_to, bool collect) {
            std::vector<double> distance(stride, inf);
            std::vector<int> previous(stride);
            std::vector<unsigned char> settled(stride);
            for (int site : suppliers)
                if (blocked_from != 0 || blocked_to != site) distance[site] = edge(0, site);
            std::vector<std::pair<double, int>> ends;
            for (int pass = 0; pass < n && !done(); ++pass) {
                int node = 0;
                for (int i = 1; i <= n; ++i)
                    if (!settled[i] && distance[i] < distance[node]) node = i;
                if (!node || distance[node] > limit + time_tolerance()) break;
                settled[node] = 1;
                const double total = distance[node] + edge(node, 0);
                if (!pickup(node) && fits(total) && (blocked_from != node || blocked_to != 0))
                    ends.emplace_back(total, node);
                for (int i = 1; i <= n; ++i) {
                    if (settled[i] || (blocked_from == node && blocked_to == i)) continue;
                    const double cost = distance[node] + edge(node, i);
                    if (cost < distance[i]) {
                        distance[i] = cost;
                        previous[i] = node;
                    }
                }
            }
            std::sort(ends.begin(), ends.end());
            Fleet trial(k);
            for (const auto& [cost, end] : ends) {
                if (done()) break;
                auto& route = trial[0].route;
                route.clear();
                for (int site = end; site; site = previous[site]) route.push_back(site);
                std::reverse(route.begin(), route.end());
                update(trial[0]);
                if (record(trial, trial[0].self, "directed-seed")) {
                    fleet = best;
                    return true;
                }
                if (collect) {
                    int prev = 0;
                    for (int site : route) {
                        if (blocked.size() == 12) break;
                        const std::pair pair{prev, site};
                        if (std::find(blocked.begin(), blocked.end(), pair) == blocked.end())
                            blocked.push_back(pair);
                        prev = site;
                    }
                }
            }
            return false;
        };
        if (!try_paths(-1, -1, true)) {
            // A binary64 clock tie can hide an exactly feasible path. Try
            // different predecessors after the exact interface rejects a chain.
            for (auto [from, to] : blocked)
                if (done() || try_paths(from, to, false)) break;
        }
    }

    void walk(Fleet& fleet, int rounds, double heat) {
        double value = measure(fleet);
        const double scale = std::max(1e-100, std::max(value, best_value));
        const double time_scale = limit > 0 ? limit : 1;
        const auto violation = [&](double elapsed) {
            return std::max(0.0, elapsed - limit) / time_scale;
        };
        auto owner = occupancy(fleet);
        Fleet local_best = fleet;
        double local_value = value;
        Levels merged;
        for (int step = 0; step < rounds && !done(); ++step) {
            const int a = pick(k);
            const auto& original = fleet[a].route;
            if (original.empty()) continue;
            const int na = static_cast<int>(original.size());
            const int i = pick(na);
            const int action = pick(12);
            int b = -1;
            Column first, second;
            first.route = original;
            if (action < 3) {
                const int j = pick(na);
                if (i == j) continue;
                if (action == 0) std::swap(first.route[i], first.route[j]);
                else if (action == 1)
                    std::reverse(first.route.begin() + std::min(i, j),
                                 first.route.begin() + std::max(i, j) + 1);
                else if (i < j) std::rotate(first.route.begin() + i, first.route.begin() + i + 1,
                                            first.route.begin() + j + 1);
                else std::rotate(first.route.begin() + j, first.route.begin() + i, first.route.begin() + i + 1);
            } else if (action == 6) {
                if (na <= 2) continue;
                first.route.erase(first.route.begin() + i);
            } else if (action < 6) {
                const auto& candidates = pickup(original[i]) ? suppliers : agencies;
                const int site = candidates[pick(static_cast<int>(candidates.size()))];
                if (owner[site] >= 0) continue;
                if (action < 5) first.route[i] = site;
                else first.route.insert(first.route.begin() + pick(na + 1), site);
            } else {
                if (k < 2) continue;
                b = (a + 1 + pick(k - 1)) % k;
                second.route = fleet[b].route;
                const int nb = static_cast<int>(second.route.size());
                if (!nb) continue;
                const int j = pick(nb);
                if (action < 9) std::swap(first.route[i], second.route[j]);
                else if (action == 9 && na > 2) {
                    second.route.insert(second.route.begin() + j, first.route[i]);
                    first.route.erase(first.route.begin() + i);
                } else if (action >= 10) {
                    first.route.erase(first.route.begin() + i, first.route.end());
                    first.route.insert(first.route.end(), fleet[b].route.begin() + j, fleet[b].route.end());
                    second.route.erase(second.route.begin() + j, second.route.end());
                    second.route.insert(second.route.end(), original.begin() + i, original.end());
                } else continue;
            }
            if (!endpoints(first.route) || !endpoints(second.route)) continue;
            const double ta = route_time(first.route), tb = route_time(second.route);
            if (violation(ta) > 0.22 || violation(tb) > 0.22) continue;
            const auto bg = background(fleet, a, b);
            update(first);
            double gain, late = violation(ta) - violation(fleet[a].elapsed);
            double time_delta = ta - fleet[a].elapsed;
            if (b < 0) gain = contribution(first, bg) - contribution(fleet[a], bg);
            else {
                update(second);
                gain = pair_value(first.levels, second.levels, bg, merged)
                      - pair_value(fleet[a].levels, fleet[b].levels, bg, merged);
                late += violation(tb) - violation(fleet[b].elapsed);
                time_delta += tb - fleet[b].elapsed;
            }
            const double cooling = (1.0 - static_cast<double>(step) / rounds);
            const double temp = heat * (0.12 + 0.88 * cooling * cooling);
            const double time_penalty = 1.8 + 4.2 * (1 - cooling);
            const double delta = gain / scale - time_penalty * late - 0.004 * time_delta / time_scale;
            if (delta < 0 && uniform() >= std::exp(delta / temp)) continue;
            for (int site : fleet[a].route) owner[site] = -1;
            if (b >= 0) for (int site : fleet[b].route) owner[site] = -1;
            fleet[a] = std::move(first);
            for (int site : fleet[a].route) owner[site] = a;
            if (b >= 0) {
                fleet[b] = std::move(second);
                for (int site : fleet[b].route) owner[site] = b;
            }
            value += gain;
            if (value >= local_value - 1e-12 * local_value &&
                std::all_of(fleet.begin(), fleet.end(), [&](const auto& col) { return fits(col.elapsed); })) {
                value = measure(fleet);
                local_value = value;
                local_best = fleet;
                record(fleet, value, "anneal");
            }
        }
        fleet = std::move(local_best);
    }

    // v2.1's best long-run improvements came from rebuilding one whole route
    // against the rest of the fleet. Reproduce that move with the v3 greedy
    // repair: clear one slot, refill it at several merit ratios, archive every
    // generated column for the set-packing master, and keep the best fleet.
    void rebuild_slot(Fleet& fleet, int slot, double ratio, double noise) {
        Fleet trial = fleet;
        trial[slot] = Column{};
        repair(trial, false, ratio, noise, 0);
        const double value = measure(trial);
        archive(trial, value);
        record(trial, value, "rebuild-slot");
        if (value > measure(fleet)) fleet = std::move(trial);
    }

    // Clearing two routes at once is a much larger perturbation than the
    // single-slot rebuild and can replace the station mix of a whole region.
    void rebuild_pair(Fleet& fleet, int a, int b, double ratio, double noise) {
        Fleet trial = fleet;
        trial[a] = Column{};
        trial[b] = Column{};
        repair(trial, false, ratio, noise, 0);
        const double value = measure(trial);
        archive(trial, value);
        record(trial, value, "rebuild-pair");
        if (value > measure(fleet)) fleet = std::move(trial);
    }

    // Intensive joint improvement around the incumbent.
    void deep_finalize(Fleet& fleet, bool broad) {
        // Work on a copy: record() moves the global incumbent forward only
        // when a candidate is a strict improvement, so an interrupted pass
        // cannot leave the reported best fleet in a worse state.
        Fleet work = fleet;
        for (int round = 0; round < 64 && !done(); ++round) {
            Fleet snapshot = work;
            const double before = measure(work);
            bool changed = false;
            for (auto& col : work) {
                if (done()) break;
                // Role-preserving shortening keeps the allocation levels.
                if (shorten(col.route, true, 8)) {
                    update(col);
                    changed = true;
                }
                if (done()) break;
                // A role-agnostic pass can free time that the preserving pass
                // cannot. The whole round is reverted if it lowers the score.
                if (shorten(col.route, false, 8)) {
                    update(col);
                    changed = true;
                }
            }
            if (done()) break;
            if (k > 1) {
                bool any = false;
                for (int a = 0; a < k && !done(); ++a)
                    for (int b = a + 1; b < k && !done(); ++b)
                        if (cross_improve(work, a, b)) any = true;
                if (any) changed = true;
            }
            if (done()) break;
            if (exchange_unvisited(work, 40)) changed = true;
            if (done()) break;
            if (relocate(work)) changed = true;
            if (done()) break;
            if (round == 0) {
                for (int slot = 0; slot < k && !done(); ++slot)
                    for (const auto [ratio, noise] : {std::pair{0.65, 0.0},
                                                      std::pair{3.0, 0.20},
                                                      std::pair{10.0, 0.40}}) {
                        if (done()) break;
                        rebuild_slot(work, slot, ratio, noise);
                    }
                if (broad) {
                    for (int a = 0; a < k && !done(); ++a)
                        for (int b = a + 1; b < k && !done(); ++b)
                            for (const double ratio : {0.65, 6.0}) {
                                if (done()) break;
                                rebuild_pair(work, a, b, ratio, 0.15 * ratio);
                            }
                }
            }
            if (done()) break;
            record(work, measure(work), "deep-finalize");
            const double after = measure(work);
            if (after + 1e-12 * std::abs(before) < before) {
                work = std::move(snapshot);
                break;
            }
            if (after > before + 1e-12 * std::abs(before)) changed = true;
            if (!changed) break;
        }
    }

    void small_enumeration(Fleet& current) {
        if (n > 12 || done()) return;
        const auto stop = Clock::now() + std::chrono::milliseconds(60);
        if (Clock::now() >= stop) return;
        struct Item { unsigned mask; Column column; };
        std::unordered_map<unsigned, std::vector<Column>> unique;
        Route route;
        std::uint64_t work = 0;
        const auto enumerate = [&](auto&& self, unsigned mask, double elapsed) -> void {
            if ((++work & 127) == 0 && Clock::now() >= stop) work = 4'000'000;
            if (work >= 4'000'000) return;
            const int last = route.back();
            if (!pickup(last) && fits(elapsed + edge(last, 0))) {
                Column candidate;
                candidate.route = route;
                update(candidate);
                auto& options = unique[mask];
                const auto worse = [](const Column& a, const Column& b) {
                    if (a.self != b.self) return a.self < b.self;
                    return a.elapsed > b.elapsed;
                };
                const auto same = [&](const Column& old) {
                    return old.levels.size() == candidate.levels.size()
                        && std::equal(old.levels.begin(), old.levels.end(),
                                      candidate.levels.begin(), [](Level a, Level b) {
                            return a.rate == b.rate && a.people == b.people;
                        });
                };
                const auto matching = std::find_if(options.begin(), options.end(), same);
                if (matching != options.end()) {
                    if (matching->elapsed > candidate.elapsed) *matching = std::move(candidate);
                } else if (options.size() < 8) {
                    options.push_back(std::move(candidate));
                } else {
                    auto weakest = std::min_element(options.begin(), options.end(), worse);
                    if (worse(*weakest, candidate)) *weakest = std::move(candidate);
                }
            }
            for (int node = 1; node <= n; ++node) {
                const unsigned bit = 1U << (node - 1);
                if ((mask & bit) || !fits(elapsed + edge(last, node))) continue;
                route.push_back(node);
                self(self, mask | bit, elapsed + edge(last, node));
                route.pop_back();
                if (work >= 4'000'000) break;
            }
        };
        for (int source : suppliers) {
            if (!fits(edge(0, source)) || Clock::now() >= stop) continue;
            route = {source};
            enumerate(enumerate, 1U << (source - 1), edge(0, source));
        }
        std::vector<Item> columns;
        for (auto& [mask, entries] : unique)
            for (auto& col : entries) columns.push_back({mask, std::move(col)});
        std::sort(columns.begin(), columns.end(), [](const auto& a, const auto& b) {
            if (a.column.self != b.column.self) return a.column.self > b.column.self;
            if (a.mask != b.mask) return a.mask < b.mask;
            return a.column.route < b.column.route;
        });
        Fleet draft(k);
        const auto pack_end = Clock::now() + std::chrono::milliseconds(30);
        const auto pack_small = [&](auto&& self, std::size_t index, unsigned mask, int slot, double root) -> void {
            if (slot == k || done() || Clock::now() >= pack_end) return;
            for (std::size_t j = index; j < columns.size(); ++j) {
                if (done() || Clock::now() >= pack_end) return;
                const auto& col = columns[j];
                const auto own_root = std::sqrt(col.column.self);
                if (std::pow(root + (k - slot) * own_root, 2) <= best_value) break;
                if (mask & col.mask) continue;
                draft[slot] = col.column;
                record(draft, measure(draft), "small-enumeration");
                self(self, j + 1, mask | col.mask, slot + 1, root + own_root);
                draft[slot] = {};
            }
        };
        pack_small(pack_small, 0, 0, 0, 0);
        if (best_value > measure(current)) current = best;
    }

public:
    Search(int nodes, int vehicles, unsigned seed, double seconds, double cap,
           double duration, const double* stock, const double* pop, const double* cost,
           Publish notify)
        : n(nodes), k(vehicles), stride(nodes + 1), capacity(cap), limit(duration),
          total_people(0), arc(cost),
          supply(stock, stock + stride), people(pop, pop + stride), profile(supply, people, cap),
          deadline(Clock::now() + std::chrono::duration_cast<Clock::duration>(
                     std::chrono::duration<double>(seconds))),
          rng(seed), publish(notify) {
        const double max_people = *std::max_element(people.begin(), people.end());
        if (max_people > 1e85 || (max_people > 0 && max_people < 1e-85))
            for (double& p : people) p /= max_people;
        total_people = std::accumulate(people.begin() + 1, people.end(), 0.0);
        const double max_stock = std::min(capacity, *std::max_element(supply.begin(), supply.end()));
        if (max_stock > 1e85 || (max_stock > 0 && max_stock < 1e-85)) {
            quantity_scale = max_stock;
            for (double& q : supply) q = std::min(q, capacity) / quantity_scale;
            capacity = std::min(capacity / quantity_scale, static_cast<double>(n));
            profile.set_capacity(capacity);
        }
        for (int i = 1; i <= n; ++i) {
            if (pickup(i)) suppliers.push_back(i);
            else agencies.push_back(i);
        }
        k = std::min({k, static_cast<int>(suppliers.size()), static_cast<int>(agencies.size())});
        best.resize(k);
        by_people = agencies;
        std::sort(by_people.begin(), by_people.end(), [&](int a, int b) {
            return people[a] == people[b] ? a < b : people[a] > people[b];
        });
    }

    int run() {
        if (k == 0 || capacity == 0 || done()) return 0;
        destroy_weight[5] = 0.6;
        destroy_weight[6] = 0.6;
        setup_seeds();
        setup_sparse_arcs();
        Fleet current(k);
        if (seeds.empty()) {
            directed_seeds(current);
            if (current[0].route.empty()) return callback_failure ? 3 : 0;
        } else {
            const auto merit = [&](const Seed& seed) {
                return seed.rate * people[seed.agency] * people[seed.agency] /
                    (1 + (limit > 0 ? seed.time / limit : 0));
            };
            const auto first = std::max_element(seeds.begin(), seeds.end(), [&](const Seed& a, const Seed& b) {
                return merit(a) < merit(b);
            });
            Fleet seed(k);
            seed[0].route = {first->source, first->agency};
            update(seed[0]);
            record(seed, seed[0].self, "short-seed");
            if (n >= 128) current = std::move(seed);
        }
        fill_sparse_routes(current, Clock::now() + std::chrono::milliseconds(190));
        repair(current, false, 3.0, 0);
        record(current, measure(current), "initial");
        if (best_value == 0 && !seeds.empty()) directed_seeds(current);
        polish(current, true);
        record(current, measure(current), "initial-polish");
        small_enumeration(current);
        double value = measure(current);
        std::uint64_t last_best = 0;
        while (!done()) {
            const auto iteration = ++iterations;
            const double previous_incumbent = best_value;
            if (iteration % 137 == 0 && iteration - last_best > 24) {
                deep_finalize(best, true);
                current = best;
                value = measure(current);
            }
            const auto stagnation = iteration - last_best;
            if (iteration % 4 == 0) {
                const auto clock = Clock::now();
                walk(current, n >= 60 ? 720 : 240, 0.009 + 0.009 * uniform());
                g_walk_us += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - clock).count();
                value = measure(current);
                archive(current, value);
            }
            if (iteration % 5 == 0) {
                Fleet warm = iteration % 10 == 0 ? best : current;
                if (reprice(warm, iteration)) {
                    current = std::move(warm);
                    value = measure(current);
                }
            }
            if (iteration % 11 == 0) {
                const auto clock = Clock::now();
                if (compatible_master(current)) value = measure(current);
                g_master_us += std::chrono::duration_cast<std::chrono::microseconds>(
                    Clock::now() - clock).count();
            }

            const bool restart = stagnation > 12 && iteration % 23 == 0;
            const bool scratch = stagnation > 20 && iteration % 47 == 0;
            Fleet trial = scratch ? Fleet(k) : (restart ? best : current);
            const int op = choose(destroy_weight), repair_op = choose(repair_weight);
            const auto destroy_clock = Clock::now();
            const int target = destroy(trial, op, restart || iteration % 13 == 0);
            g_destroy_us += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - destroy_clock).count();
            if (iteration % 3 == 0) {
                for (auto& col : trial) {
                    if (shorten(col.route, iteration % 7 != 0, 3)) update(col);
                }
            }
            if (!sparse_arcs.empty() && (op == 2 || scratch)) {
                fill_sparse_routes(trial, Clock::now() + std::chrono::milliseconds(20));
            }
            const double ratio = 0.65 + 10 * uniform() * uniform();
            const auto repair_clock = Clock::now();
            repair(trial, repair_op == 1, ratio, 0.04 + 0.32 * uniform(), target);
            g_repair_us += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - repair_clock).count();
            double score = measure(trial);
            bool improved = record(trial, score, "alns");
            const bool close = score >= 0.97 * best_value;
            if (iteration % 5 == 1 || improved || (close && iteration % 3 == 0)) {
                const auto polish_clock = Clock::now();
                polish(trial, close || improved);
                g_polish_us += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - polish_clock).count();
                score = measure(trial);
                improved |= record(trial, score, "alns-polish");
            }
            if (score >= 0.86 * best_value || iteration % 11 == 0) archive(trial, score);
            const double heat = 0.0017 + 0.018 * std::pow(1 - (iteration % 53) / 53.0, 2);
            const double scale = std::max(best_value, 1e-100);
            const bool accepted = score >= value || uniform() < std::exp((score - value) / (scale * heat));
            if (accepted || improved || (scratch && iteration % 2)) {
                current = std::move(trial);
                value = score;
            }
            if (best_value > previous_incumbent) last_best = iteration;
            const double reward = improved ? 4.0 : (accepted && score >= 0.98 * best_value ? 1.5 : 0.6);
            destroy_weight[op] = 0.90 * destroy_weight[op] + 0.10 * reward;
            repair_weight[repair_op] = 0.90 * repair_weight[repair_op] + 0.10 * reward;
        }
        g_iterations = iterations;
        return callback_failure ? 3 : 0;
    }
};

}  // namespace v3

extern "C" unsigned long long v4_iterations() { return v3::g_iterations; }
extern "C" unsigned long long v4_repair_us() { return v3::g_repair_us; }
extern "C" unsigned long long v4_walk_us() { return v3::g_walk_us; }
extern "C" unsigned long long v4_polish_us() { return v3::g_polish_us; }
extern "C" unsigned long long v4_master_us() { return v3::g_master_us; }
extern "C" unsigned long long v4_destroy_us() { return v3::g_destroy_us; }
extern "C" unsigned long long v4_deep_us() { return v3::g_deep_us; }

extern "C" int astra_primal(int n, int k, unsigned seed, double seconds, double capacity,
                          double limit, const double* supply, const double* population,
                          const double* arcs, v3::Publish publish) {
    if (n < 0 || n > 512 || k < 1 || !std::isfinite(seconds) || seconds < 0 || seconds > 86400 ||
        !std::isfinite(capacity) || capacity < 0 || !std::isfinite(limit) || limit < 0 ||
        !supply || !population || !arcs || !publish) return 2;
    for (int i = 1; i <= n; ++i) {
        if (!std::isfinite(supply[i]) || !std::isfinite(population[i]) || supply[i] < 0 ||
            population[i] < 0 || ((supply[i] > 0) == (population[i] > 0))) return 2;
    }
    for (int i = 0; i < (n + 1) * (n + 1); ++i)
        if (!std::isfinite(arcs[i]) || arcs[i] < 0) return 2;
    try {
        v3::Search search(n, k, seed, seconds, capacity, limit, supply, population,
                          arcs, publish);
        return search.run();
    } catch (const std::exception&) {
        return 4;
    }
}
