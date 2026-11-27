#include "primal_profile.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <numeric>
#include <random>
#include <utility>

namespace subsequence {

using Clock = std::chrono::steady_clock;
using v3::Levels;
using v3::Route;
using Fleet = std::vector<Route>;
using Publisher = int (*)(const char*, const int*, int, double);
constexpr double infinity = std::numeric_limits<double>::infinity();

struct End {
    float reward;
    int time, last;
};

class Search {
    int n, k, horizon, stride;
    double limit, total;
    const double* edges;
    std::vector<double> supply, people;
    std::vector<int> ticks;
    v3::Profile water;
    std::mt19937 rng;
    Clock::time_point deadline;
    Publisher notify;
    Fleet best;
    double best_value;
    bool failed = false;

    bool done() const { return failed || Clock::now() >= deadline; }
    bool source(int node) const { return supply[node] > 0; }
    int pick(int count) { return static_cast<int>(rng() % count); }
    double uniform() { return std::generate_canonical<double, 32>(rng); }
    double edge(int a, int b) const { return edges[a * (n + 1) + b]; }
    int time(int a, int b) const { return ticks[a * (n + 1) + b]; }

    double route_time(const Route& route) const {
        double result = 0;
        int last = 0;
        for (int node : route) {
            result += edge(last, node);
            last = node;
        }
        return result + edge(last, 0);
    }

    v3::Background background(const Fleet& fleet, int omit) {
        v3::Background result;
        Levels levels;
        for (int v = 0; v < k; ++v) {
            if (v == omit) continue;
            water(fleet[v], levels);
            result.levels.insert(result.levels.end(), levels.begin(), levels.end());
        }
        result.finish();
        return result;
    }

    double contribution(const Levels& levels, const v3::Background& fixed) const {
        return v3::lorentz(levels, total) + fixed.cross(levels, total);
    }

    bool publish(const Fleet& fleet, double value) {
        if (done() || value <= best_value * (1 + 2e-13)) return false;
        std::vector<int> packed{0};
        for (const auto& route : fleet) {
            if (route.empty()) continue;
            ++packed[0];
            packed.push_back(static_cast<int>(route.size()));
            packed.insert(packed.end(), route.begin(), route.end());
        }
        const int accepted = notify("giant-subsequence", packed.data(), static_cast<int>(packed.size()), value);
        failed = accepted < 0;
        if (accepted <= 0) return false;
        best = fleet;
        best_value = value;
        return true;
    }

    Route giant(const Fleet& fleet, int route, unsigned round) {
        std::vector<bool> unavailable(n + 1);
        for (int v = 0; v < k; ++v)
            if (v != route)
                for (int node : fleet[v]) unavailable[node] = true;
        Route order = fleet[route];
        const int size = static_cast<int>(order.size());
        if (size >= 7 && round > 0) {
            const int begin = pick(size - 4);
            const int end = begin + 2 + pick(std::min(8, size - begin - 1));
            if (round % 5 == 1)
                order.erase(order.begin() + begin, order.begin() + end);
            if (round % 5 == 2)
                std::reverse(order.begin() + begin, order.begin() + end);
            if (round % 5 == 3)
                std::rotate(order.begin(), order.begin() + 1 + pick(size - 1), order.end());
        }
        for (int node : order) unavailable[node] = true;
        Route available;
        for (int node = 1; node <= n; ++node)
            if (!unavailable[node]) available.push_back(node);
        std::vector<double> priority(n + 1);
        if (round)
            for (int node : available)
                priority[node] = (.012 + .008 * (round % 4)) * (uniform() - .5);

        // Keep the chosen backbone order, embedding the other nodes by least
        // added directed cost. A node can be skipped later by the DAG.
        while (!available.empty() && !done()) {
            double least = infinity;
            int at = 0, insert = 0;
            for (int index = 0; index < static_cast<int>(available.size()); ++index) {
                const int node = available[index];
                for (int i = 0; i <= static_cast<int>(order.size()); ++i) {
                    const int before = i ? order[i - 1] : 0;
                    const int after = i < static_cast<int>(order.size()) ? order[i] : 0;
                    const double extra = edge(before, node) + edge(node, after) - edge(before, after);
                    const double tie = round ? priority[node] * (1 + 0.037 * ((node + i) % 11)) : 0;
                    if (extra + tie < least) {
                        least = extra + tie;
                        at = index;
                        insert = i;
                    }
                }
            }
            order.insert(order.begin() + insert, available[at]);
            available.erase(available.begin() + at);
        }
        return order;
    }

    Route traceback(const Route& order, const std::vector<int>& predecessor, End end) const {
        Route route;
        int last = end.last, remaining = end.time;
        while (last >= 0) {
            route.push_back(order[last]);
            const int before = predecessor[last * stride + remaining];
            remaining -= time(before < 0 ? 0 : order[before], order[last]);
            last = before;
        }
        std::reverse(route.begin(), route.end());
        return route;
    }

    void select(const Route& order, const v3::Background& fixed, double beta,
                Route& selected, double& selected_value, double& quality,
                double time_credit) {
        const int m = static_cast<int>(order.size());
        std::vector<float> rewards(m * stride, -std::numeric_limits<float>::infinity());
        std::vector<int> predecessor(m * stride, -1);
        std::vector<End> ends;
        for (int j = 0; j < m; ++j) {
            if (done()) return;
            const int node = order[j];
            const float prize = static_cast<float>(supply[node] + beta * people[node]);
            auto* row = rewards.data() + j * stride;
            auto* parents = predecessor.data() + j * stride;
            const int enter = time(0, node);
            if (source(node) && enter <= horizon) row[enter] = prize;
            for (int i = 0; i < j; ++i) {
                const int distance = time(order[i], node);
                const auto* earlier = rewards.data() + i * stride;
                for (int t = distance; t <= horizon; ++t) {
                    const float value = earlier[t - distance] + prize;
                    if (value > row[t]) {
                        row[t] = value;
                        parents[t] = i;
                    }
                }
            }
            if (source(node)) continue;
            const int go_home = time(node, 0);
            float high = -std::numeric_limits<float>::infinity();
            // Retain only the additive record holders for each ending station.
            for (int t = 0; t + go_home <= horizon; ++t) {
                if (row[t] <= high) continue;
                high = row[t];
                ends.push_back({high, t, j});
            }
        }

        const int count = std::min<std::size_t>(48, ends.size());
        std::partial_sort(ends.begin(), ends.begin() + count, ends.end(), [](End a, End b) {
            if (a.reward != b.reward) return a.reward > b.reward;
            if (a.time != b.time) return a.time < b.time;
            return a.last < b.last;
        });
        Levels levels;
        for (int i = 0; i < count && !done(); ++i) {
            auto route = traceback(order, predecessor, ends[i]);
            const double elapsed = route_time(route);
            if (elapsed > limit + 2e-12) continue;
            water(route, levels);
            const double own = contribution(levels, fixed);
            const double merit = own - time_credit * elapsed;
            if (merit > quality + 2e-13 * std::abs(quality)) {
                selected = std::move(route);
                selected_value = own;
                quality = merit;
            }
        }
    }

public:
    Search(int nodes, int slots, unsigned seed, double seconds, double capacity, double limit_,
           const double* stocks, const double* pop, const double* arcs, int max_ticks, Publisher callback)
        : n(nodes), k(slots), horizon(limit_ > 0 ? max_ticks : 1), stride(horizon + 1),
          limit(limit_), edges(arcs), supply(stocks, stocks + n + 1), people(pop, pop + n + 1),
          ticks((n + 1) * (n + 1)), water(supply, people, capacity), rng(seed),
          deadline(Clock::now() + std::chrono::duration_cast<Clock::duration>(
              std::chrono::duration<double>(seconds))), notify(callback), best_value(0) {
        total = std::accumulate(people.begin(), people.end(), 0.0);
        for (int i = 0; i < static_cast<int>(ticks.size()); ++i)
            ticks[i] = static_cast<int>(std::lround(edges[i] * horizon));
    }

    int run(Fleet fleet) {
        best = fleet;
        best_value = v3::lorentz(background(fleet, -1).levels, total);
        const double base_rate = std::accumulate(supply.begin(), supply.end(), 0.0) / total;
        for (unsigned round = 0; !done(); ++round) {
            const int a = round % k;
            const unsigned mode = round / k;
            if (round % (k * 7) == 0) fleet = best;
            const auto fixed = background(fleet, a);
            Levels current;
            water(fleet[a], current);
            const double baseline = contribution(current, fixed);
            const double rate = current.empty() ? base_rate :
                std::accumulate(current.begin(), current.end(), 0.0,
                    [](double sum, v3::Level level) { return sum + level.people * level.rate; }) /
                std::accumulate(current.begin(), current.end(), 0.0,
                    [](double sum, v3::Level level) { return sum + level.people; });
            const double fixed_value = v3::lorentz(fixed.levels, total);
            const double cash = (mode % 7 > 1 ? 0.034 : 0.0) * best_value;
            double selected_value = baseline, quality = baseline - cash * route_time(fleet[a]);
            auto selected = fleet[a];
            const auto order = giant(fleet, a, mode);
            constexpr std::array<double, 5> tilt{0.38, 0.65, 1.0, 1.6, 2.8};
            for (double multiplier : tilt) {
                if (done()) break;
                select(order, fixed, multiplier * std::max(rate, 0.35 * base_rate),
                       selected, selected_value, quality, cash);
                if (selected_value + fixed_value > best_value * (1 + 2e-13)) {
                    auto candidate = fleet;
                    candidate[a] = selected;
                    publish(candidate, selected_value + fixed_value);
                }
            }
            fleet[a] = std::move(selected);
        }
        return failed ? 3 : 0;
    }
};

} // namespace subsequence

extern "C" int astra_subsequence(int n, int k, unsigned seed, double seconds, double capacity,
                                 double limit, const double* supply, const double* population,
                                 const double* arcs, int ticks, const int* packed, int count,
                                 subsequence::Publisher publisher) {
    if (n < 2 || n > 512 || k < 1 || ticks < 1 || ticks > 800 || count < 3 || !publisher ||
        !packed || !(capacity > 0) || !(seconds > 0) || packed[0] < 1 || packed[0] > k) return 2;
    try {
        subsequence::Fleet warm(k);
        int offset = 1;
        for (int i = 0; i < packed[0]; ++i) {
            if (offset >= count) return 2;
            const int size = packed[offset++];
            if (size < 2 || size > count - offset) return 2;
            warm[i].assign(packed + offset, packed + offset + size);
            offset += size;
        }
        if (offset != count) return 2;
        subsequence::Search search(n, k, seed, seconds, capacity, limit, supply, population,
                                    arcs, ticks, publisher);
        return search.run(std::move(warm));
    } catch (const std::exception&) {
        return 4;
    }
}
