#include "primal_profile.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <random>
#include <utility>

namespace surgery {

using Clock = std::chrono::steady_clock;
using v3::Background;
using v3::Levels;
using v3::Route;
using Fleet = std::vector<Route>;
using Publish = int (*)(const char*, const int*, int, double);

constexpr double time_error = 2e-12;

struct Problem {
    int n;
    double capacity, limit;
    const double* costs;
    const double* visits;
    const double* flows;
    std::vector<double> supply, people;
    std::vector<int> suppliers;
    double total;

    Problem(int n_, double capacity_, double limit_, const double* stock,
            const double* population, const double* arcs, const double* y, const double* x)
        : n(n_), capacity(capacity_), limit(limit_), costs(arcs), visits(y), flows(x),
          supply(stock, stock + n + 1), people(population, population + n + 1),
          total(std::accumulate(people.begin(), people.end(), 0.0)) {
        for (int i = 1; i <= n; ++i)
            if (supplier(i)) suppliers.push_back(i);
    }

    double edge(int x, int y) const { return costs[x * (n + 1) + y]; }
    bool supplier(int node) const { return supply[node] > 0; }
    bool fits(double cost) const { return cost <= limit + time_error; }
    double hint(int prev, int node, int next) const {
        return visits ? visits[node] + flows[prev * (n + 1) + node] +
            flows[node * (n + 1) + next] : 0.0;
    }
};

// Retained directed subpaths can be spliced and reversed without scanning their sites.
struct Path {
    int first = 0, last = 0;
    double inner = 0;
};

class RouteInfo {
    const Problem& p;
    std::vector<double> forward, backward;

public:
    Route sites;
    Levels levels;
    double cost = 0;

    RouteInfo(const Problem& p_, const Route& route, v3::Profile& water)
        : p(p_), forward(route.size()), backward(route.size()), sites(route) {
        for (std::size_t i = 1; i < sites.size(); ++i) {
            forward[i] = forward[i - 1] + p.edge(sites[i - 1], sites[i]);
            backward[i] = backward[i - 1] + p.edge(sites[i], sites[i - 1]);
        }
        if (!sites.empty())
            cost = p.edge(0, sites.front()) + forward.back() + p.edge(sites.back(), 0);
        water(sites, levels);
    }

    int size() const { return static_cast<int>(sites.size()); }

    Path span(int first, int last, bool reverse) const {
        if (first == last) return {};
        if (reverse)
            return {sites[last - 1], sites[first], backward[last - 1] - backward[first]};
        return {sites[first], sites[last - 1], forward[last - 1] - forward[first]};
    }
};

struct Segment {
    const RouteInfo* route;
    int first, last;
    bool reverse;
};

struct Edit {
    std::array<Segment, 4> segments;
    int count = 0;

    void append(const RouteInfo& route, int first, int last, bool reverse = false) {
        if (first < last) segments[count++] = {&route, first, last, reverse};
    }

    void insert(int node) { segments[count++] = {nullptr, node, 0, false}; }

    Route materialize() const {
        Route result;
        for (int i = 0; i < count; ++i) {
            const auto [route, first, last, reverse] = segments[i];
            if (!route) result.push_back(first);
            else if (reverse)
                for (int j = last; j-- > first;) result.push_back(route->sites[j]);
            else result.insert(result.end(), route->sites.begin() + first, route->sites.begin() + last);
        }
        return result;
    }

    double time(const Problem& p) const {
        if (!count) return 0;
        double elapsed = 0;
        int last = 0;
        for (int i = 0; i < count; ++i) {
            const auto& s = segments[i];
            const auto path = s.route ? s.route->span(s.first, s.last, s.reverse)
                                      : Path{s.first, s.first, 0};
            if (!i && !p.supplier(path.first)) return 2;
            elapsed += p.edge(last, path.first) + path.inner;
            last = path.last;
        }
        return p.supplier(last) ? 2 : elapsed + p.edge(last, 0);
    }
};

struct Choice {
    int a = -1, b = -1;
    Route first, second;
    double value = -std::numeric_limits<double>::infinity();
    double merit = -std::numeric_limits<double>::infinity();
};

class Search {
    const Problem& p;
    int k;
    Clock::time_point deadline;
    std::mt19937 random;
    Publish publish;
    v3::Profile water;
    Fleet best;
    Fleet pending;
    double best_value = 0;
    double pending_value = 0;
    const char* pending_source = nullptr;
    Clock::time_point last_publication{};
    bool error = false;

    bool done() const { return error || Clock::now() >= deadline; }
    int pick(int count) { return static_cast<int>(random() % count); }

    Background background(const std::vector<RouteInfo>& info, int a, int b = -1) {
        Background result;
        for (int i = 0; i < k; ++i)
            if (i != a && i != b)
                result.levels.insert(result.levels.end(), info[i].levels.begin(), info[i].levels.end());
        result.finish();
        return result;
    }

    std::vector<RouteInfo> describe(const Fleet& fleet) {
        std::vector<RouteInfo> result;
        result.reserve(k);
        for (const auto& route : fleet) result.emplace_back(p, route, water);
        return result;
    }

    double score(const Fleet& fleet) {
        Levels all, one;
        for (const auto& route : fleet) {
            water(route, one);
            all.insert(all.end(), one.begin(), one.end());
        }
        std::sort(all.begin(), all.end(), v3::rate_order);
        return v3::lorentz(all, p.total);
    }

    void publish_pending() {
        if (done()) return;
        if (!pending_source) return;
        std::vector<int> encoded{0};
        for (const auto& route : pending) {
            if (route.empty()) continue;
            ++encoded.front();
            encoded.push_back(static_cast<int>(route.size()));
            encoded.insert(encoded.end(), route.begin(), route.end());
        }
        const int accepted = publish(pending_source, encoded.data(), static_cast<int>(encoded.size()), pending_value);
        if (accepted > 0) {
            best = std::move(pending);
            best_value = pending_value;
        }
        pending_source = nullptr;
        pending_value = 0;
        last_publication = Clock::now();
        error = accepted < 0;
    }

    void record(const Fleet& candidate, const char* source) {
        if (done()) return;
        const double value = score(candidate);
        if (value > std::max(best_value, pending_value) * (1 + 2e-13)) {
            pending = candidate;
            pending_value = value;
            pending_source = source;
        }
        if (best_value == 0 || Clock::now() - last_publication >= std::chrono::milliseconds(85))
            publish_pending();
    }

    Route shortest_seed(const std::vector<bool>& used) {
        const double inf = std::numeric_limits<double>::infinity();
        std::vector<double> dist(p.n + 1, inf);
        std::vector<int> predecessor(p.n + 1);
        std::vector<bool> settled(p.n + 1);
        for (int node : p.suppliers)
            if (!used[node]) dist[node] = p.edge(0, node);
        double shortest = p.limit + time_error;
        int end = 0;
        while (!done()) {
            int node = 0;
            for (int i = 1; i <= p.n; ++i)
                if (!settled[i] && dist[i] < dist[node]) node = i;
            if (!node || dist[node] > shortest) break;
            settled[node] = true;
            const double close = dist[node] + p.edge(node, 0);
            if (!p.supplier(node) && close <= shortest) {
                shortest = close;
                end = node;
            }
            for (int next = 1; next <= p.n; ++next) {
                const double length = dist[node] + p.edge(node, next);
                if (!used[next] && !settled[next] && length < dist[next]) {
                    predecessor[next] = node;
                    dist[next] = length;
                }
            }
        }
        Route result;
        for (; end; end = predecessor[end]) result.push_back(end);
        std::reverse(result.begin(), result.end());
        return result;
    }

    void seed(Fleet& fleet) {
        std::vector<bool> used(p.n + 1);
        for (const auto& route : fleet)
            for (int node : route) used[node] = true;
        for (auto& route : fleet) {
            if (!route.empty() || done()) continue;
            double merit = 0;
            for (int s : p.suppliers) {
                if (used[s] || done()) continue;
                for (int d = 1; d <= p.n; ++d) {
                    if (used[d] || p.supplier(d)) continue;
                    const double time = p.edge(0, s) + p.edge(s, d) + p.edge(d, 0);
                    if (!p.fits(time)) continue;
                    const double value = p.supply[s] * p.people[d] *
                        (1 + 0.12 * p.hint(0, s, d)) / (0.055 + time);
                    if (value > merit) {
                        route = {s, d};
                        merit = value;
                    }
                }
            }
            if (route.empty()) route = shortest_seed(used);
            for (int site : route) used[site] = true;
            record(fleet, "surgery-seed");
        }
    }

    double contribution(const Levels& levels, const Background& fixed) const {
        return v3::lorentz(levels, p.total) + fixed.cross(levels, p.total);
    }

    Choice window_search(const std::vector<RouteInfo>& info, int a, bool paired,
                         double time_credit, bool shaking, bool construct = false) {
        const auto& route = info[a];
        const auto fixed = background(info, a);
        const double baseline = contribution(route.levels, fixed);
        Choice selected;
        selected.merit = shaking ? -0.13 * best_value : 2e-13 * best_value;
        Levels levels;
        std::vector<bool> used(p.n + 1);
        for (const auto& r : info)
            for (int node : r.sites) used[node] = true;
        std::vector<int> unused;
        for (int node = 1; node <= p.n; ++node)
            if (!used[node]) unused.push_back(node);
        if (shaking) std::shuffle(unused.begin(), unused.end(), random);
        const int size = route.size();
        const int start = shaking ? pick(size + 1) : 0;
        const int positions = shaking ? std::min(4, size + 1) : size + 1;
        for (int step = 0; step < positions && !done(); ++step) {
            const int first = (start + step) % (size + 1);
            const int left = first ? route.sites[first - 1] : 0;
            const int max_removed = construct ? 0 : (paired ? 3 : 1);
            for (int removed = 0; removed <= std::min(max_removed, size - first); ++removed) {
                const int last = first + removed;
                const int right = last < size ? route.sites[last] : 0;
                const auto prefix = route.span(0, first, false);
                const auto suffix = route.span(last, size, false);
                const double fixed_time = prefix.inner + suffix.inner +
                    p.edge(0, prefix.first) + p.edge(suffix.last, 0);
                const double room = p.limit + time_error - fixed_time;
                if (room < 0) continue;
                auto try_insertion = [&](int x, int y, double time) {
                    if ((x && !left && !p.supplier(x)) ||
                        (!right && p.supplier(y ? y : x))) return;
                    Edit edit;
                    edit.append(route, 0, first);
                    if (x) edit.insert(x);
                    if (y) edit.insert(y);
                    edit.append(route, last, size);
                    const int length = size - removed + (x != 0) + (y != 0);
                    if ((length == 1) || !p.fits(time)) return;
                    if (!x && !left && right && !p.supplier(right)) return;
                    if (!x && left && !right && p.supplier(left)) return;
                    auto candidate = edit.materialize();
                    water(candidate, levels);
                    const double change = contribution(levels, fixed) - baseline;
                    const double value = construct
                        ? change / (0.012 + std::max(0.0, time - route.cost))
                        : change + time_credit * (route.cost - time);
                    bool prefer = value > selected.merit;
                    if (shaking && prefer) prefer = pick(3) != 0;
                    if (prefer)
                        selected = {a, -1, std::move(candidate), {}, change, value};
                };
                if (removed && p.edge(left, right) <= room)
                    try_insertion(0, 0, fixed_time + p.edge(left, right));
                unsigned scanned = 0;
                for (int x : unused) {
                    if ((++scanned & 31) == 0 && done()) return selected;
                    if (!left && !p.supplier(x)) continue;
                    const double front = p.edge(left, x);
                    if (front > room) continue;
                    const double one = front + p.edge(x, right);
                    if (one <= room) try_insertion(x, 0, fixed_time + one);
                    if (!paired) continue;
                    for (int y : unused) {
                        if (y == x || (!right && p.supplier(y))) continue;
                        const double two = front + p.edge(x, y) + p.edge(y, right);
                        if (two <= room) try_insertion(x, y, fixed_time + two);
                    }
                }
            }
        }
        return selected;
    }

    Choice reconnect(const std::vector<RouteInfo>& info, double time_credit, bool pairs) {
        Choice selected;
        selected.merit = 2e-13 * best_value;
        Levels first_levels, second_levels, both;
        auto improved = [&](int a, int b, const Edit& edit, const Edit& other,
                            const Background& fixed, double baseline) {
            const double cost = edit.time(p);
            if (!p.fits(cost)) return;
            const double second_cost = b >= 0 ? other.time(p) : 0;
            if (!p.fits(second_cost)) return;
            auto first = edit.materialize();
            water(first, first_levels);
            double value = contribution(first_levels, fixed);
            double saved = info[a].cost - cost;
            Route second;
            if (b >= 0) {
                second = other.materialize();
                water(second, second_levels);
                both.clear();
                std::merge(first_levels.begin(), first_levels.end(), second_levels.begin(),
                           second_levels.end(), std::back_inserter(both), v3::rate_order);
                value = contribution(both, fixed);
                saved += info[b].cost - second_cost;
            }
            const double change = value - baseline;
            const double merit = change + saved * time_credit;
            if (merit > selected.merit)
                selected = {a, b, std::move(first), std::move(second), change, merit};
        };

        for (int a = 0; a < k && !done(); ++a) {
            const auto& left = info[a];
            const int na = left.size();
            const auto fixed = background(info, a);
            const double baseline = contribution(left.levels, fixed);
            for (int i = 0; i < na && !done(); ++i) {
                for (int j = i + 1; j < na; ++j) {
                    Edit reversed;
                    reversed.append(left, 0, i);
                    reversed.append(left, i, j + 1, true);
                    reversed.append(left, j + 1, na);
                    improved(a, -1, reversed, {}, fixed, baseline);
                    for (int width = 1; width <= std::min(3, j - i); ++width) {
                        Edit after;
                        after.append(left, 0, i);
                        after.append(left, i + width, j + 1);
                        after.append(left, i, i + width);
                        after.append(left, j + 1, na);
                        improved(a, -1, after, {}, fixed, baseline);
                    }
                    for (int width = 1; width <= std::min(3, na - j); ++width) {
                        Edit before;
                        before.append(left, 0, i);
                        before.append(left, j, j + width);
                        before.append(left, i, j);
                        before.append(left, j + width, na);
                        improved(a, -1, before, {}, fixed, baseline);
                    }
                }
            }
            if (!pairs) continue;
            for (int b = a + 1; b < k && !done(); ++b) {
                const auto& right = info[b];
                const int nb = right.size();
                const auto fixed_pair = background(info, a, b);
                both.clear();
                std::merge(left.levels.begin(), left.levels.end(), right.levels.begin(),
                           right.levels.end(), std::back_inserter(both), v3::rate_order);
                const double base_pair = contribution(both, fixed_pair);
                for (int i = 0; i <= na && !done(); ++i) {
                    for (int j = 0; j <= nb; ++j) {
                        auto splice = [&](int ea, int eb) {
                            Edit first, second;
                            first.append(left, 0, i);
                            first.append(right, j, eb);
                            first.append(left, ea, na);
                            second.append(right, 0, j);
                            second.append(left, i, ea);
                            second.append(right, eb, nb);
                            improved(a, b, first, second, fixed_pair, base_pair);
                        };
                        splice(na, nb);
                        for (int wa = 0; wa <= std::min(2, na - i); ++wa)
                            for (int wb = 0; wb <= std::min(2, nb - j); ++wb)
                                if (wa || wb) splice(i + wa, j + wb);
                    }
                }
            }
        }
        return selected;
    }

    void apply(Fleet& fleet, Choice choice, const char* source) {
        fleet[choice.a] = std::move(choice.first);
        if (choice.b >= 0) fleet[choice.b] = std::move(choice.second);
        record(fleet, source);
    }

public:
    Search(const Problem& p_, int k_, unsigned seed, double seconds, Publish callback)
        : p(p_), k(k_), deadline(Clock::now() + std::chrono::duration_cast<Clock::duration>(
              std::chrono::duration<double>(seconds))), random(seed), publish(callback),
          water(p.supply, p.people, p.capacity), best(k) {}

    int run(Fleet fleet, bool has_warm) {
        if (has_warm) {
            best = fleet;
            best_value = score(fleet);
        }
        seed(fleet);

        // Greedy insertions quickly use spare time before considering two-site windows.
        for (int pass = 0; pass < p.n && !done(); ++pass) {
            const auto info = describe(fleet);
            Choice chosen;
            for (int a = 0; a < k && !done(); ++a) {
                auto candidate = window_search(info, a, false, 0, false, !has_warm);
                if (candidate.merit > chosen.merit && candidate.a >= 0)
                    chosen = std::move(candidate);
            }
            if (chosen.a < 0) break;
            apply(fleet, std::move(chosen), "surgery-fill");
        }
        publish_pending();

        for (std::uint64_t round = 0; !done(); ++round) {
            if (round % 12 == 0 || deadline - Clock::now() < std::chrono::milliseconds(120))
                publish_pending();
            if (round % 12 == 0) fleet = best;
            const double price = (round % 12 < 5 ? 0.014 : 0.06) * best_value;
            const auto info = describe(fleet);
            auto reconnecting = reconnect(info, price, true);
            if (reconnecting.a >= 0) {
                apply(fleet, std::move(reconnecting), "surgery-reconnect");
                continue;
            }
            Choice choice;
            for (int a = 0; a < k && !done(); ++a) {
                auto candidate = window_search(info, a, true, price, false);
                if (candidate.a >= 0 && candidate.merit > choice.merit)
                    choice = std::move(candidate);
            }
            if (choice.a >= 0) {
                apply(fleet, std::move(choice), "surgery-window");
                continue;
            }
            const int a = pick(k);
            auto shake = window_search(info, a, true, 2 * price, true);
            if (shake.a >= 0) apply(fleet, std::move(shake), "surgery-shake");
        }
        return error ? 3 : 0;
    }
};

} // namespace surgery

extern "C" int astra_surgery(int n, int k, unsigned seed, double seconds, double capacity,
                             double limit, const double* supply, const double* population,
                             const double* arcs, const int* packed, int count,
                             const double* visits, const double* flows, surgery::Publish publish) {
    if (n < 2 || n > 512 || k < 1 || count < 1 || !packed || !publish ||
        !(seconds > 0) || !(capacity > 0) || !std::isfinite(limit) || limit < 0) return 2;
    try {
        surgery::Problem problem(n, capacity, limit, supply, population, arcs, visits, flows);
        k = std::min({k, static_cast<int>(problem.suppliers.size()), n - static_cast<int>(problem.suppliers.size())});
        if (!k || packed[0] < 0 || packed[0] > k) return 2;
        surgery::Fleet warm(k);
        std::vector<bool> seen(n + 1);
        int offset = 1;
        for (int v = 0; v < packed[0]; ++v) {
            if (offset >= count) return 2;
            const int size = packed[offset++];
            if (size < 2 || size > count - offset) return 2;
            for (int i = 0; i < size; ++i) {
                const int node = packed[offset++];
                if (node < 1 || node > n || seen[node]) return 2;
                seen[node] = true;
                warm[v].push_back(node);
            }
            if (!problem.supplier(warm[v].front()) || problem.supplier(warm[v].back())) return 2;
        }
        if (offset != count) return 2;
        surgery::Search search(problem, k, seed, seconds, publish);
        return search.run(std::move(warm), packed[0] > 0);
    } catch (const std::exception&) {
        return 4;
    }
}
