#include "primal_profile.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <random>
#include <utility>

namespace levels {

constexpr int grids = 11;
constexpr double tolerance = 1e-12;
using Clock = std::chrono::steady_clock;
using Route = v3::Route;
using Fleet = std::vector<Route>;
using Quantities = std::array<double, grids>;
using Publish = int (*)(const char*, const int*, int, double);

// For an input load x, sent(x) = sent0 + min(x, low), while
// output(x) = out0 + clamp(x-low, 0, width).
struct Transfer {
    double out0, low, width, sent0;
};

inline Transfer compose(Transfer left, Transfer right) {
    const double offset = right.low - left.out0;
    const double lost = std::clamp(offset, 0.0, left.width);
    const double pass = std::clamp(offset + right.width, lost, left.width);
    return {right.out0 + std::clamp(-offset, 0.0, right.width),
            left.low + lost, pass - lost,
            left.sent0 + right.sent0 + std::min(left.out0, right.low)};
}

struct Piece {
    int first = 0, last = 0;
    double inner = 0;
    std::array<Transfer, grids> transfer{};
};

class Problem {
public:
    int n;
    double cap, limit, people, average_rate, maximum_rate;
    const double* arcs;
    const double* guide_visit;
    const double* guide_arc;
    std::vector<double> supply, population;
    std::vector<int> sources, targets;
    std::vector<std::vector<int>> incoming, outgoing;
    std::vector<Piece> single;
    Quantities weights{};

    Problem(int n_, double cap_, double limit_, const double* stock,
            const double* population_, const double* arcs_,
            const double* visits, const double* corridor)
        : n(n_), cap(cap_), limit(limit_), arcs(arcs_), guide_visit(visits), guide_arc(corridor),
          supply(stock, stock + n + 1), population(population_, population_ + n + 1),
          incoming(n + 1), outgoing(n + 1), single(n + 1) {
        people = std::accumulate(population.begin(), population.end(), 0.0);
        double smallest = std::numeric_limits<double>::infinity();
        for (int node = 1; node <= n; ++node) {
            if (is_source(node)) sources.push_back(node);
            else {
                targets.push_back(node);
                smallest = std::min(smallest, population[node]);
            }
        }
        average_rate = std::accumulate(supply.begin(), supply.end(), 0.0) / people;
        maximum_rate = cap / smallest;
        if (guide_arc) {
            for (int i = 0; i <= n; ++i) {
                for (int j = 1; j <= n; ++j) {
                    if (i == j) continue;
                    if (flow(i, j) > 0) outgoing[i].push_back(j);
                    if (flow(j, i) > 0) incoming[i].push_back(j);
                }
                auto trim = [](std::vector<int>& row, auto value) {
                    const auto limit = std::min<std::size_t>(8, row.size());
                    std::partial_sort(row.begin(), row.begin() + limit, row.end(), [&](int a, int b) {
                        const double x = value(a), y = value(b);
                        return x == y ? a < b : x > y;
                    });
                    row.resize(limit);
                };
                trim(outgoing[i], [&](int j) { return flow(i, j); });
                trim(incoming[i], [&](int j) { return flow(j, i); });
            }
        }
    }

    double edge(int i, int j) const { return arcs[i * (n + 1) + j]; }
    double flow(int i, int j) const { return guide_arc ? guide_arc[i * (n + 1) + j] : 0; }
    double visit(int i) const { return guide_visit ? guide_visit[i] : 0; }
    bool is_source(int node) const { return supply[node] > 0; }
    double time(const Piece& piece) const {
        return piece.first ? edge(0, piece.first) + piece.inner + edge(piece.last, 0) : 0;
    }
    bool endpoints(const Piece& piece) const {
        return !piece.first || (is_source(piece.first) && !is_source(piece.last));
    }

    void grid(double center, double phase) {
        constexpr Quantities ratios{0.27, 0.45, 0.60, 0.74, 0.87, 1.01,
                                    1.17, 1.37, 1.65, 2.15, 0};
        double earlier = 0;
        center = std::min(center * phase, maximum_rate / 2.35);
        for (int j = 0; j < grids; ++j) {
            const double rate = j == grids - 1 ? maximum_rate : ratios[j] * center;
            weights[j] = 1 / ((rate - earlier) * people);
            earlier = rate;
            for (int node = 1; node <= n; ++node) {
                auto& piece = single[node];
                piece.first = piece.last = node;
                const double take = std::min(cap, supply[node]);
                const double send = std::min(cap, rate * population[node]);
                piece.transfer[j] = {take, send, cap - take - send, 0};
            }
        }
    }

    Piece join(const Piece& a, const Piece& b) const {
        if (!a.first) return b;
        if (!b.first) return a;
        Piece result;
        result.first = a.first;
        result.last = b.last;
        result.inner = a.inner + edge(a.last, b.first) + b.inner;
        for (int j = 0; j < grids; ++j)
            result.transfer[j] = compose(a.transfer[j], b.transfer[j]);
        return result;
    }

    double value(const Quantities& cumulative) const {
        double result = 0, previous = 0;
        for (int j = 0; j < grids; ++j) {
            const double increment = std::max(0.0, cumulative[j] - previous);
            result += (weights[j] * increment) * increment;
            previous = cumulative[j];
        }
        return result;
    }

    double excess(double t) const {
        const double gap = std::max(0.0, t - limit);
        return gap * (1 + 3 * gap);
    }
};

class CachedRoute {
    struct Cell { Piece forward, reverse; };
    const Problem& problem;
    int base = 1;
    std::vector<Cell> tree;
    std::vector<Piece> prefix, suffix;

public:
    Route sites;
    double time = 0;

    explicit CachedRoute(const Problem& p) : problem(p) { reset({}); }
    int size() const { return static_cast<int>(sites.size()); }
    const Piece& whole() const { return prefix.back(); }

    void reset(Route route) {
        sites = std::move(route);
        const int n = size();
        base = 1;
        while (base < n) base *= 2;
        tree.resize(2 * base);
        prefix.resize(n + 1);
        suffix.resize(n + 1);
        prefix.front() = suffix.back() = {};
        for (int i = 0; i < base; ++i) {
            const auto& piece = i < n ? problem.single[sites[i]] : problem.single[0];
            tree[base + i] = {piece, piece};
        }
        for (int i = 0; i < n; ++i) {
            prefix[i + 1] = problem.join(prefix[i], tree[base + i].forward);
            suffix[n - i - 1] = problem.join(tree[base + n - i - 1].forward, suffix[n - i]);
        }
        for (int i = base - 1; i; --i) {
            const auto& l = tree[2 * i];
            const auto& r = tree[2 * i + 1];
            tree[i] = {problem.join(l.forward, r.forward), problem.join(r.reverse, l.reverse)};
        }
        time = problem.time(whole());
    }

    Piece range(int first, int last, bool reverse) const {
        if (!reverse) {
            if (first == 0) return prefix[last];
            if (last == size()) return suffix[first];
        }
        Piece l, r;
        for (first += base, last += base; first < last; first /= 2, last /= 2) {
            if (first & 1) {
                const auto& cell = tree[first++];
                l = reverse ? problem.join(cell.reverse, l) : problem.join(l, cell.forward);
            }
            if (last & 1) {
                const auto& cell = tree[--last];
                r = reverse ? problem.join(r, cell.reverse) : problem.join(cell.forward, r);
            }
        }
        return reverse ? problem.join(r, l) : problem.join(l, r);
    }
};

struct Part {
    int slot, first, last;
    bool reverse;
};

struct Patch {
    std::array<Part, 6> parts;
    int size = 0;
    void slice(int source, int first, int last, bool reverse = false) {
        if (first < last) parts[size++] = {source, first, last, reverse};
    }
    void site(int node) { parts[size++] = {-1, node, 0, false}; }
};

struct Move {
    int a, b;
    Patch left, right;
};

struct Evaluation {
    Quantities cumulative{};
    double guide = 0, violation = 0, clock = 0;
    bool legal = true, feasible = true;
};

class State {
    const Problem& problem;

    Piece render(const Patch& patch) const {
        Piece result;
        for (int i = 0; i < patch.size; ++i) {
            const auto& part = patch.parts[i];
            result = problem.join(result, part.slot < 0 ? problem.single[part.first]
                : routes[part.slot].range(part.first, part.last, part.reverse));
        }
        return result;
    }

    void account(Evaluation& out, int slot, const Patch& patch) const {
        const auto changed = render(patch);
        const double new_time = problem.time(changed), old_time = routes[slot].time;
        out.legal &= problem.endpoints(changed) &&
                     new_time <= (problem.limit > 0 ? 1.43 * problem.limit : 0) + tolerance;
        for (int j = 0; j < grids; ++j)
            out.cumulative[j] += changed.transfer[j].sent0 - routes[slot].whole().transfer[j].sent0;
        out.violation += problem.excess(new_time) - problem.excess(old_time);
        out.clock += new_time - old_time;
    }

public:
    std::vector<CachedRoute> routes;
    std::vector<int> used;
    Evaluation current;

    State(const Problem& problem_, int k) : problem(problem_), used(problem.n + 1, -1) {
        routes.reserve(k);
        for (int i = 0; i < k; ++i) routes.emplace_back(problem);
    }

    void load(const Fleet& fleet) {
        for (int v = 0; v < static_cast<int>(routes.size()); ++v) routes[v].reset(fleet[v]);
        refresh();
    }

    void refresh() {
        // Recompute the totals after accepted edits to avoid accumulated error.
        current = {};
        std::fill(used.begin(), used.end(), -1);
        for (int v = 0; v < static_cast<int>(routes.size()); ++v) {
            const auto& route = routes[v];
            for (int node : route.sites) used[node] = v;
            for (int j = 0; j < grids; ++j) current.cumulative[j] += route.whole().transfer[j].sent0;
            current.clock += route.time;
            current.violation += problem.excess(route.time);
        }
        current.guide = problem.value(current.cumulative);
        current.feasible = current.violation <= tolerance;
    }

    Evaluation score(const Move& move) const {
        auto result = current;
        account(result, move.a, move.left);
        if (move.b >= 0) account(result, move.b, move.right);
        result.feasible = result.violation <= tolerance;
        result.guide = problem.value(result.cumulative);
        return result;
    }

    Route render_sites(const Patch& patch) const {
        Route result;
        for (int i = 0; i < patch.size; ++i) {
            const auto& part = patch.parts[i];
            if (part.slot < 0) { result.push_back(part.first); continue; }
            const auto& route = routes[part.slot].sites;
            if (part.reverse) {
                for (int j = part.last; j-- > part.first;) result.push_back(route[j]);
            } else result.insert(result.end(), route.begin() + part.first, route.begin() + part.last);
        }
        return result;
    }

    Fleet copy() const {
        Fleet result;
        result.reserve(routes.size());
        for (const auto& route : routes) result.push_back(route.sites);
        return result;
    }

    void apply(const Move& move) {
        auto left = render_sites(move.left);
        auto right = render_sites(move.right);
        routes[move.a].reset(std::move(left));
        if (move.b >= 0) routes[move.b].reset(std::move(right));
        refresh();
    }
};

class Search {
    Problem& problem;
    int k;
    Clock::time_point deadline;
    std::mt19937 rng;
    Publish publish;
    v3::Profile water;
    Fleet best;
    double best_value = 0;
    bool failure = false;

    int pick(int n) { return static_cast<int>(rng() % static_cast<unsigned>(n)); }
    double uniform() { return std::generate_canonical<double, 32>(rng); }
    bool done() const { return failure || Clock::now() >= deadline; }

    double water_score(const Fleet& fleet) {
        v3::Levels all, one;
        for (const auto& route : fleet) {
            water(route, one);
            all.insert(all.end(), one.begin(), one.end());
        }
        std::sort(all.begin(), all.end(), v3::rate_order);
        return v3::lorentz(all, problem.people);
    }

    double center_rate() {
        double mass = 0, sent = 0;
        v3::Levels profile;
        for (const auto& route : best) {
            water(route, profile);
            for (auto level : profile) {
                mass += level.people;
                sent += level.people * level.rate;
            }
        }
        return mass > 0 ? sent / mass : problem.average_rate;
    }

    bool record(const Fleet& candidate, const char* label) {
        if (done()) return false;
        const double worth = water_score(candidate);
        if (worth <= best_value + 2e-13 * best_value) return false;
        std::vector<int> message{0};
        for (const auto& route : candidate) {
            if (route.empty()) continue;
            ++message[0];
            message.push_back(static_cast<int>(route.size()));
            message.insert(message.end(), route.begin(), route.end());
        }
        const int accepted = publish(label, message.data(), static_cast<int>(message.size()), worth);
        failure = accepted < 0;
        if (accepted <= 0) return false;
        best_value = worth;
        best = candidate;
        return true;
    }

    Route directed_seed(const std::vector<int>& used) const {
        int n = problem.n;
        const double inf = std::numeric_limits<double>::infinity();
        std::vector<double> distance(n + 1, inf);
        std::vector<int> previous(n + 1);
        std::vector<bool> settled(n + 1);
        for (int site : problem.sources)
            if (used[site] < 0) distance[site] = problem.edge(0, site);
        double length = problem.limit + tolerance;
        int last = 0;
        while (!done()) {
            int node = 0;
            for (int i = 1; i <= n; ++i)
                if (!settled[i] && distance[i] < distance[node]) node = i;
            if (!node || distance[node] > length) break;
            settled[node] = true;
            if (!problem.is_source(node) && distance[node] + problem.edge(node, 0) < length) {
                length = distance[node] + problem.edge(node, 0);
                last = node;
            }
            for (int target = 1; target <= n; ++target) {
                if (used[target] >= 0 || settled[target]) continue;
                const double finish = distance[node] + problem.edge(node, target);
                if (finish < distance[target] && finish <= length) {
                    previous[target] = node;
                    distance[target] = finish;
                }
            }
        }
        Route route;
        for (; last; last = previous[last]) route.push_back(last);
        std::reverse(route.begin(), route.end());
        return route;
    }

    Fleet seed_fleet(bool diversify) {
        Fleet fleet(k);
        std::vector<int> used(problem.n + 1, -1);
        std::vector<double> preference(problem.n + 1, 1);
        if (diversify)
            for (int i = 1; i <= problem.n; ++i) preference[i] = 0.58 + 0.84 * uniform();
        double occupied_population = 0;
        for (int slot = 0; slot < k && !done(); ++slot) {
            double key = 0;
            int source = 0, target = 0;
            for (int a : problem.sources) {
                if (used[a] >= 0) continue;
                for (int b : problem.targets) {
                    if (used[b] >= 0) continue;
                    const double cost = problem.edge(0, a) + problem.edge(a, b) + problem.edge(b, 0);
                    if (cost > problem.limit + tolerance) continue;
                    const double hint = problem.guide_visit
                        ? 1 + 0.18 * (problem.visit(a) + problem.visit(b)) + 0.7 * problem.flow(a, b)
                        : 1;
                    const double value = problem.supply[a] * (problem.population[b] + occupied_population) *
                        preference[a] * preference[b] * hint / (1 + 3 * cost);
                    if (value > key) {
                        source = a;
                        target = b;
                        key = value;
                    }
                }
            }
            auto& route = fleet[slot];
            route = key > 0 ? Route{source, target} : directed_seed(used);
            if (route.empty()) break;
            for (int site : route) {
                used[site] = slot;
                occupied_population += problem.population[site];
            }
        }
        return fleet;
    }

    Move replace(const State& state, int slot, int index, int node) const {
        Move move{slot, -1, {}, {}};
        move.left.slice(slot, 0, index);
        move.left.site(node);
        move.left.slice(slot, index + 1, state.routes[slot].size());
        return move;
    }

    Move insert(const State& state, int slot, int index, int node) const {
        Move move{slot, -1, {}, {}};
        move.left.slice(slot, 0, index);
        move.left.site(node);
        move.left.slice(slot, index, state.routes[slot].size());
        return move;
    }

    void fill(State& state) {
        // Greedily use spare time before the stochastic search begins.
        for (int count = 0; count < problem.n && !done(); ++count) {
            Move best_move{};
            double best_gain = 0;
            for (int a = 0; a < k && !done(); ++a) {
                const int size = state.routes[a].size();
                if (!size) continue;
                for (int node = 1; node <= problem.n; ++node) {
                    if (state.used[node] >= 0) continue;
                    for (int index = 0; index <= size; ++index) {
                        if ((index == 0 && !problem.is_source(node)) ||
                            (index == size && problem.is_source(node))) continue;
                        const auto move = insert(state, a, index, node);
                        const auto trial = state.score(move);
                        if (!trial.legal || !trial.feasible) continue;
                        const double gain = (trial.guide - state.current.guide) /
                                            (1 + 3 * std::max(0.0, trial.clock - state.current.clock));
                        if (gain > best_gain) { best_move = move; best_gain = gain; }
                    }
                }
            }
            if (best_gain <= 1e-13 * state.current.guide || done()) break;
            state.apply(best_move);
            if (count % 6 == 5) record(state.copy(), "levels-insert");
        }
        record(state.copy(), "levels-insert");
    }

    int guided_site(const State& state, int previous, int next, int sample,
                    int preserve_role) {
        double best_hint = -1;
        int selected = sample;
        auto check = [&](int site) {
            if (state.used[site] >= 0 || (preserve_role >= 0 &&
                problem.is_source(site) != problem.is_source(preserve_role))) return;
            const double hint = problem.flow(previous, site) + problem.flow(site, next) +
                                0.15 * problem.visit(site) + 0.035 * uniform();
            if (hint > best_hint) { best_hint = hint; selected = site; }
        };
        check(sample);
        for (int site : problem.outgoing[previous]) check(site);
        for (int site : problem.incoming[next]) check(site);
        for (int i = 0; i < 3; ++i) check(1 + pick(problem.n));
        return selected;
    }

    bool propose(const State& state, Move& move, bool guided) {
        const int a = pick(k);
        const auto& route = state.routes[a];
        const int na = route.size();
        move = {a, -1, {}, {}};
        if (!na) {
            const int s = problem.sources[pick(static_cast<int>(problem.sources.size()))];
            const int d = problem.targets[pick(static_cast<int>(problem.targets.size()))];
            if (state.used[s] >= 0 || state.used[d] >= 0) return false;
            move.left.site(s);
            move.left.site(d);
            return true;
        }
        const int op = pick(16);
        const int i = pick(na);
        if (op < 4) {
            int node = 1 + pick(problem.n);
            if (op < 2) {
                const auto& role = problem.is_source(route.sites[i]) ? problem.sources : problem.targets;
                node = role[pick(static_cast<int>(role.size()))];
            }
            const int position = op < 2 ? i : pick(na + 1);
            if (guided && pick(3) == 0) {
                const int previous = position ? route.sites[position - 1] : 0;
                const int after = position + (op < 2);
                const int next = after < na ? route.sites[after] : 0;
                node = guided_site(state, previous, next, node, op < 2 ? route.sites[i] : -1);
            }
            if (state.used[node] >= 0) return false;
            move = op < 2 ? replace(state, a, i, node) : insert(state, a, position, node);
        } else if (op == 4 || op == 5) {
            const int end = std::min(na, i + (op == 4 ? 1 : 2 + pick(3)));
            move.left.slice(a, 0, i);
            move.left.slice(a, end, na);
        } else if (op < 9 || k == 1) {
            const int j = pick(na);
            if (i == j) return false;
            const int low = std::min(i, j), high = std::max(i, j);
            move.left.slice(a, 0, low);
            if (op == 6 || (op >= 9 && (op & 1))) {
                move.left.slice(a, low, high + 1, true);
            } else if (op == 7 || op >= 9) {
                if (i == low) {
                    move.left.slice(a, low + 1, high + 1);
                    move.left.slice(a, low, low + 1);
                } else {
                    move.left.slice(a, high, high + 1);
                    move.left.slice(a, low, high);
                }
            } else {
                move.left.slice(a, high, high + 1);
                move.left.slice(a, low + 1, high);
                move.left.slice(a, low, low + 1);
            }
            move.left.slice(a, high + 1, na);
        } else {
            const int b = (a + 1 + pick(k - 1)) % k;
            const int nb = state.routes[b].size();
            int j = nb ? pick(nb) : 0;
            if (nb && guided && pick(4) == 0) {
                const int left_a = i ? route.sites[i - 1] : 0;
                const auto& other = state.routes[b].sites;
                auto desirability = [&](int cut) {
                    const int left_b = cut ? other[cut - 1] : 0;
                    return problem.flow(left_a, other[cut]) + problem.flow(left_b, route.sites[i]);
                };
                for (int t = 0; t < 3; ++t) {
                    const int cut = pick(nb);
                    if (desirability(cut) > desirability(j)) j = cut;
                }
            }
            const int end_a = op == 14 ? na : std::min(na, i + (op < 12 ? 1 : 1 + pick(5)));
            const int end_b = op == 14 ? nb : std::min(nb, j + (op < 12 ? 1 : 1 + pick(5)));
            const bool relocate = op == 10 || op == 12 || !nb;
            move.b = b;
            move.left.slice(a, 0, i);
            if (!relocate) move.left.slice(b, j, end_b, op == 15);
            move.left.slice(a, end_a, na);
            move.right.slice(b, 0, j);
            move.right.slice(a, i, end_a, op == 15);
            move.right.slice(b, relocate ? j : end_b, nb);
        }
        return true;
    }

    void polish(State& state, double scale) {
        // Preserve the best exact score while considering near ties in the
        // rate grid. Insertion and same-role replacement are exhaustive.
        for (int sweep = 0; sweep < 2 && !done(); ++sweep) {
            Move selected{};
            double peak = best_value;
            bool improvement = false;
            for (int a = 0; a < k && !done(); ++a) {
                const auto& route = state.routes[a];
                for (int i = 0; i <= route.size() && !done(); ++i) {
                    for (int node = 1; node <= problem.n; ++node) {
                        if (state.used[node] >= 0) continue;
                        for (int change = 0; change < 2; ++change) {
                            if (change && (i == route.size() ||
                                problem.is_source(route.sites[i]) != problem.is_source(node))) continue;
                            const auto move = change ? replace(state, a, i, node) : insert(state, a, i, node);
                            const auto result = state.score(move);
                            if (!result.legal || !result.feasible ||
                                result.guide < state.current.guide - 0.013 * scale) continue;
                            auto fleet = state.copy();
                            fleet[a] = state.render_sites(move.left);
                            const double score = water_score(fleet);
                            if (score > peak + 2e-13 * peak) {
                                peak = score;
                                selected = move;
                                improvement = true;
                            }
                        }
                    }
                }
            }
            if (!improvement || done()) return;
            auto fleet = state.copy();
            fleet[selected.a] = state.render_sites(selected.left);
            if (!record(fleet, "levels-polish")) return;
            state.apply(selected);
        }
    }

public:
    Search(Problem& p, int vehicles, unsigned seed, double budget, Publish send)
        : problem(p), k(vehicles), deadline(Clock::now() +
            std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(budget))),
          rng(seed), publish(send), water(p.supply, p.population, p.cap), best(k) {}

    int run(Fleet warm, bool has_warm) {
        if (has_warm) {
            best = warm;
            best_value = water_score(warm);
        } else {
            warm = seed_fleet(false);
            record(warm, "levels-seed");
        }
        problem.grid(center_rate(), 1);
        State state(problem, k);
        state.load(warm);
        fill(state);
        const int length = 30'000 + 420 * problem.n;
        constexpr std::array<double, 4> phases{1, 0.89, 1.13, 0.96};
        double penalty = 0.27;
        for (std::uint64_t epoch = 0; !done(); ++epoch) {
            if (epoch) {
                problem.grid(center_rate(), phases[epoch % phases.size()]);
                state.load(best);
                if (epoch % 4 == 3) polish(state, std::max(best_value, 1e-30));
                if (epoch % 9 == 7) {
                    state.load(seed_fleet(true));
                    fill(state);
                }
            }
            const double scale = std::max({best_value, state.current.guide, 1e-150});
            const double certified_guide = state.current.guide;
            double largest_checked_guide = certified_guide;
            int eligible = 0, sampled = 0;
            const bool guided = problem.guide_visit && epoch % 4 != 1;
            for (int step = 0; step < length; ++step) {
                if ((step & 127) == 0 && done()) return failure ? 3 : 0;
                Move move{};
                if (!propose(state, move, guided)) continue;
                const auto trial = state.score(move);
                if (!trial.legal) continue;
                eligible += trial.feasible;
                if (++sampled == 256) {
                    if (eligible < 42) penalty = std::min(4.0, penalty * 1.11);
                    if (eligible > 165) penalty = std::max(0.045, penalty * 0.94);
                    sampled = eligible = 0;
                }
                const double remaining = 1 - static_cast<double>(step) / length;
                const double temp = 0.00055 + (epoch % 3 == 0 ? 0.014 : 0.007) * remaining * remaining;
                const double delta = (trial.guide - state.current.guide) / scale
                    - penalty * (trial.violation - state.current.violation)
                    - 0.0020 * (trial.clock - state.current.clock);
                const bool take = delta > 0 || uniform() < std::exp(delta / temp);
                bool improved = false;
                if (trial.feasible && trial.guide >= 0.935 * certified_guide &&
                    (trial.guide > largest_checked_guide + 1e-10 * scale || (take && (step & 15) == 0))) {
                    largest_checked_guide = std::max(largest_checked_guide, trial.guide);
                    auto fleet = state.copy();
                    fleet[move.a] = state.render_sites(move.left);
                    if (move.b >= 0) fleet[move.b] = state.render_sites(move.right);
                    improved = record(fleet, move.b < 0 ? "levels-local" : "levels-cross");
                }
                if (take || improved) state.apply(move);
            }
        }
        return failure ? 3 : 0;
    }
};

}  // namespace levels

extern "C" int astra_levels(int n, int k, unsigned seed, double seconds, double capacity,
                            double limit, const double* supply, const double* population,
                            const double* arcs, const int* packed, int count,
                            const double* visits, const double* corridor, levels::Publish publish) {
    if (n < 2 || n > 512 || k < 1 || count < 1 || !packed || !publish ||
        !(seconds > 0) || !(capacity > 0) || !std::isfinite(limit) || limit < 0) return 2;
    try {
        levels::Problem instance(n, capacity, limit, supply, population, arcs, visits, corridor);
        if (instance.sources.empty() || instance.targets.empty()) return 2;
        k = std::min({k, static_cast<int>(instance.sources.size()), static_cast<int>(instance.targets.size())});
        levels::Fleet warm(k);
        if (packed[0] < 0 || packed[0] > k) return 2;
        std::vector<bool> used(n + 1);
        int offset = 1;
        for (int v = 0; v < packed[0]; ++v) {
            if (offset >= count) return 2;
            const int size = packed[offset++];
            if (size < 2 || size > count - offset) return 2;
            for (int i = 0; i < size; ++i) {
                const int node = packed[offset++];
                if (node < 1 || node > n || used[node]) return 2;
                used[node] = true;
                warm[v].push_back(node);
            }
            if (!instance.is_source(warm[v].front()) || instance.is_source(warm[v].back())) return 2;
        }
        if (offset != count) return 2;
        levels::Search search(instance, k, seed, seconds, publish);
        return search.run(std::move(warm), packed[0] > 0);
    } catch (const std::exception&) {
        return 4;
    }
}
