// Elementary directed prefix search. Walk tables are immutable and owned by Python.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <queue>
#include <vector>

using Int = std::int64_t;
using Wide = __int128_t;
using WalkValue = int (*)(void*, int, int, int, int);
using Witness = void (*)(int, const std::uint8_t*, Int, Int*);
using Checkpoint = void (*)(Int, const Int*, int*);

struct Walk {
    void* handle;
    const int* prices;
    const int* bits;
    int full;
    Int supply_weight, population_weight, unit;
};

namespace {
constexpr int max_nodes = 111;
constexpr int impossible_walk = -1000000000;
constexpr Int impossible = -1;
using Clock = std::chrono::steady_clock;
enum Stat {
    expanded, generated, time_pruned, bound_pruned, bounds_called, walk_calls,
    walk_tighter, allocations, max_heap, max_dfs, dfs_switches, stat_count
};

struct Node {
    Int unloaded = 0, onboard = 0, population = 0, ub = 0;
    std::array<std::uint64_t, 2> visited{};
    int length = 0, time = 0;
    std::array<std::uint8_t, max_nodes> route{};

    int last() const { return length ? route[length - 1] : 0; }
    int previous() const { return length > 1 ? route[length - 2] : 0; }
    bool has(int v) const { return (visited[v / 64] >> (v % 64)) & 1; }
    void add(int v, int dt, Int supply, Int pop, Int capacity) {
        visited[v / 64] |= std::uint64_t(1) << (v % 64);
        route[length++] = static_cast<std::uint8_t>(v);
        time += dt;
        if (supply) {
            onboard = std::min(capacity, onboard + supply);
        } else {
            unloaded += onboard;
            onboard = 0;
            population += pop;
        }
    }
};

struct Context {
    int n, limit;
    Int capacity, prior;
    std::vector<int> cost, dist, out, in, charge;
    std::vector<Int> supply, people;
    std::vector<int> paid_supply, paid_population, largest_supply, largest_population;
    std::vector<Walk> tables;
    WalkValue walk_value;
    bool fairness;

    Context(int nodes, int horizon, Int cap, Int upper, const int* times,
            const Int* supplies, const Int* populations, const Walk* walks,
            int nt, WalkValue value, bool box)
        : n(nodes), limit(horizon), capacity(cap), prior(upper),
          cost(times, times + n*n), dist(cost), out(n), in(n), charge(n),
          supply(supplies, supplies+n), people(populations, populations+n),
          walk_value(value), fairness(box) {
        if (nt) tables.assign(walks, walks+nt);
        for (int i = 0; i < n; ++i) dist[i*n+i] = 0;
        for (int k = 0; k < n; ++k)
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    dist[i*n+j] = std::min(dist[i*n+j], dist[i*n+k] + dist[k*n+j]);

        // Every original arc pays these two nonnegative endpoint charges.
        for (int i = 0; i < n; ++i) {
            out[i] = limit + 1;
            for (int j = 0; j < n; ++j)
                if (i != j) out[i] = std::min(out[i], cost[i*n+j]);
        }
        for (int j = 0; j < n; ++j) {
            in[j] = limit + 1;
            for (int i = 0; i < n; ++i)
                if (i != j) in[j] = std::min(in[j], cost[i*n+j] - out[i]);
            charge[j] = out[j] + in[j];
        }
        for (int i = 1; i < n; ++i) {
            if (supply[i]) {
                paid_supply.push_back(i);
                largest_supply.push_back(i);
            } else {
                paid_population.push_back(i);
                largest_population.push_back(i);
            }
        }
        auto cheaper = [&](int a, int b) { return charge[a] < charge[b]; };
        std::sort(paid_supply.begin(), paid_supply.end(), cheaper);
        std::sort(paid_population.begin(), paid_population.end(), cheaper);
        std::sort(largest_supply.begin(), largest_supply.end(),
                  [&](int a, int b) { return supply[a] > supply[b]; });
        std::sort(largest_population.begin(), largest_population.end(),
                  [&](int a, int b) { return people[a] > people[b]; });
    }

    bool closes(const Node& node) const {
        return people[node.last()] && node.time + cost[node.last()*n] <= limit;
    }

    static Int apply_walk(Int upper, Int total, Int a, Int b, Int flow, Int pop) {
        // The real total is on or below a*flow + b*pop = total.
        Wide t = total;
        upper = static_cast<Int>(std::min<Wide>(upper, t*t / (4*Wide(a)*b)));
        if (2*Wide(a)*flow <= t)
            upper = static_cast<Int>(std::min<Wide>(upper, Wide(flow)*(t-Wide(a)*flow)/b));
        if (2*Wide(b)*pop <= t)
            upper = static_cast<Int>(std::min<Wide>(upper, Wide(pop)*(t-Wide(b)*pop)/a));
        return upper;
    }

    Int box(const Node& node, const std::array<bool,max_nodes>& allowed,
            int adds, Int flow, Int population, Int upper) const {
        // This is the fixed-capacity fairness truncation for the largest
        // feasible populations. Skip it when even the largest cannot saturate.
        Wide fixed = 0, tail = population;
        int h = 0;
        for (int v : largest_population) {
            if (!node.has(v)) {
                if (!adds || !allowed[v]) continue;
                --adds;
            }
            Int p = people[v];
            if (Wide(p) * (flow - h*Wide(capacity)) <= Wide(capacity)*tail) break;
            tail -= p;
            ++h;
            fixed += Wide(capacity) * (2*h-1) * p;
            upper = static_cast<Int>(std::min<Wide>(upper, fixed+tail*(flow+h*Wide(capacity))));
        }
        return upper;
    }

    Int bound(const Node& node, bool weighted, Int cutoff, std::array<Int,stat_count>& stats) const {
        ++stats[bounds_called];
        int last = node.last(), remaining = limit - node.time;
        if (remaining < dist[last*n]) return impossible;

        std::array<bool,max_nodes> allowed{};
        std::array<int,2> masks{}, prices{};
        Int supply_cap = node.unloaded + node.onboard;
        Int pop_cap = node.population;
        int suppliers = 0, agencies = 0;
        const size_t active = weighted ? tables.size() : 0;
        for (int v = 1; v < n; ++v) {
            if (node.has(v) || dist[last*n+v] + dist[v*n] > remaining) continue;
            allowed[v] = true;
            supply_cap += supply[v];
            pop_cap += people[v];
            suppliers += supply[v] > 0;
            agencies += people[v] > 0;
            for (size_t t = 0; t < active; ++t) {
                masks[t] |= tables[t].bits[v];
                prices[t] += tables[t].prices[v];
            }
        }
        Int best = closes(node) ? node.unloaded * node.population : impossible;
        if (!agencies) return std::min(node.ub, best);
        Int flow_cap = std::min(supply_cap, node.unloaded + agencies*capacity);
        Int rough = std::min(node.ub, flow_cap*pop_cap);
        if (rough <= cutoff) return rough;

        std::array<Int,2> weighted_totals{};
        for (size_t t = 0; t < active; ++t) {
            const Walk& walk = tables[t];
            ++stats[walk_calls];
            int value = walk_value(walk.handle, masks[t] & walk.full, remaining, last, node.previous());
            if (value == impossible_walk) return impossible;
            Wide total = Wide(walk.supply_weight)*(node.unloaded+node.onboard)
                       + Wide(walk.population_weight)*node.population
                       + Wide(walk.unit)*(value+prices[t]);
            if (total < 0) return impossible;
            weighted_totals[t] = static_cast<Int>(total);
            Int tightened = apply_walk(rough, weighted_totals[t], walk.supply_weight,
                                       walk.population_weight, flow_cap, pop_cap);
            if (tightened < rough) ++stats[walk_tighter];
            rough = tightened;
        }
        if (rough <= cutoff) return rough;
        best = std::min(rough, best);
        int budget = remaining - out[last] - in[0];
        if (budget < 0) return best;

        std::array<int,max_nodes> ps_cost{}, ds_cost{};
        std::array<Int,max_nodes> ps_size{}, ds_size{};
        int count = 0;
        for (int v : paid_supply)
            if (allowed[v]) { ps_cost[count+1] = ps_cost[count] + charge[v]; ++count; }
        count = 0;
        for (int v : paid_population)
            if (allowed[v]) { ds_cost[count+1] = ds_cost[count] + charge[v]; ++count; }
        count = 0;
        for (int v : largest_supply)
            if (allowed[v]) { ps_size[count+1] = ps_size[count] + supply[v]; ++count; }
        count = 0;
        for (int v : largest_population)
            if (allowed[v]) { ds_size[count+1] = ds_size[count] + people[v]; ++count; }
        int a_count = suppliers;
        for (int b_count = 1; b_count <= agencies; ++b_count) {
            if (ds_cost[b_count] > budget) break;
            while (a_count && ps_cost[a_count]+ds_cost[b_count] > budget) --a_count;
            Int flow = std::min(node.unloaded + node.onboard + ps_size[a_count],
                                node.unloaded + b_count*capacity);
            Int population = node.population + ds_size[b_count];
            Int upper = std::min(rough, flow*population);
            for (size_t t = 0; t < active && upper > best; ++t)
                upper = apply_walk(upper, weighted_totals[t], tables[t].supply_weight,
                                   tables[t].population_weight, flow, population);
            if (fairness && upper > best && flow > capacity)
                upper = box(node, allowed, b_count, flow, population, upper);
            best = std::max(best, upper);
        }
        return best;
    }
};

struct Worse {
    bool operator()(const Node& a, const Node& b) const {
        if (a.ub != b.ub) return a.ub < b.ub;
        if (a.length != b.length) return a.length < b.length;
        Int potential_a = a.unloaded*a.population, potential_b = b.unloaded*b.population;
        return potential_a < potential_b;
    }
};

struct Deferred {
    Node node;
    Int max;
};

struct Search {
    const Context& cx;
    bool per_prefix;
    std::size_t heap_limit;
    Witness witness;
    Checkpoint checkpoint;
    Int lb;
    Clock::time_point end;
    std::priority_queue<Node, std::vector<Node>, Worse> heap;
    std::vector<Deferred> dfs;
    std::array<Int,stat_count>& stats;

    Search(const Context& ctx, bool every, std::size_t cap, Int incumbent,
           double seconds, Witness offer, Checkpoint check, std::array<Int,stat_count>& counts)
        : cx(ctx), per_prefix(every), heap_limit(cap), witness(offer), checkpoint(check), lb(incumbent),
          end(Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds))),
          stats(counts) {}

    Int global(Int parent = impossible) const {
        return std::max({lb, parent, heap.empty() ? impossible : heap.top().ub,
                         dfs.empty() ? impossible : dfs.back().max});
    }

    bool expired() const { return Clock::now() >= end; }

    bool visit_children(const Node& node, std::vector<Node>& children) {
        int last = node.last();
        for (int v = 1; v < cx.n; ++v) {
            if (expired()) return false;
            if (node.has(v) || (!last && !cx.supply[v])) continue;
            int dt = cx.cost[last*cx.n+v];
            if (dt+node.time+cx.dist[v*cx.n] > cx.limit) { ++stats[time_pruned]; continue; }
            Node child = node;
            child.add(v, dt, cx.supply[v], cx.people[v], cx.capacity);
            ++stats[generated];
            child.ub = cx.bound(child, per_prefix || !last, lb, stats);
            if (child.ub <= lb) { ++stats[bound_pruned]; continue; }
            children.push_back(child);
        }
        return true;
    }

    void add_children(std::vector<Node>& children, bool stack) {
        if (stack || heap.size()+children.size() > heap_limit) {
            if (!stack) ++stats[dfs_switches];
            std::sort(children.begin(), children.end(), Worse{});
            Int max = dfs.empty() ? impossible : dfs.back().max;
            for (const Node& child : children) {
                max = std::max(max, child.ub);
                dfs.push_back({child, max});
            }
        } else {
            for (const Node& child : children) heap.push(child);
        }
        stats[max_heap] = std::max<Int>(stats[max_heap], heap.size());
        stats[max_dfs] = std::max<Int>(stats[max_dfs], dfs.size());
    }

    bool report(Int upper) const {
        int proceed = 0;
        checkpoint(upper, stats.data(), &proceed);
        return proceed == 1;
    }

    int run(Int& upper) {
        Node root;
        root.ub = cx.prior;
        upper = global(root.ub);
        std::vector<Node> children;
        children.reserve(cx.n);
        if (!visit_children(root, children)) return 0;
        add_children(children, false);
        upper = global();
        if (!report(upper)) return 0;
        auto next_report = Clock::now() + std::chrono::milliseconds(500);
        while (upper > lb) {
            if (expired()) return 0;
            bool stack = !dfs.empty();
            Node node = stack ? dfs.back().node : heap.top();
            if (stack) dfs.pop_back(); else heap.pop();
            ++stats[expanded];
            if (cx.closes(node) && node.unloaded*node.population > lb && node.ub > lb) {
                ++stats[allocations];
                Int value = -1;
                witness(node.length, node.route.data(), node.ub, &value);
                if (value < 0) { upper = global(node.ub); return 0; }
                if (value > lb) {
                    lb = value;
                    upper = global(node.ub);
                    if (!report(upper)) return 0;
                }
            }
            if (node.ub > lb) {
                children.clear();
                if (!visit_children(node, children)) { upper = global(node.ub); return 0; }
                add_children(children, stack);
            }
            upper = global();
            if (Clock::now() >= next_report) {
                if (!report(upper)) return 0;
                next_report = Clock::now() + std::chrono::milliseconds(500);
            }
        }
        return 1;
    }
};
} // namespace

extern "C" {
void* hpdp_weighted_create(int n, int limit, Int cap, Int prior, const int* times,
                           const Int* supply, const Int* pop, const Walk* table, int count,
                           WalkValue value, int fairness) {
    if (n < 3 || n > max_nodes || limit < 1 || limit > 1600 || count < 0 || count > 2
        || (count && !value)) return nullptr;
    try {
        return new Context(n, limit, cap, prior, times, supply, pop, table, count, value, fairness != 0);
    } catch (const std::bad_alloc&) { return nullptr; }
}

void hpdp_weighted_free(void* handle) {
    delete static_cast<Context*>(handle);
}

Int hpdp_weighted_bound(void* handle, int length, const std::uint8_t* route) {
    const auto& cx = *static_cast<Context*>(handle);
    if (length < 1 || length >= cx.n) return impossible;
    Node node;
    node.ub = cx.prior;
    for (int i = 0; i < length; ++i) {
        int v = route[i];
        if (v < 1 || v >= cx.n || node.has(v) || (!i && !cx.supply[v])) return impossible;
        node.add(v, cx.cost[node.last()*cx.n+v], cx.supply[v], cx.people[v], cx.capacity);
    }
    std::array<Int,stat_count> stats{};
    return cx.bound(node, true, impossible, stats);
}

int hpdp_weighted_search_v2(void* handle, int per_prefix, int heap_limit, Int incumbent,
                         double seconds, Witness witness, Checkpoint checkpoint,
                         Int* upper, Int* counters) {
    const auto& context = *static_cast<Context*>(handle);
    std::array<Int,stat_count> stats{};
    if (heap_limit < 1 || !(seconds >= 0.0) || !witness || !checkpoint) return -1;
    try {
        Search search(context, per_prefix != 0, heap_limit, incumbent, seconds,
                      witness, checkpoint, stats);
        int status = search.run(*upper);
        std::copy(stats.begin(), stats.end(), counters);
        return status;
    } catch (const std::bad_alloc&) { return -2; }
}
}
