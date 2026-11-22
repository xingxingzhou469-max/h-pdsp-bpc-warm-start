#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <tuple>
#include <vector>

namespace {
using Mask = uint32_t;
using Bound = int64_t;
using Clock = std::chrono::steady_clock;
constexpr int max_sites = 29;  // includes the depot
constexpr int infinity = 65534;  // uint16 stores tail + 1, with 0 unused
#ifndef HPDP_SPARSE_BUCKET_BITS
#define HPDP_SPARSE_BUCKET_BITS 22
#endif
#ifndef HPDP_SPARSE_MAX_SETS
#define HPDP_SPARSE_MAX_SETS 262144
#endif
static_assert(HPDP_SPARSE_BUCKET_BITS >= 1 && HPDP_SPARSE_BUCKET_BITS <= 22);
static_assert(HPDP_SPARSE_MAX_SETS >= 1 && HPDP_SPARSE_MAX_SETS <= 262144);
constexpr size_t cache_buckets = size_t{1} << HPDP_SPARSE_BUCKET_BITS;
constexpr size_t max_candidates = HPDP_SPARSE_MAX_SETS;

int first_bit(Mask mask) { return __builtin_ctz(mask); }

struct Candidate {
    Mask mask;
    Bound upper;
};

struct Context {
    int n;  // non-depot sites numbered 1..n. Position zero stays the depot.
    int limit;
    Bound capacity;
    std::array<Bound, max_sites> supply{}, population{};
    std::array<int, max_sites> service{};
    std::array<std::array<int, max_sites>, max_sites> travel{};
    std::array<std::array<int, max_sites>, max_sites> outgoing{}, incoming{};
    std::vector<int> pop_order, branch_order;
    Mask pickup = 0, delivery = 0;
    std::vector<Candidate> candidates;
    std::vector<std::array<uint64_t, 2>> tails;
    std::array<int, max_sites> order{};
    // Enum states, objective cuts, basic time cuts, assignment cuts, DP states,
    // order states, time cuts in orders, flow cuts, evaluated routes, time-infeasible sets,
    // order cuts from earlier agencies, fixed-quantity target states, target closures.
    // Slots 17–20 record tail hits, tail replacements, retained sets, and memory.
    std::array<int64_t, 21> counts{};
    Clock::time_point stop;
    bool expired = false;
    bool set_limit = false;
    uint64_t calls = 0;

    Context(int size, int time_limit, Bound truck, const int* time, const int* times_at_sites,
            const Bound* stock, const Bound* people)
        : n(size), limit(time_limit), capacity(truck) {
        for (int i = 0; i <= n; ++i) {
            supply[i] = stock[i];
            population[i] = people[i];
            service[i] = times_at_sites[i];
            if (i > 0) {
                Mask bit = Mask{1} << (i-1);
                if (stock[i] > 0) pickup |= bit;
                else {
                    delivery |= bit;
                    pop_order.push_back(i);
                }
                branch_order.push_back(i);
            }
            for (int j = 0; j <= n; ++j) {
                travel[i][j] = time[i*(n+1)+j];
                outgoing[i][j] = incoming[i][j] = j;
            }
        }
        for (int i = 0; i <= n; ++i) {
            std::sort(outgoing[i].begin(), outgoing[i].begin()+n+1,
                      [&](int a, int b){return travel[i][a] < travel[i][b];});
            std::sort(incoming[i].begin(), incoming[i].begin()+n+1,
                      [&](int a, int b){return travel[a][i] < travel[b][i];});
        }
        std::sort(pop_order.begin(), pop_order.end(),
                  [&](int a, int b){return population[a] > population[b];});
        Bound total_pop = std::accumulate(population.begin(), population.end(), Bound{0});
        Bound total_supply = std::accumulate(supply.begin(), supply.end(), Bound{0});
        std::sort(branch_order.begin(), branch_order.end(), [&](int a, int b){
            return std::tuple{supply[a]*total_pop + population[a]*total_supply, service[a], a}
                 > std::tuple{supply[b]*total_pop + population[b]*total_supply, service[b], b};
        });
    }

    void set_budget(double seconds) {
        stop = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
        expired = seconds <= 0;
        calls = 0;
    }

    bool timed_out() {
        if ((++calls & 8191) == 0) expired |= Clock::now() >= stop;
        return expired;
    }

    int arc(int a, int b) const {
        if (a == b || (a == 0 && supply[b] == 0) || (b == 0 && population[a] == 0))
            return infinity;
        return travel[a][b];
    }

    Bound box(Mask selected, Bound covered, Bound flow) const {
        Bound value = flow * covered;
        Bound fixed = 0, h = 0;
        for (int v : pop_order) {
            if (!(selected & (Mask{1} << (v-1)))) continue;
            ++h;
            Bound people = population[v];
            if (people * (flow - (h-1)*capacity) <= capacity * covered) break;
            covered -= people;
            fixed += capacity * (2*h-1)*people;
            value = fixed + covered * (flow + h*capacity);
        }
        return value;
    }

    // One nearest allowable incoming and outgoing arc, only for stations
    // already forced into the tour. Undecided sites may supply these arcs.
    int local_bound(Mask required, Mask possible) const {
        Mask selected = 1 | (required << 1);
        Mask allowed = 1 | (possible << 1);
        int total_out = 0, total_in = 0;
        while (selected) {
            int v = first_bit(selected);
            selected &= selected - 1;
            int best_out = infinity, best_in = infinity;
            for (int i = 0; i <= n; ++i) {
                int to = outgoing[v][i];
                if ((allowed & (Mask{1} << to)) && arc(v, to) != infinity) {
                    best_out = travel[v][to];
                    break;
                }
            }
            for (int i = 0; i <= n; ++i) {
                int from = incoming[v][i];
                if ((allowed & (Mask{1} << from)) && arc(from, v) != infinity) {
                    best_in = travel[from][v];
                    break;
                }
            }
            total_out += best_out;
            total_in += best_in;
        }
        return std::max(total_out, total_in);
    }

    // Hungarian algorithm on selected stations plus depot. The assignment
    // permits multiple circuits. It lower-bounds any single directed cycle.
    int assignment_bound(Mask selected) const {
        std::array<int, max_sites+1> sites{};
        int k = 1;
        while (selected) {
            sites[++k] = first_bit(selected) + 1;
            selected &= selected - 1;
        }
        std::array<int, max_sites+1> u{}, v{}, matching{}, prev{};
        for (int i = 1; i <= k; ++i) {
            matching[0] = i;
            int column = 0;
            std::array<int, max_sites+1> gap{};
            std::array<bool, max_sites+1> used{};
            gap.fill(infinity * max_sites);
            do {
                used[column] = true;
                int row = matching[column];
                int next_column = 0;
                int delta = infinity * max_sites;
                for (int j = 1; j <= k; ++j) {
                    if (used[j]) continue;
                    int reduced = arc(sites[row], sites[j]) - u[row] - v[j];
                    if (reduced < gap[j]) {
                        gap[j] = reduced;
                        prev[j] = column;
                    }
                    if (gap[j] < delta) {
                        delta = gap[j];
                        next_column = j;
                    }
                }
                for (int j = 0; j <= k; ++j) {
                    if (used[j]) {
                        u[matching[j]] += delta;
                        v[j] -= delta;
                    } else {
                        gap[j] -= delta;
                    }
                }
                column = next_column;
            } while (matching[column] != 0);

            do {
                int before = prev[column];
                matching[column] = matching[before];
                column = before;
            } while (column != 0);
        }
        return -v[0];
    }

    void enumerate(int i, Mask required, Mask possible, int mandatory_service,
                   Bound stock, Bound people, int agencies, Bound cutoff) {
        if (timed_out() || set_limit) return;
        ++counts[0];
        Bound flow = std::min(stock, capacity * agencies);
        if (!(possible & pickup) || !(possible & delivery) || flow*people <= cutoff) {
            ++counts[1];
            return;
        }
        Bound upper = box(possible, people, flow);
        if (upper <= cutoff) {
            ++counts[1];
            return;
        }
        if (mandatory_service > limit || mandatory_service + local_bound(required, possible) > limit) {
            ++counts[2];
            return;
        }
        if (i == n) {
            if (mandatory_service + assignment_bound(required) > limit) {
                ++counts[3];
                return;
            }
            if (candidates.size() == max_candidates) {
                set_limit = true;
                return;
            }
            candidates.push_back({required, upper});
            ++counts[19];
            return;
        }

        int v = branch_order[i];
        Mask bit = Mask{1} << (v-1);
        enumerate(i+1, required|bit, possible, mandatory_service+service[v], stock, people, agencies, cutoff);
        enumerate(i+1, required, possible^bit, mandatory_service, stock-supply[v], people-population[v],
                  agencies-(population[v] > 0), cutoff);
    }

    void collect(Bound cutoff, double seconds) {
        set_budget(seconds);
        if (expired) return;
        enumerate(0, 0, pickup|delivery, 0,
                  std::accumulate(supply.begin(), supply.end(), Bound{0}),
                  std::accumulate(population.begin(), population.end(), Bound{0}),
                  static_cast<int>(pop_order.size()), cutoff);
        expired |= Clock::now() >= stop;
        if (expired || set_limit) return;
        std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b){
            return std::tuple{-a.upper, a.mask} < std::tuple{-b.upper, b.mask};
        });
        expired |= Clock::now() >= stop;
    }

    int tail(Mask mask, int v) {
        if (timed_out()) return infinity;
        Mask bit = Mask{1} << (v-1);
        const uint64_t key = (uint64_t(mask) << 5) | v;
        uint64_t hash = (key ^ (key >> 16)) * UINT64_C(0xbf58476d1ce4e5b9);
        hash = (hash ^ (hash >> 27)) * UINT64_C(0x94d049bb133111eb);
        auto& bucket = tails[(hash ^ (hash >> 31)) & (cache_buckets-1)];
        for (uint64_t entry : bucket) {
            if (entry && (entry >> 16) == key) {
                ++counts[17];
                return int(entry & 65535) - 1;
            }
        }
        ++counts[4];
        const Mask rest = mask ^ bit;
        int best = infinity;
        if (!rest) {
            best = arc(v, 0);
        } else {
            for (Mask choices = rest; choices && !expired; choices &= choices-1) {
                int next = first_bit(choices)+1;
                int step = travel[v][next];
                if (step >= best) continue;
                best = std::min(best, step+tail(rest, next));
            }
        }
        if (!expired) {
            // Recursion may replace another state in this bucket. Always
            // check for spare space after recursion, and never reuse a value
            // whose complete key differs from the caller's state.
            int slot = !bucket[0] ? 0 : !bucket[1] ? 1 : int(hash >> 63);
            counts[18] += bucket[slot] != 0;
            bucket[slot] = (key << 16) | uint64_t(best+1);
        }
        return best;
    }

    using Observe = void (*)(int, const int*, Bound*);

    bool observe_order(int length, const int* route, Bound& cutoff, Observe observe) {
        if (Clock::now() >= stop) {
            expired = true;
            return false;
        }
        Bound newest = -1;
        observe(length, route, &newest);
        if (newest >= 0) cutoff = newest;
        expired = newest < 0 || Clock::now() >= stop;
        return !expired;
    }

    // Try to realize the set relaxation's ideal capped proportional delivery.
    // A completely realized target reaches the set bound; failing to find one
    // never prunes an order that could still improve the incumbent.
    bool target_order(Mask remaining, int last, int spent, int length, Bound load, Bound stock,
                      Bound demand, Bound denominator, Bound truck,
                      const std::array<Bound, max_sites>& quantities, int budget,
                      int& work, Bound& cutoff, Observe observe, bool exact) {
        if (timed_out() || work <= 0 || load+stock < demand) return false;
        --work;
        ++counts[exact ? 11 : 15];
        if (!remaining) {
            if (spent + arc(last, 0) > budget) return false;
            return observe_order(length, order.data(), cutoff, observe);
        }
        Mask choices = last ? remaining : remaining & pickup;
        if (!(remaining & (remaining-1))) choices &= delivery;
        for (int i = 0; i <= n && work > 0 && choices && !expired; ++i) {
            int next = outgoing[last][i];
            if (next == 0) continue;
            Mask bit = Mask{1} << (next-1);
            if (!(choices & bit)) continue;
            choices ^= bit;
            if (load < quantities[next]) continue;
            int duration = spent+travel[last][next];
            if (duration > budget || (exact && duration+tail(remaining, next) > budget)) continue;
            order[length] = next;
            Bound pickup_amount = supply[next] * denominator;
            if (target_order(remaining^bit, next, duration, length+1,
                             std::min(truck, load+pickup_amount) - quantities[next],
                             stock-pickup_amount, demand-quantities[next], denominator,
                             truck, quantities, budget, work, cutoff, observe, exact)) return true;
        }
        return false;
    }

    void beam_target(Mask mask, Bound stock, Bound demand, Bound denominator, Bound truck,
                     const std::array<Bound, max_sites>& quantities, int budget,
                     Bound& cutoff, Observe observe) {
        struct RouteLabel {
            Mask remaining;
            Bound load, stock, demand;
            int last, elapsed, deadline_floor;
            std::array<int, max_sites> route{};
        };
        constexpr size_t width = 96;
        std::array<int, max_sites> cheapest_out{}, cheapest_in{};
        for (int u = 0; u <= n; ++u) {
            cheapest_out[u] = cheapest_in[u] = infinity;
            for (int v = 0; v <= n; ++v) {
                cheapest_out[u] = std::min(cheapest_out[u], arc(u, v));
                cheapest_in[u] = std::min(cheapest_in[u], arc(v, u));
            }
        }
        auto floor = [&](Mask remaining, int last) {
            int current_out = cheapest_out[last], final_in = cheapest_in[0];
            while (remaining) {
                int v = first_bit(remaining)+1;
                remaining &= remaining-1;
                current_out += cheapest_out[v];
                final_in += cheapest_in[v];
            }
            return std::max(current_out, final_in);
        };
        std::vector<RouteLabel> layer, next_layer;
        layer.push_back({mask, 0, stock, demand, 0, 0, 0, {}});
        next_layer.reserve(width * n);
        for (int depth = 0; !layer.empty() && !expired; ++depth) {
            next_layer.clear();
            for (const auto& label : layer) {
                if (timed_out()) return;
                ++counts[15];
                Mask choices = label.last ? label.remaining : label.remaining & pickup;
                if (!(label.remaining & (label.remaining-1))) choices &= delivery;
                while (choices) {
                    int v = first_bit(choices)+1;
                    Mask bit = Mask{1} << (v-1);
                    choices ^= bit;
                    if (label.load < quantities[v]) continue;
                    int elapsed = label.elapsed+travel[label.last][v];
                    Mask remaining = label.remaining^bit;
                    int end = elapsed + (remaining ? floor(remaining,v) : arc(v,0));
                    if (end > budget) continue;
                    Bound collected = supply[v]*denominator;
                    Bound load = std::min(truck,label.load+collected)-quantities[v];
                    Bound next_stock = label.stock-collected;
                    Bound next_demand = label.demand-quantities[v];
                    if (load+next_stock < next_demand) continue;
                    RouteLabel candidate{remaining, load, next_stock, next_demand, v, elapsed, end, label.route};
                    candidate.route[depth] = v;
                    if (!remaining) {
                        observe_order(depth+1, candidate.route.data(), cutoff, observe);
                        return;
                    }
                    next_layer.push_back(candidate);
                }
            }
            auto key = [](const auto& a) { return std::tuple{a.remaining,a.last,a.load}; };
            std::sort(next_layer.begin(),next_layer.end(),[&](const auto& a,const auto& b){
                return std::tuple{key(a),a.elapsed} < std::tuple{key(b),b.elapsed};
            });
            next_layer.erase(std::unique(next_layer.begin(),next_layer.end(),[&](const auto& a,const auto& b){
                return key(a)==key(b);
            }),next_layer.end());
            if (next_layer.size()>width) {
                auto ranking = [](const auto& a){
                    return std::tuple{a.deadline_floor,a.elapsed,-a.load,a.remaining,a.last};
                };
                std::partial_sort(next_layer.begin(),next_layer.begin()+width,next_layer.end(),[&](const auto& a,const auto& b){
                    return ranking(a)<ranking(b);
                });
                next_layer.resize(width);
            }
            layer.swap(next_layer);
        }
    }

    void try_target(Mask mask, Bound stock, Bound people, int agencies, int budget,
                    Bound& cutoff, Observe observe, bool exact=true) {
        Bound available = std::min(stock, capacity*agencies);
        Bound total = available, denominator = people;
        std::array<Bound, max_sites> quantities{};
        for (int v : pop_order) {
            if (!(mask & (Mask{1} << (v-1)))) continue;
            if (population[v]*available <= capacity*denominator) break;
            quantities[v] = capacity;
            denominator -= population[v];
            available -= capacity;
        }
        for (int v : pop_order) {
            if (!(mask & (Mask{1} << (v-1)))) continue;
            quantities[v] = quantities[v] ? capacity * denominator : available * population[v];
        }
        if (!exact) {
            beam_target(mask, stock*denominator, total*denominator, denominator,
                        capacity*denominator, quantities, budget, cutoff, observe);
            return;
        }
        int work = 50'000;
        target_order(mask, 0, 0, 0, 0, stock*denominator, total*denominator, denominator,
                     capacity*denominator, quantities, budget, work, cutoff, observe, exact);
    }

    void audit(Mask remaining, int last, int spent, int length, Bound unloaded, Bound load,
               Bound stock, int agencies, int budget, Mask full, Bound covered, Bound seen, Bound& cutoff,
               Observe observe) {
        if (timed_out()) return;
        ++counts[5];
        Bound flow = std::min(unloaded+load+stock, unloaded+capacity*agencies);
        if (flow*covered <= cutoff || box(full, covered, flow) <= cutoff) {
            ++counts[7];
            return;
        }
        // Earlier agencies cannot benefit from any stock picked up later.
        // Aggregate them into one population-weighted bucket before using
        // the cross-bucket absolute-difference term of the fairness penalty.
        if (flow * (covered-seen) + unloaded * covered <= cutoff) {
            ++counts[10];
            return;
        }
        if (seen > 0) {
            ++counts[13];
            Bound capped_unloaded = 0;
            Bound aboard = 0;
            for (int k = 0; k < length; ++k) {
                int site = order[k];
                if (supply[site] > 0) {
                    aboard = std::min(capacity * covered, aboard + supply[site] * covered);
                } else {
                    Bound delivery_now = std::min(aboard, population[site] * flow);
                    aboard -= delivery_now;
                    capped_unloaded += delivery_now;
                }
            }
            if (flow * (covered-seen) + capped_unloaded <= cutoff) {
                ++counts[14];
                return;
            }
        }
        if (!remaining) {
            ++counts[8];
            observe_order(length, order.data(), cutoff, observe);
            return;
        }
        Mask choices = last ? remaining : remaining & pickup;
        if (!(remaining & (remaining-1))) choices &= delivery;
        while (choices && !expired) {
            int next = first_bit(choices)+1;
            Mask bit = Mask{1} << (next-1);
            choices ^= bit;
            int duration = spent+travel[last][next];
            if (duration > budget || duration+tail(remaining, next) > budget) {
                ++counts[6];
                continue;
            }
            order[length] = next;
            if (supply[next] > 0)
                audit(remaining^bit, next, duration, length+1, unloaded, std::min(capacity, load+supply[next]),
                      stock-supply[next], agencies, budget, full, covered, seen, cutoff, observe);
            else
                audit(remaining^bit, next, duration, length+1, unloaded+load, 0, stock, agencies-1,
                      budget, full, covered, seen+population[next], cutoff, observe);
        }
    }

    void inspect(size_t i, Bound cutoff, double seconds, Observe observe) {
        set_budget(seconds);
        if (expired) return;
        const Mask mask = candidates[i].mask;
        int budget = limit, agencies = 0;
        Bound people = 0, stock = 0;
        for (Mask positions = mask; positions; positions &= positions-1) {
            int v = first_bit(positions)+1;
            budget -= service[v];
            agencies += population[v] > 0;
            people += population[v];
            stock += supply[v];
        }
        if (i < 4) {
            try_target(mask, stock, people, agencies, budget, cutoff, observe, false);
            if (expired) return;
            if (cutoff >= candidates[i].upper) {
                ++counts[16];
                return;
            }
        }
        if (tails.empty()) {
            tails.resize(cache_buckets);
            counts[20] = tails.size() * sizeof(tails[0]);
        }
        expired = Clock::now() >= stop;
        if (expired) return;
        bool possible = false;
        for (Mask choices = mask & pickup; choices && !expired; choices &= choices-1) {
            int next = first_bit(choices)+1;
            if (travel[0][next]+tail(mask, next) <= budget) {
                possible = true;
                break;
            }
        }
        if (possible) {
            try_target(mask, stock, people, agencies, budget, cutoff, observe);
            if (expired) return;
            if (cutoff >= candidates[i].upper) {
                ++counts[12];
                return;
            }
            audit(mask, 0, 0, 0, 0, 0, stock, agencies, budget, mask, people, 0, cutoff, observe);
        } else if (!expired) {
            ++counts[9];
        }
        expired |= Clock::now() >= stop;
    }
};
}  // namespace

extern "C" {
Context* hpdp_sparse_create(int size, int limit, Bound capacity, const int* time, const int* service,
                           const Bound* stock, const Bound* pop) noexcept {
    try {
        return new Context(size, limit, capacity, time, service, stock, pop);
    } catch (...) {
        return nullptr;
    }
}
void hpdp_sparse_destroy(Context* context) { delete context; }
int hpdp_sparse_counter_count() noexcept {
    return std::tuple_size<decltype(Context::counts)>::value;
}
int hpdp_sparse_collect(Context* context, Bound cutoff, double seconds) noexcept {
    try {
        context->collect(cutoff, seconds);
        if (context->expired || Clock::now() >= context->stop) return -1;
        if (context->set_limit) return -3;
        return static_cast<int>(context->candidates.size());
    } catch (...) {
        return -2;
    }
}
Bound hpdp_sparse_upper(Context* context, size_t i) { return context->candidates[i].upper; }
uint32_t hpdp_sparse_mask(Context* context, size_t i) { return context->candidates[i].mask; }
int hpdp_sparse_inspect_v2(Context* context, size_t i, Bound cutoff, double seconds,
                       Context::Observe observe) noexcept {
    try {
        context->inspect(i, cutoff, seconds, observe);
        return context->expired || Clock::now() >= context->stop ? -1 : 0;
    } catch (...) {
        return -2;
    }
}
Bound hpdp_sparse_count(Context* context, int i) {
    if (i < 0 || size_t(i) >= context->counts.size()) return -1;
    return context->counts[i];
}
}
