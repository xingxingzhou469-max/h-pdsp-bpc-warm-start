#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

using Int = int64_t;
using Mask = uint32_t;

constexpr int maximum = 24;
int first(Mask mask) { return __builtin_ctz(mask) + 1; }
Mask bit(int position) { return Mask{1} << (position - 1); }

struct Totals {
    Int stock{}, people{};
    int service{}, arrivals{}, departures{}, agencies{};
    Totals operator+(const Totals& rhs) const {
        return {stock+rhs.stock, people+rhs.people, service+rhs.service,
                arrivals+rhs.arrivals, departures+rhs.departures, agencies+rhs.agencies};
    }
};

struct Verifier {
    int n{}, limit{};
    Int capacity{}, claimed{};
    std::array<Int, maximum+1> stock{}, pop{};
    std::array<int, maximum+1> service{}, route{};
    std::array<std::array<int, maximum+1>, maximum+1> travel{};
    Mask suppliers{}, agencies{};
    int split{};
    Mask lowbits{};
    Int largest{};
    int maximum_length{};
    std::array<int, maximum+1> maximizing_route{};
    std::vector<Totals> low, high;
    std::vector<uint16_t> prefixes;
    uint64_t all_sets{}, time_filter{}, commodity_filter{}, water_filter{}, dp_states{};
    uint64_t residual_sets{}, infeasible_sets{}, reverse_nodes{}, flow_filter{}, fairness_filter{}, scored{};

    void read() {
        if (!(std::cin >> n >> capacity >> limit >> claimed) || n < 2 || n > maximum ||
            limit < 0 || limit >= 65'533 || capacity <= 0 || claimed < 0)
            throw std::runtime_error("Unsupported audit input");
        for (int i = 0; i <= n; ++i) std::cin >> stock[i] >> pop[i] >> service[i];
        for (int i = 0; i <= n; ++i)
            for (int j = 0; j <= n; ++j) std::cin >> travel[i][j];
        if (!std::cin || stock[0] || pop[0] || service[0])
            throw std::runtime_error("Incomplete depot input");
        std::array<Totals, maximum+1> station{};
        for (int i = 0; i <= n; ++i) {
            int in = limit + 1, out = limit + 1;
            if (service[i] < 0 || stock[i] < 0 || pop[i] < 0 ||
                (i && (stock[i] == 0) == (pop[i] == 0)))
                throw std::runtime_error("Invalid station");
            for (int j = 0; j <= n; ++j) {
                if (travel[i][j] < 0) throw std::runtime_error("Negative travel time");
                // Allow arcs through every active node in this lower bound.
                if (i != j) {
                    in = std::min(in, travel[j][i]);
                    out = std::min(out, travel[i][j]);
                }
            }
            // Any pickup larger than capacity is equivalent to capacity.
            stock[i] = std::min(capacity, stock[i]);
            station[i] = {stock[i], pop[i], service[i], in, out, pop[i] > 0};
            if (stock[i]) suppliers |= bit(i);
            if (pop[i]) agencies |= bit(i);
        }
        split = n / 2;
        lowbits = (Mask{1} << split) - 1;
        low.resize(Mask{1} << split);
        high.resize(Mask{1} << (n - split));
        for (Mask mask = 1; mask < low.size(); ++mask)
            low[mask] = low[mask & (mask-1)] + station[first(mask)];
        for (Mask mask = 1; mask < high.size(); ++mask)
            high[mask] = high[mask & (mask-1)] + station[split+first(mask)];
        low[0].arrivals = station[0].arrivals;
        low[0].departures = station[0].departures;
        // Add the depot once per set, including sets with nonempty low bits.
        for (size_t i = 1; i < low.size(); ++i) {
            low[i].arrivals += station[0].arrivals;
            low[i].departures += station[0].departures;
        }
        prefixes.resize(size_t(n) << n);
    }

    Totals totals(Mask mask) const { return low[mask & lowbits] + high[mask >> split]; }

    // Explicit capped proportional allocation, with the Gini loss evaluated pair by pair.
    Int set_bound(Mask agency_mask, Int total_people, Int flow) const {
        const Mask all = agency_mask;
        Int denominator = total_people, remainder = flow;
        Mask saturated = 0;
        while (true) {
            Mask next_saturated = 0;
            for (Mask nodes = agency_mask; nodes; nodes &= nodes-1) {
                int v = first(nodes);
                if (pop[v] * remainder > capacity * denominator) next_saturated |= bit(v);
            }
            if (!next_saturated) break;
            for (Mask nodes = next_saturated; nodes; nodes &= nodes-1) {
                denominator -= pop[first(nodes)];
                remainder -= capacity;
            }
            saturated |= next_saturated;
            agency_mask ^= next_saturated;
        }
        if (!saturated) return flow * total_people;
        std::array<Int, maximum+1> amounts{};
        for (Mask nodes = all; nodes; nodes &= nodes-1) {
            int v = first(nodes);
            amounts[v] = saturated & bit(v) ? capacity * denominator : pop[v] * remainder;
        }
        Int penalty = 0;
        for (Mask nodes = all; nodes; nodes &= nodes-1) {
            int v = first(nodes);
            for (Mask later = nodes & (nodes-1); later; later &= later-1) {
                int w = first(later);
                penalty += std::abs(pop[v] * amounts[w] - pop[w] * amounts[v]);
            }
        }
        if (denominator <= 0) throw std::runtime_error("Invalid allocation denominator");
        return flow * total_people - penalty / denominator;
    }

    // Minimum depot-to-end path visiting exactly mask, including service at end.
    int prefix(Mask mask, int end) {
        auto& entry = prefixes[size_t(n)*mask + end-1];
        if (entry) return entry - 1;
        ++dp_states;
        int best = limit + 1;
        const Mask before = mask ^ bit(end);
        if (!before) {
            if (suppliers & bit(end)) best = std::min(best, travel[0][end] + service[end]);
        } else if (before & suppliers) {
            for (Mask prev = before; prev; prev &= prev-1) {
                int v = first(prev);
                int arc = travel[v][end] + service[end];
                if (arc < best) best = std::min(best, arc + prefix(before, v));
            }
        }
        entry = best + 1;
        return best;
    }

    // Enumerate delivery subsets directly to obtain ranks of the uncompressed route.
    Int score(int length, Int covered) const {
        std::vector<Int> people(1), rank(1);
        std::array<int, maximum+1> delivery_index{};
        Mask count = 1;
        for (int i = length-1; i >= 0; --i) {
            int v = route[i];
            if (pop[v]) {
                delivery_index[v] = count;
                count <<= 1;
            }
        }
        people.resize(count);
        rank.resize(count);
        for (Mask mask = 1; mask < count; ++mask) {
            Int load = 0;
            for (int i = length-1; i >= 0; --i) {
                int v = route[i];
                if (stock[v]) load = std::min(capacity, load + stock[v]);
                else if (mask & delivery_index[v]) {
                    rank[mask] += load;
                    people[mask] += pop[v];
                    load = 0;
                }
            }
        }
        // Find the lowest fair rate after locking the bottleneck subsets already found.
        Int value = 0, left = covered;
        const Mask full = count-1;
        for (Mask fixed = 0; fixed != full;) {
            Mask remaining = full ^ fixed;
            Mask selected = remaining;
            Int mass = people[remaining], quantity = rank[full] - rank[fixed];
            for (Mask mask = remaining; mask; mask = (mask-1) & remaining) {
                const Int q = rank[fixed|mask] - rank[fixed], p = people[mask];
                if (q * mass < quantity * p ||
                    (q * mass == quantity * p && p > mass)) {
                    selected = mask;
                    mass = p;
                    quantity = q;
                }
            }
            left -= mass;
            value += quantity * (mass + 2*left);
            fixed |= selected;
        }
        return value;
    }

    void peel(Mask remaining, int next, int suffix_time, int length, Int served,
              Int backward_load, Int people, Int flow_ceiling) {
        ++reverse_nodes;
        const Totals free = totals(remaining);
        // Reverse the arcs of commodity flow: agencies can send capacity and suppliers accept stock.
        const Int flow = std::min({flow_ceiling, served+free.stock,
                                  served+backward_load+capacity*free.agencies});
        if (flow * people <= claimed) {
            ++flow_filter;
            return;
        }
        // Agencies before this suffix can only be served by suppliers outside the suffix.
        if (flow * (people - free.people) + free.stock * people <= claimed) {
            ++fairness_filter;
            return;
        }
        if (!remaining) {
            ++scored;
            const Int value = score(length, people);
            if (value > largest) {
                largest = value;
                maximum_length = length;
                maximizing_route = route;
            }
            return;
        }
        for (Mask choices = next ? remaining : remaining & agencies; choices; choices &= choices-1) {
            int v = first(choices);
            int transition = travel[v][next] + suffix_time;
            if (transition + prefix(remaining, v) > limit) continue;
            route[length] = v;
            Int returned = std::min(stock[v], backward_load);
            peel(remaining ^ bit(v), v, transition + service[v], length+1,
                 served + returned, stock[v] ? backward_load-returned : capacity, people, flow_ceiling);
        }
    }

    void verify() {
        const Mask full = suppliers | agencies;
        for (Mask mask = 1; mask <= full; ++mask) {
            ++all_sets;
            const Totals measure = totals(mask);
            if (measure.service + std::max(measure.arrivals, measure.departures) > limit) {
                ++time_filter;
                continue;
            }
            const Int flow = std::min(measure.stock, capacity * measure.agencies);
            if (flow * measure.people <= claimed) {
                ++commodity_filter;
                continue;
            }
            const Int upper = set_bound(mask & agencies, measure.people, flow);
            if (upper <= claimed) {
                ++water_filter;
                continue;
            }
            bool reachable = false;
            for (Mask ends = mask & agencies; ends; ends &= ends-1) {
                int end = first(ends);
                if (prefix(mask, end) + travel[end][0] <= limit) {
                    reachable = true;
                    break;
                }
            }
            if (!reachable) {
                ++infeasible_sets;
                continue;
            }
            ++residual_sets;
            auto before = scored;
            peel(mask, 0, 0, 0, 0, 0, measure.people, flow);
            std::cout << "set," << mask << ',' << upper << ',' << scored-before << '\n';
        }
        std::cout << "all_sets " << all_sets << '\n'
                  << "time_filter " << time_filter << '\n'
                  << "commodity_filter " << commodity_filter << '\n'
                  << "water_filter " << water_filter << '\n'
                  << "dp_states " << dp_states << '\n'
                  << "infeasible_sets " << infeasible_sets << '\n'
                  << "residual_sets " << residual_sets << '\n'
                  << "reverse_nodes " << reverse_nodes << '\n'
                  << "flow_filter " << flow_filter << '\n'
                  << "fairness_filter " << fairness_filter << '\n'
                  << "scored " << scored << '\n'
                  << "max_scored_numerator " << largest << '\n'
                  << "best_route";
        for (int i = maximum_length-1; i >= 0; --i) std::cout << ' ' << maximizing_route[i];
        std::cout << '\n';
    }
};

int main() {
    Verifier verifier;
    verifier.read();
    verifier.verify();
    return verifier.largest > verifier.claimed;
}
