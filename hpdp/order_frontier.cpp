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
constexpr int max_sites = 21;  // includes the depot
constexpr int infinity = 65534;  // uint16 stores tail + 1, with 0 unused

int first_bit(Mask mask) { return __builtin_ctz(mask); }

uint64_t mix(uint64_t value) {
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

// The signature contains capped consecutive supplies and summed consecutive
// populations, with the sign identifying the role. A cache slot also retains
// the entire structure, so a hash collision only causes replacement.
struct OrderMemo {
    Mask remaining = 0;
    int last = 0, spent = 0, size = 0;
    uint32_t generation = 0;
    std::array<Bound, max_sites-1> blocks{};
};

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
    std::vector<uint16_t> tails;
    std::array<int, max_sites> order{};
    const size_t cache_slots;
    const bool prefix_rank_bound;
    const bool minimum_delivery;
    std::vector<OrderMemo> memo;
    uint32_t generation = 0;
    std::array<Bound, max_sites-1> blocks{};
    std::array<uint64_t, max_sites> block_hashes{};
    // Enum states, objective cuts, basic time cuts, assignment cuts, DP states,
    // order states, time cuts in orders, flow cuts, evaluated routes, time-infeasible sets,
    // order cuts from earlier agencies, fixed-quantity target states, target closures.
    std::array<int64_t, 23> counts{};
    Clock::time_point stop;
    bool expired = false;
    uint64_t calls = 0;

    Context(int size, int time_limit, Bound truck, const int* time, const int* times_at_sites,
            const Bound* stock, const Bound* people, size_t slots, bool rank_bound, bool minimum)
        : n(size), limit(time_limit), capacity(truck), cache_slots(slots),
          prefix_rank_bound(rank_bound), minimum_delivery(minimum) {
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
        if (timed_out()) return;
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
            candidates.push_back({required, upper});
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
        if (expired) return;
        std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b){
            return std::tuple{-a.upper, a.mask} < std::tuple{-b.upper, b.mask};
        });
        expired |= Clock::now() >= stop;
    }

    int tail(Mask mask, int v) {
        if (timed_out()) return infinity;
        Mask bit = Mask{1} << (v-1);
        const Mask low = mask & (bit-1);
        const Mask high = (mask >> v) << (v-1);
        const size_t index = (size_t(v-1) << (n-1)) + (low | high);
        uint16_t& found = tails[index];
        if (found) return found-1;
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
        if (!expired) found = static_cast<uint16_t>(best+1);
        return best;
    }

    using Observe = void (*)(int, const int*, Bound*);

    // Try to realize the set relaxation's ideal capped proportional delivery.
    // A completely realized target reaches the set bound; failing to find one
    // never prunes an order that could still improve the incumbent.
    bool target_order(Mask remaining, int last, int spent, int length, Bound load, Bound stock,
                      Bound demand, Bound denominator, Bound truck,
                      const std::array<Bound, max_sites>& quantities, int budget,
                      int& work, Bound& cutoff, Observe observe) {
        if (timed_out() || work <= 0 || load+stock < demand) return false;
        --work;
        ++counts[11];
        if (!remaining) {
            Bound newest = -1;
            observe(length, order.data(), &newest);
            if (newest < 0) expired = true;
            else cutoff = newest;
            return !expired;
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
            if (duration > budget || duration+tail(remaining, next) > budget) continue;
            order[length] = next;
            Bound pickup_amount = supply[next] * denominator;
            if (target_order(remaining^bit, next, duration, length+1,
                             std::min(truck, load+pickup_amount) - quantities[next],
                             stock-pickup_amount, demand-quantities[next], denominator,
                             truck, quantities, budget, work, cutoff, observe)) return true;
        }
        return false;
    }

    struct Target {
        Bound total, denominator;
        std::array<Bound, max_sites> quantities{};
    };

    Target ideal_target(Mask mask, Bound stock, Bound people, int agencies) const {
        Bound available = std::min(stock, capacity*agencies);
        Target target{available, people};
        for (int v : pop_order) {
            if (!(mask & (Mask{1} << (v-1)))) continue;
            if (population[v]*available <= capacity*target.denominator) break;
            target.quantities[v] = capacity;
            target.denominator -= population[v];
            available -= capacity;
        }
        for (int v : pop_order) {
            if (!(mask & (Mask{1} << (v-1)))) continue;
            target.quantities[v] = target.quantities[v]
                ? capacity * target.denominator : available * population[v];
        }
        return target;
    }

    void try_target(Mask mask, Bound stock, const Target& target, int budget,
                    Bound& cutoff, Observe observe) {
        int work = 50'000;
        target_order(mask, 0, 0, 0, 0, stock*target.denominator, target.total*target.denominator,
                     target.denominator, capacity*target.denominator, target.quantities,
                     budget, work, cutoff, observe);
    }

    // Prefixes that match have identical continuation profiles. Time
    // dominance applies only at the same endpoint. At a complete feasible
    // route no continuation remains, so the last stop can be ignored.
    bool dominated(Mask remaining, int last, int spent, int block_count) {
        if (!remaining) last = spent = 0;
        const uint64_t state = (uint64_t{remaining} << 5) | last;
        const size_t index = (mix(state) ^ block_hashes[block_count]) & (memo.size()-1);
        auto& entry = memo[index];
        ++counts[17];
        if (entry.generation == generation) {
            if (entry.remaining == remaining && entry.last == last && entry.size == block_count
                    && std::equal(blocks.begin(), blocks.begin()+block_count, entry.blocks.begin())) {
                if (entry.spent <= spent) {
                    ++counts[remaining ? 15 : 16];
                    return true;
                }
                entry.spent = spent;
                return false;
            }
            ++counts[18];
        }
        entry.remaining = remaining;
        entry.last = last;
        entry.spent = spent;
        entry.size = block_count;
        entry.generation = generation;
        std::copy_n(blocks.begin(), block_count, entry.blocks.begin());
        return false;
    }

    template<bool memo_on>
    void extend_blocks(int& count, int next) {
        const Bound value = supply[next] ? supply[next] : -population[next];
        if (count && (value > 0) == (blocks[count-1] > 0)) {
            if (value > 0) blocks[count-1] = std::min(capacity, blocks[count-1]+value);
            else blocks[count-1] += value;
        } else {
            blocks[count++] = value;
        }
        if constexpr (memo_on)
            block_hashes[count] = mix(block_hashes[count-1] ^ static_cast<uint64_t>(blocks[count-1]));
    }

    // Retain the ordered capacity constraints for past agencies. Give all
    // future agencies a single independent supply and cap the combined
    // delivery by flow. The virtual group's rank is flow whenever selected.
    struct PrefixRank {
        Bound capacity, flow;
        int past;
        const std::array<Bound, max_sites-1>& blocks;
        Mask virtual_group;
        Bound virtual_population;

        Bound rank(Mask selected) const {
            if (selected & virtual_group) return flow;
            Bound load = 0, total = 0;
            for (int i = 0; i < past; ++i) {
                load = std::min(capacity, load + blocks[2*i]);
                if (selected & (Mask{1} << i)) {
                    total += load;
                    load = 0;
                }
            }
            return total;
        }

        Bound population(Mask selected) const {
            Bound total = selected & virtual_group ? virtual_population : 0;
            for (int i = 0; i < past; ++i) {
                if (selected & (Mask{1} << i)) total -= blocks[2*i+1];
            }
            return total;
        }

        Mask chord_split(Mask fixed, Mask allowed, Bound price, Bound scale, Bound end_rank) const {
            if (fixed & virtual_group) return 0;
            // An allowed set containing the virtual group has rank flow.
            // Its least penalized representative is already the endpoint.
            Bound out = 0, in = 0;
            Mask out_mask = 0, in_mask = 0;
            const Bound carry = capacity * scale;
            constexpr Bound forbidden = Bound{1} << 61;
            for (int i = 0; i < past; ++i) {
                Bound next_out = std::min(out, in + carry) + blocks[2*i] * scale;
                Mask next_out_mask = in + carry < out ? in_mask : out_mask;
                if (out < in) {
                    in = out;
                    in_mask = out_mask;
                }
                out = next_out;
                out_mask = next_out_mask;

                Mask bit = Mask{1} << i;
                next_out = std::min(out, in + carry);
                next_out_mask = in + carry < out ? in_mask : out_mask;
                if (out < in) {
                    in = out;
                    in_mask = out_mask;
                }
                if (fixed & bit) in = forbidden;
                else if (allowed & bit) in -= price * blocks[2*i+1];
                out = next_out;
                out_mask = next_out_mask | (allowed & bit);
            }
            Bound penalty = allowed & virtual_group ? price * virtual_population : 0;
            if (std::min(out, in) + penalty >= end_rank * scale) return 0;
            return out <= in ? out_mask : in_mask;
        }

        Bound area(Mask left, Mask right, Bound rank_l, Bound rank_r, Bound pop_l,
                   Bound pop_r, Bound covered) const {
            Bound weight = pop_r - pop_l, amount = rank_r - rank_l;
            Mask span = right ^ left;
            if ((span & (span-1)) && amount) {
                Mask middle = chord_split(left, right, amount, weight, rank_r);
                if (middle) {
                    Bound rank_m = rank(middle), pop_m = population(middle);
                    return area(left, middle, rank_l, rank_m, pop_l, pop_m, covered)
                         + area(middle, right, rank_m, rank_r, pop_m, pop_r, covered);
                }
            }
            return amount * (weight + 2 * (covered-pop_r));
        }
    };

    template<bool memo_on, bool rank_on, bool minimum_on = false>
    void audit(Mask remaining, int last, int spent, int length, Bound unloaded, Bound load,
               Bound stock, int agencies, int budget, Mask full, Bound covered, Bound seen,
               int block_count, Bound& cutoff, Observe observe, const Target& target,
               Bound set_upper, Bound after_minimum = 0, Bound ideal_remaining = 0) {
        if (timed_out()) return;
        ++counts[5];
        if constexpr (minimum_on) {
            // Sum every remaining agency's necessary minimum without
            // clipping at zero. Ignore the capacity of later pickup runs.
            const Bound deficit = ideal_remaining - after_minimum - stock * target.denominator;
            if (agencies && deficit > 0 && (deficit+agencies-1)/agencies > set_upper-cutoff) {
                ++counts[22];
                return;
            }
        }
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
        if constexpr (memo_on) {
            if (length >= 3 && dominated(remaining, last, spent, block_count)) return;
        }
        if constexpr (rank_on) {
            // Up to one historical delivery block has only the single
            // aggregate deficit already handled by the capped demand bound.
            if (block_count >= 4) {
                ++counts[19];
                const int past = block_count / 2;
                const Mask future = agencies ? Mask{1} << past : 0;
                const Mask groups = ((Mask{1} << past)-1) | future;
                const PrefixRank relaxation{capacity, flow, past, blocks, future, covered-seen};
                if (relaxation.area(0, groups, 0, flow, 0, covered, covered) <= cutoff) {
                    ++counts[20];
                    return;
                }
            }
        }
        if (!remaining) {
            ++counts[8];
            Bound new_bound = -1;
            observe(length, order.data(), &new_bound);
            if (new_bound < 0) expired = true;
            else cutoff = new_bound;
            return;
        }
        Mask choices = last ? remaining : remaining & pickup;
        if (!(remaining & (remaining-1))) choices &= delivery;
        while (choices && !expired) {
            int next = first_bit(choices)+1;
            Mask bit = Mask{1} << (next-1);
            choices ^= bit;
            Bound next_after_minimum = after_minimum;
            if constexpr (minimum_on) {
                // Any improvement must deliver at least the ideal amount
                // minus the current set gap divided by the free population.
                if (supply[next]) {
                    next_after_minimum = std::min(capacity * target.denominator,
                                                   after_minimum + supply[next] * target.denominator);
                } else {
                    Bound required = std::max(Bound{0}, target.quantities[next] - (set_upper-cutoff));
                    if (after_minimum < required) {
                        ++counts[21];
                        continue;
                    }
                    next_after_minimum -= required;
                }
            }
            int duration = spent+travel[last][next];
            if (duration > budget || duration+tail(remaining, next) > budget) {
                ++counts[6];
                continue;
            }
            order[length] = next;
            int extended = block_count;
            Bound saved = 0;
            uint64_t saved_hash = 0;
            if constexpr (memo_on || rank_on) {
                if (block_count) saved = blocks[block_count-1];
                if constexpr (memo_on) saved_hash = block_hashes[block_count];
                extend_blocks<memo_on>(extended, next);
            }
            if (supply[next] > 0)
                audit<memo_on, rank_on, minimum_on>(remaining^bit, next, duration, length+1, unloaded,
                     std::min(capacity, load+supply[next]), stock-supply[next], agencies, budget,
                     full, covered, seen, extended, cutoff, observe, target, set_upper,
                     next_after_minimum, ideal_remaining);
            else
                audit<memo_on, rank_on, minimum_on>(remaining^bit, next, duration, length+1, unloaded+load, 0,
                     stock, agencies-1, budget, full, covered, seen+population[next], extended, cutoff,
                     observe, target, set_upper, next_after_minimum,
                     minimum_on ? ideal_remaining-target.quantities[next] : 0);
            if constexpr (memo_on || rank_on) {
                if (block_count) blocks[block_count-1] = saved;
                if constexpr (memo_on) block_hashes[block_count] = saved_hash;
            }
        }
    }

    void inspect(size_t i, Bound cutoff, double seconds, Observe observe) {
        set_budget(seconds);
        if (expired) return;
        if (tails.empty()) tails.resize(size_t(n) << (n-1), 0);
        expired = Clock::now() >= stop;
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
        bool possible = false;
        for (Mask choices = mask & pickup; choices && !expired; choices &= choices-1) {
            int next = first_bit(choices)+1;
            if (travel[0][next]+tail(mask, next) <= budget) {
                possible = true;
                break;
            }
        }
        if (possible) {
            const Target target = ideal_target(mask, stock, people, agencies);
            const Bound set_upper = candidates[i].upper;
            try_target(mask, stock, target, budget, cutoff, observe);
            if (expired) return;
            if (cutoff >= set_upper) {
                ++counts[12];
                return;
            }
            if (cache_slots && memo.empty()) memo.resize(cache_slots);
            ++generation;
            if (minimum_delivery)
                audit<false, false, true>(mask, 0, 0, 0, 0, 0, stock, agencies, budget, mask,
                                         people, 0, 0, cutoff, observe, target, set_upper, 0,
                                         target.total*target.denominator);
            else if (cache_slots && prefix_rank_bound)
                audit<true, true>(mask, 0, 0, 0, 0, 0, stock, agencies, budget, mask,
                                  people, 0, 0, cutoff, observe, target, set_upper);
            else if (cache_slots)
                audit<true, false>(mask, 0, 0, 0, 0, 0, stock, agencies, budget, mask,
                                   people, 0, 0, cutoff, observe, target, set_upper);
            else if (prefix_rank_bound)
                audit<false, true>(mask, 0, 0, 0, 0, 0, stock, agencies, budget, mask,
                                   people, 0, 0, cutoff, observe, target, set_upper);
            else
                audit<false, false>(mask, 0, 0, 0, 0, 0, stock, agencies, budget, mask,
                                    people, 0, 0, cutoff, observe, target, set_upper);
        } else if (!expired) {
            ++counts[9];
        }
        expired |= Clock::now() >= stop;
    }
};
}  // namespace

extern "C" {
Context* hpdp_order_frontier_create(int size, int limit, Bound capacity, const int* time, const int* service,
                           const Bound* stock, const Bound* pop, size_t slots, bool rank_bound,
                           bool minimum) noexcept {
    try {
        if (size < 2 || size >= max_sites || slots > (size_t{1} << 18)
                || (slots && (slots & (slots-1))) || (minimum && (slots || rank_bound))) return nullptr;
        return new Context(size, limit, capacity, time, service, stock, pop, slots, rank_bound, minimum);
    } catch (...) {
        return nullptr;
    }
}
void hpdp_order_frontier_destroy(Context* context) { delete context; }
int hpdp_order_frontier_counter_count() noexcept {
    return std::tuple_size<decltype(Context::counts)>::value;
}
int hpdp_order_frontier_collect(Context* context, Bound cutoff, double seconds) noexcept {
    try {
        context->collect(cutoff, seconds);
        return context->expired || Clock::now()>=context->stop ? -1 : static_cast<int>(context->candidates.size());
    } catch (...) {
        return -2;
    }
}
Bound hpdp_order_frontier_upper(Context* context, size_t i) { return context->candidates[i].upper; }
uint32_t hpdp_order_frontier_mask(Context* context, size_t i) { return context->candidates[i].mask; }
int hpdp_order_frontier_inspect_v2(Context* context, size_t i, Bound cutoff, double seconds,
                       Context::Observe observe) noexcept {
    try {
        context->inspect(i, cutoff, seconds, observe);
        return context->expired || Clock::now()>=context->stop ? -1 : 0;
    } catch (...) {
        return -2;
    }
}
Bound hpdp_order_frontier_count(Context* context, int i) {
    if(i<0 || size_t(i)>=context->counts.size()) return -1;
    return context->counts[i];
}
}
