// Exact few-supplier search. New lower bounds are acknowledged by the witness verifier.
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <unistd.h>

using I = std::int64_t;
using Wide = __int128_t;
using Mask = std::uint64_t;
using Clock = std::chrono::steady_clock;
constexpr int Impossible = -1;
constexpr int Forever = 1000000;
constexpr int Donors = 4;
constexpr int Sites = 67;

struct Expired {};
struct Disconnected {};

std::string decimal(Wide value) {
    if (value == 0) return "0";
    const bool negative = value < 0;
    if (negative) value = -value;
    std::string text;
    while (value) {
        text += static_cast<char>('0' + value % 10);
        value /= 10;
    }
    if (negative) text += '-';
    std::reverse(text.begin(), text.end());
    return text;
}

Wide read_wide(std::istream& input) {
    std::string literal;
    if (!(input >> literal) || literal.empty() || literal.size() > 37) {
        throw std::runtime_error("Invalid workbook numeric token");
    }
    Wide value = 0;
    for (char c : literal) {
        if (c < '0' || c > '9') throw std::runtime_error("Expected an unsigned integer");
        value = 10 * value + c - '0';
    }
    return value;
}

template <class T> T read_number(std::istream& input, T maximum = std::numeric_limits<T>::max()) {
    Wide value = read_wide(input);
    if (value > maximum) throw std::runtime_error("Workbook number outside audit domain");
    return static_cast<T>(value);
}

template <class T> std::vector<T> read_vector(std::istream& input, int n, T maximum = std::numeric_limits<T>::max()) {
    std::vector<T> values(n);
    for (auto& value : values) value = read_number<T>(input, maximum);
    return values;
}

struct Graph {
    int n, limit, nd, na, zero_targets = 0;
    I cap, people = 0;
    Wide exact_limit, quantum, initial_cutoff;
    std::vector<int> sites, ret;
    std::vector<I> supply, pop, offered;
    std::vector<Wide> exact_ret, start;
    std::array<std::array<int, Sites>, Sites> cost{};
    std::array<std::array<Wide, Sites>, Sites> exact_cost{};

    int grid(Wide value) const {
        return static_cast<int>(std::min(Wide(limit+1), value/quantum));
    }

    explicit Graph(std::istream& input) {
        const int count = read_number<int>(input, Sites);
        if (count < 3) throw std::runtime_error("Need a depot and two active sites");
        cap = read_number<I>(input);
        exact_limit = read_wide(input);
        quantum = read_wide(input);
        initial_cutoff = read_wide(input);
        if (!quantum || exact_limit/quantum > 750 || !cap) throw std::runtime_error("Invalid time grid or capacity");
        limit = static_cast<int>(exact_limit/quantum);
        auto q = read_vector<I>(input, count);
        auto p = read_vector<I>(input, count);
        std::vector<Wide> service(count);
        for (auto& time : service) time = read_wide(input);
        std::vector<std::vector<Wide>> travel(count, std::vector<Wide>(count));
        for (auto& row : travel) for (auto& time : row) time = read_wide(input);
        if (q[0] || p[0] || service[0]) throw std::runtime_error("Invalid depot");

        std::vector<int> suppliers, agencies;
        Wide total = 0, total_people = 0;
        for (int i = 1; i < count; ++i) {
            if ((q[i] > 0) == (p[i] > 0)) throw std::runtime_error("Invalid role");
            (q[i] ? suppliers : agencies).push_back(i);
            total += q[i];
            total_people += p[i];
        }
        nd = suppliers.size();
        na = agencies.size();
        n = nd + na;
        if (nd < 1 || nd > Donors || na < 1 || na > 62 || total > cap ||
            total_people*(limit+2) > std::numeric_limits<I>::max()/4) {
            throw std::runtime_error("Amounts, population, or role count outside the supported range");
        }
        people = static_cast<I>(total_people);
        std::vector<bool> forced(count);
        for (int v : agencies) {
            for (int u = 1; u < count; ++u) {
                if (u != v && (travel[u][v]+service[v])/quantum == 0) {
                    forced[v] = true;
                    ++zero_targets;
                    break;
                }
            }
        }
        std::sort(agencies.begin(), agencies.end(), [&](int a, int b) {
            if (forced[a] != forced[b]) return bool(forced[a]);
            if (p[a] != p[b]) return p[a] > p[b];
            return a < b;
        });
        auto order = suppliers;
        order.insert(order.end(), agencies.begin(), agencies.end());
        for (int i = 0; i < n; ++i) {
            const int a = order[i];
            sites.push_back(a);
            pop.push_back(p[a]);
            exact_ret.push_back(travel[a][0]);
            ret.push_back(grid(travel[a][0]));
            start.push_back(travel[0][a]+service[a]);
            supply.push_back(q[a]);
            for (int j = 0; j < n; ++j) {
                const int b = order[j];
                exact_cost[i][j] = travel[a][b]+service[b];
                cost[i][j] = grid(exact_cost[i][j]);
            }
        }
        offered.assign(1<<nd, 0);
        for (int mask = 1; mask < (1<<nd); ++mask) {
            offered[mask] = supply[std::countr_zero(static_cast<unsigned>(mask))] + offered[mask&(mask-1)];
        }
    }
};

struct Block {
    I people;
    I fresh;
};

// Conditional on visiting the particular agencies in order, each segment between
// supplier arrivals may share its aggregate delivery proportionally. Take the
// lowest average prefix among the remaining blocks. This chooses the breakpoints
// of nondecreasing rates. Every breakpoint uses exactly its cumulative supply.
Wide group_upper(const std::array<Block, Donors>& blocks, int length) {
    Wide score = 0;
    I tail_people = 0;
    for (int i = 0; i < length; ++i) tail_people += blocks[i].people;
    for (int i = 0; i < length;) {
        int end = i;
        I chosen_people = 0, pop = 0;
        I chosen_supply = 0, supply = 0;
        for (int j = i; j < length; ++j) {
            pop += blocks[j].people;
            supply += blocks[j].fresh;
            if (chosen_people == 0 || Wide(supply)*chosen_people <= Wide(chosen_supply)*pop) {
                chosen_people = pop;
                chosen_supply = supply;
                end = j;
            }
        }
        if (chosen_people <= 0) throw std::runtime_error("Empty allocation block");
        tail_people -= chosen_people;
        score += Wide(chosen_supply) * (chosen_people + 2*tail_people);
        i = end+1;
    }
    return score;
}

struct PrefixKey {
    Mask visited{};
    std::array<I, Donors-1> quantities{};
    std::array<I, Donors-1> populations{};
    int pending{}, last{}, breaks{};
    bool operator==(const PrefixKey&) const = default;
};

struct PrefixHash {
    std::size_t operator()(const PrefixKey& k) const {
        Mask v = k.visited;
        for (int i = 0; i < k.breaks; ++i) {
            v = (v ^ (Mask(k.quantities[i]) + Mask(8493453)*k.populations[i])) * 0x9e3779b97f4a7c15ULL;
        }
        return v ^ (0xc6a4a7935bd1e995ULL * (k.pending*Sites+k.last));
    }
};

struct Audit {
    const Graph& w;
    const int tracked, masks, times, donor_masks;
    const bool use_table;
    const std::size_t layer;
    const Clock::time_point deadline;
    const pid_t parent;
    std::vector<I> cover;
    std::vector<int> cardinality;
    std::vector<Mask> reachable;
    std::vector<int> least_arrival, density_order;
    std::vector<int> path;
    std::unordered_map<PrefixKey, Wide, PrefixHash> seen;
    std::uint64_t count=0, table_cut=0, reach_cut=0, equity_cut=0, dominance=0, allocations=0, ctx=0;
    Wide cutoff;

    Audit(const Graph& w, int k, Clock::time_point end, pid_t owner)
        : w(w), tracked(std::max(0,k)), masks(1<<tracked), times(w.limit+1), donor_masks(1<<w.nd),
          use_table(k>=0), layer(std::size_t(times)*w.n), deadline(end), parent(owner),
          cover(use_table?layer*masks*donor_masks:0,Impossible),
          cardinality(use_table && !w.zero_targets?layer*donor_masks:0,Impossible),
          reachable(layer*donor_masks), cutoff(w.initial_cutoff) {}

    void check_time() const {
        if (getppid()!=parent) throw Disconnected{};
        if (Clock::now() >= deadline) throw Expired{};
    }

    int tag(int site) const {
        const int pos = site-w.nd;
        return pos < tracked ? (1<<pos) : 0;
    }

    std::size_t at(int budget, int last) const {
        return std::size_t(budget)*w.n+last;
    }

    I value(int tracked_mask, int pending, int budget, int last) const {
        return use_table?cover[(tracked_mask*donor_masks+pending)*layer+at(budget,last)]:w.people;
    }

    void prepare_walks() {
        if (!use_table) return;
        // Use original arcs, and permit repeats only for untracked agencies.
        // Future suppliers must be exactly the pending compulsory suppliers.
        for (int a = masks-1; a >= 0; --a) {
            for (int d = 0; d < donor_masks; ++d) {
                check_time();
                for (int t = 0; t < times; ++t) {
                    for (int from = 0; from < w.n; ++from) {
                        I best = !d && from>=w.nd && w.ret[from]<=t ? 0 : Impossible;
                        int stops = static_cast<int>(best);
                        for (int to = 0; to < w.n; ++to) {
                            const int move = w.cost[from][to];
                            if (from == to || move > t) continue;
                            int next_d=d, next_a=a;
                            if (to < w.nd) {
                                const int donor = 1<<to;
                                if (!(d & donor)) continue;
                                next_d ^= donor;
                            } else {
                                const int agency = tag(to);
                                if (a & agency) continue;
                                next_a |= agency;
                            }
                            I suffix = value(next_a,next_d,t-move,to);
                            if (suffix >= 0) best = std::max(best,suffix+w.pop[to]);
                            if (a == 0 && !cardinality.empty()) {
                                suffix = cardinality[next_d*layer+at(t-move,to)];
                                if (suffix >= 0) stops = std::max(stops,static_cast<int>(suffix)+(to>=w.nd));
                            }
                        }
                        cover[(a*donor_masks+d)*layer+at(t,from)] = best;
                        if (a==0 && !cardinality.empty()) cardinality[d*layer+at(t,from)] = stops;
                    }
                }
            }
        }
    }

    void prepare_reach() {
        auto dist=w.cost;
        for (int i=0;i<w.n;++i) dist[i][i]=0;
        for (int k=0;k<w.n;++k) for(int i=0;i<w.n;++i) for(int j=0;j<w.n;++j) {
            dist[i][j]=std::min(dist[i][j],dist[i][k]+dist[k][j]);
        }
        auto back=w.ret;
        for(int i=0;i<w.n;++i) for(int j=0;j<w.n;++j) back[i]=std::min(back[i],dist[i][j]+w.ret[j]);
        for (int agency=w.nd; agency<w.n; ++agency) {
            check_time();
            // Lower bound for visiting all pending donors plus the named agency.
            std::array<std::array<int, Sites>, 1<<(Donors+1)> distance{};
            for (int from=0;from<w.n;++from) distance[0][from]=back[from];
            for (int mask=1; mask<2*donor_masks; ++mask) {
                for (int from=0;from<w.n;++from) {
                    int best=Forever;
                    for (unsigned rest=mask;rest;rest&=rest-1) {
                        int bit=std::countr_zero(rest);
                        int to=bit==w.nd?agency:bit;
                        best=std::min(best,dist[from][to]+distance[mask^(1<<bit)][to]);
                    }
                    distance[mask][from]=best;
                }
            }
            for(int d=0;d<donor_masks;++d) for(int from=0;from<w.n;++from) {
                const int minimum=distance[d|donor_masks][from];
                for(int t=minimum;t<times;++t) reachable[d*layer+at(t,from)] |= Mask{1}<<(agency-w.nd);
            }
        }
        least_arrival.assign(w.n, Forever);
        for (int to=0;to<w.n;++to) {
            for (int from=0;from<w.n;++from) {
                if (from != to) least_arrival[to]=std::min(least_arrival[to],w.cost[from][to]);
            }
            if (to>=w.nd) density_order.push_back(to);
        }
        std::sort(density_order.begin(),density_order.end(),[&](int a,int b) {
            return Wide(w.pop[a])*least_arrival[b]>Wide(w.pop[b])*least_arrival[a];
        });
    }

    I travel_population_upper(Mask agencies, int stops, int time, int pending) const {
        I best=0;
        int min_return=Forever;
        for (Mask set=agencies;set;set&=set-1) {
            int v=w.nd+std::countr_zero(set);
            if (stops>0) best+=w.pop[v];
            --stops;
            min_return=std::min(min_return,w.ret[v]);
        }
        if (best==0) return 0;
        for (unsigned rest=pending;rest;rest&=rest-1) time-=least_arrival[std::countr_zero(rest)];
        time-=min_return;
        if (time < 0) return 0;
        I cap=0;
        for (int v:density_order) {
            if (!(agencies&(Mask{1}<<(v-w.nd)))) continue;
            if (time<least_arrival[v]) {
                cap+=static_cast<I>(Wide(w.pop[v])*time/least_arrival[v]);
                break;
            }
            time-=least_arrival[v];
            cap+=w.pop[v];
        }
        return std::min(best,cap);
    }

    Wide fairness(const PrefixKey& state, I population_upper, I total) const {
        std::array<Block,Donors> blocks{};
        I people=0;
        I amount=0;
        for (int i=0;i<state.breaks;++i) {
            blocks[i]={state.populations[i]-people,state.quantities[i]-amount};
            people=state.populations[i];
            amount=state.quantities[i];
        }
        if (population_upper <= people) {
            // With no agency after the last recorded supply breakpoint, any
            // further supply would be unusable. Discard it from the bound.
            return state.breaks ? group_upper(blocks,state.breaks) : 0;
        }
        blocks[state.breaks]={population_upper-people,total-amount};
        return group_upper(blocks,state.breaks+1);
    }

    void propose(Wide value) {
        check_time();
        std::cout << "candidate " << decimal(value);
        for (int site : path) std::cout << ' ' << w.sites[site];
        std::cout << std::endl;
        std::string approval;
        if (!(std::cin >> approval) || approval != "approve") throw Disconnected{};
        Wide granted = read_wide(std::cin);
        if (granted != value || granted <= cutoff) throw std::runtime_error("Invalid witness acknowledgement");
        check_time();
        cutoff = granted;
    }

    void search(PrefixKey state,int tracked_mask,Wide exact_left,I population,I total) {
        if ((++count&0x3fff)==0) check_time();
        const int from=state.last, pending=state.pending, time=w.grid(exact_left);
        const auto point=at(time,from);
        if (!pending && from>=w.nd && w.exact_ret[from]<=exact_left && Wide(total)*population>cutoff) {
            ++allocations;
            const Wide bound=fairness(state,population,total);
            if (bound>cutoff) propose(bound);
        }
        const I future=value(tracked_mask,pending,time,from);
        if (future<0 || Wide(total)*(population+future)<=cutoff) {
            ++table_cut;
            return;
        }
        const Mask agencies=reachable[pending*layer+point]&~state.visited;
        const int stops=cardinality.empty()?w.na:cardinality[pending*layer+point];
        const I geo=travel_population_upper(agencies,stops,time,pending);
        const I possible=population+std::min(geo,future);
        if (Wide(total)*possible<=cutoff) {
            ++reach_cut;
            return;
        }
        if (state.breaks || (pending && population)) {
            PrefixKey relaxed=state;
            const I last_cut=state.breaks?state.populations[state.breaks-1]:0;
            if (pending && population>last_cut) {
                relaxed.populations[relaxed.breaks]=population;
                relaxed.quantities[relaxed.breaks]=total-w.offered[pending];
                ++relaxed.breaks;
            }
            if (fairness(relaxed,possible,total)<=cutoff) {
                ++equity_cut;
                return;
            }
        }
        if (seen.size()<1500000) {
            auto [prior,inserted]=seen.try_emplace(state,exact_left);
            if (!inserted) {
                if (prior->second>=exact_left) {
                    ++dominance;
                    return;
                }
                prior->second=exact_left;
            }
        }
        for (unsigned choices=pending;choices;choices&=choices-1) {
            const int v=std::countr_zero(choices);
            const Wide cost=w.exact_cost[from][v];
            if (cost>exact_left) continue;
            PrefixKey next=state;
            next.pending^=1<<v;
            next.last=v;
            const I last_cut=state.breaks?state.populations[state.breaks-1]:0;
            if (population>last_cut) {
                next.populations[next.breaks]=population;
                next.quantities[next.breaks]=total-w.offered[pending];
                ++next.breaks;
            }
            path.push_back(v);
            search(next,tracked_mask,exact_left-cost,population,total);
            path.pop_back();
        }
        for (Mask choices=agencies; choices; choices&=choices-1) {
            const int v=w.nd+std::countr_zero(choices);
            const Wide cost=w.exact_cost[from][v];
            if (cost>exact_left) continue;
            PrefixKey next=state;
            next.last=v;
            next.visited|=Mask{1}<<(v-w.nd);
            path.push_back(v);
            search(next,tracked_mask|tag(v),exact_left-cost,population+w.pop[v],total);
            path.pop_back();
        }
    }
};

struct Context {
    int first, pending;
    Wide left, upper;
    I supply;
};

int main(int argc,char** argv) {
    try {
        if (argc!=4) throw std::runtime_error("Usage: chain_oracle model-file tracked-count seconds");
        const auto start_time=Clock::now();
        const pid_t owner=getppid();
        if (owner<=1) return 0;
        std::ifstream input(argv[1]);
        Graph graph(input);
        const int request=std::stoi(argv[2]);
        const double duration=std::stod(argv[3]);
        if (request<0 || request>7 || !(duration>0 && duration<=3600)) {
            throw std::runtime_error("Invalid tracked count or runtime");
        }
        const auto deadline=start_time+std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double>(duration));
        const std::size_t plane=std::size_t(graph.limit+1)*graph.n*(1<<graph.nd);
        int tracked=std::min(graph.na,std::max(request,graph.zero_targets));
        while (tracked>7 || plane*(std::size_t(1)<<tracked)>24000000) --tracked;
        if (tracked<graph.zero_targets) tracked=-1;
        Audit proof(graph,tracked,deadline,owner);
        std::cout << "domain " << std::max(0,tracked) << ' ' << proof.use_table << std::endl;
        try {
            proof.prepare_walks();
            proof.prepare_reach();
            std::vector<Context> queue;
            for (int subset=1;subset<(1<<graph.nd);++subset) for (int first=0;first<graph.nd;++first) {
                if (!(subset&(1<<first)) || graph.start[first]>graph.exact_limit) continue;
                const int pending=subset^(1<<first);
                const Wide left=graph.exact_limit-graph.start[first];
                const I potential=proof.value(0,pending,graph.grid(left),first);
                const I supply=graph.offered[subset];
                if (potential>=0) queue.push_back({first,pending,left,Wide(supply)*potential,supply});
            }
            std::stable_sort(queue.begin(),queue.end(),
                             [](const Context& a,const Context& b) { return a.upper>b.upper; });
            auto stats=[&] {
                std::cout << ' ' << proof.count << ' ' << proof.ctx << ' ' << proof.table_cut
                          << ' ' << proof.reach_cut << ' ' << proof.equity_cut << ' ' << proof.dominance
                          << ' ' << proof.allocations << std::endl;
            };
            for (const auto& context : queue) {
                proof.check_time();
                if (context.upper<=proof.cutoff) break;
                std::cout << "bound " << decimal(context.upper);
                stats();
                proof.seen.clear();
                proof.path={context.first};
                ++proof.ctx;
                PrefixKey root{};
                root.pending=context.pending;
                root.last=context.first;
                proof.search(root,0,context.left,0,context.supply);
            }
            proof.check_time();
            std::cout << "done " << decimal(proof.cutoff);
            stats();
        } catch (const Expired&) {
            std::cout << "timeout" << std::endl;
        } catch (const Disconnected&) {
            return 0;
        }
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << '\n';
        return 2;
    }
}
