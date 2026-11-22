// Independent binding-capacity proof for two suppliers and their two delivery blocks.
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using I = std::int64_t;
using Wide = __int128_t;
using Mask = std::uint64_t;
using Clock = std::chrono::steady_clock;
constexpr int Impossible = -1;
constexpr int Forever = 1000000;
constexpr int Donors = 2;
constexpr int Sites = 67;

struct Expired {};
struct Better {};

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

Wide read_wide() {
    std::string literal;
    if (!(std::cin >> literal) || literal.empty() || literal.size() > 33) {
        throw std::runtime_error("Invalid workbook numeric token");
    }
    Wide value = 0;
    for (char c : literal) {
        if (c < '0' || c > '9') throw std::runtime_error("Expected an unsigned integer");
        value = 10 * value + c - '0';
    }
    return value;
}

template <class T> T read_number(T maximum = std::numeric_limits<T>::max()) {
    Wide value = read_wide();
    if (value > maximum) throw std::runtime_error("Workbook number outside audit domain");
    return static_cast<T>(value);
}

template <class T> std::vector<T> read_vector(int n, T maximum = std::numeric_limits<T>::max()) {
    std::vector<T> values(n);
    for (auto& value : values) value = read_number<T>(maximum);
    return values;
}

struct Workbook {
    int n, limit, nd, na;
    I cap;
    Wide cutoff;
    std::vector<int> sites, service, pop, ret, start;
    std::vector<I> supply;
    std::array<std::array<int, Sites>, Sites> cost{};
    std::vector<I> offered;

    Workbook() {
        int count = read_number<int>(Sites);
        if (count < 3) throw std::runtime_error("Audit requires a depot and two active stops");
        cap = read_number<I>();
        limit = read_number<int>(750);
        cutoff = read_wide();
        auto ids = read_vector<int>(count);
        auto q = read_vector<I>(count);
        auto p = read_vector<int>(count, 50000);
        auto t = read_vector<int>(count, 3000);
        std::vector<std::vector<int>> travel;
        for (int i = 0; i < count; ++i) travel.push_back(read_vector<int>(count, 3000));
        if (ids[0] || q[0] || p[0] || t[0] || cap == 0) throw std::runtime_error("Invalid depot");

        std::vector<int> suppliers, agencies;
        Wide total = 0;
        for (int i = 1; i < count; ++i) {
            if (t[i] <= 0 || (q[i] > 0) == (p[i] > 0)) throw std::runtime_error("Invalid role");
            (q[i] ? suppliers : agencies).push_back(i);
            total += q[i];
        }
        nd = suppliers.size();
        na = agencies.size();
        n = nd + na;
        if (nd != Donors || na < 1 || na > 62 || total > std::numeric_limits<I>::max()) {
            throw std::runtime_error("Audit requires exactly two donors and signed 64-bit total offered supply");
        }
        std::sort(agencies.begin(), agencies.end(), [&](int a, int b) {
            if (p[a] != p[b]) return p[a] > p[b];
            return ids[a] < ids[b];
        });
        auto order = suppliers;
        order.insert(order.end(), agencies.begin(), agencies.end());
        for (int i = 0; i < n; ++i) {
            int a = order[i];
            sites.push_back(ids[a]);
            service.push_back(t[a]);
            pop.push_back(p[a]);
            ret.push_back(travel[a][0]);
            start.push_back(travel[0][a]+t[a]);
            supply.push_back(std::min(q[a], cap));
            for (int j = 0; j < n; ++j) cost[i][j] = travel[a][order[j]]+t[order[j]];
        }
        offered.assign(1 << nd, 0);
        for (int mask = 1; mask < (1 << nd); ++mask) {
            offered[mask] = supply[std::countr_zero(static_cast<unsigned>(mask))] + offered[mask&(mask-1)];
        }
    }
};

struct PrefixKey {
    Mask visited{};
    int before_second = -1;
    int pending{}, last{};
    bool operator==(const PrefixKey&) const = default;
};

struct PrefixHash {
    std::size_t operator()(const PrefixKey& key) const {
        const Mask mixed = Mask(key.before_second + 1)*0x9e3779b97f4a7c15ULL;
        const Mask ending = Mask(key.pending*Sites+key.last)*0xc6a4a7935bd1e995ULL;
        return mixed^ending^key.visited;
    }
};

// Once the second donor has been visited, the first cohort is fixed.
// Lines bound ideal equity, first-load feasibility, and second-load feasibility.
Wide fixed_order_upper(int before, int after, I first, I offered, I second_max) {
    const int people = before + after;
    const Wide ideal = Wide(offered)*people;
    const Wide early = Wide(first)*before + (Wide(offered)+first)*after;
    const Wide late = (Wide(offered)+second_max)*before + Wide(second_max)*after;
    return std::min({ideal,early,late});
}

struct Audit {
    const Workbook& w;
    const int tracked, masks, times, donor_masks;
    const std::size_t layer;
    const Clock::time_point deadline;
    std::vector<int> cover;
    std::vector<int> cardinality;
    std::vector<Mask> reachable;
    std::vector<int> least_arrival, density_order;
    std::vector<int> path;
    std::unordered_map<PrefixKey, int, PrefixHash> seen;
    std::uint64_t count=0, table_cut=0, reach_cut=0, equity_cut=0, dominance=0, allocations=0, ctx=0;
    int winning_pop=0, winning_time=0;
    I first_supply=0;
    Wide winning_score=0;
    Wide root_upper=0;

    Audit(const Workbook& w, int k, Clock::time_point end)
        : w(w), tracked(k), masks(1<<k), times(w.limit+1), donor_masks(1<<w.nd),
          layer(std::size_t(times)*w.n), deadline(end), cover(layer*masks*donor_masks, Impossible),
          cardinality(layer*donor_masks, Impossible), reachable(layer*donor_masks) {}

    void check_time() const {
        if (Clock::now() >= deadline) throw Expired{};
    }

    int tag(int site) const {
        const int pos = site-w.nd;
        return pos < tracked ? (1<<pos) : 0;
    }

    std::size_t at(int budget, int last) const {
        return std::size_t(budget)*w.n+last;
    }

    int value(int tracked_mask, int pending, int budget, int last) const {
        return cover[(tracked_mask*donor_masks+pending)*layer+at(budget,last)];
    }

    void prepare_walks() {
        // Use original arcs, and permit repeats only for untracked agencies.
        // Future suppliers must be exactly the pending compulsory suppliers.
        for (int a = masks-1; a >= 0; --a) {
            for (int d = 0; d < donor_masks; ++d) {
                check_time();
                for (int t = 0; t < times; ++t) {
                    for (int from = 0; from < w.n; ++from) {
                        int best = !d && from>=w.nd && w.ret[from]<=t ? 0 : Impossible;
                        int stops = best;
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
                            int suffix = value(next_a,next_d,t-move,to);
                            if (suffix >= 0) best = std::max(best,suffix+w.pop[to]);
                            if (a == 0) {
                                suffix = cardinality[next_d*layer+at(t-move,to)];
                                if (suffix >= 0) stops = std::max(stops,suffix+(to>=w.nd));
                            }
                        }
                        cover[(a*donor_masks+d)*layer+at(t,from)] = best;
                        if (a==0) cardinality[d*layer+at(t,from)] = stops;
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
            return I(w.pop[a])*least_arrival[b]>I(w.pop[b])*least_arrival[a];
        });
    }

    int travel_population_upper(Mask agencies, int stops, int time, int pending) const {
        int best=0, min_return=Forever;
        for (Mask set=agencies;set;set&=set-1) {
            int v=w.nd+std::countr_zero(set);
            if (stops>0) best+=w.pop[v];
            --stops;
            min_return=std::min(min_return,w.ret[v]);
        }
        if (best==0) return 0;
        for (unsigned rest=pending;rest;rest&=rest-1) time-=least_arrival[std::countr_zero(rest)];
        time-=min_return;
        if (time <= 0) return 0;
        int cap=0;
        for (int v:density_order) {
            if (!(agencies&(Mask{1}<<(v-w.nd)))) continue;
            if (time<least_arrival[v]) {
                cap+=I(w.pop[v])*time/least_arrival[v];
                break;
            }
            time-=least_arrival[v];
            cap+=w.pop[v];
        }
        return std::min(best,cap);
    }

    Wide fairness(const PrefixKey& state, int present, int people, I offered) const {
        if (!state.pending) {
            if (state.before_second < 0) return Wide(offered)*people;
            return fixed_order_upper(state.before_second,people-state.before_second,
                                     first_supply,offered,std::min(offered,w.cap));
        }
        const I second_max = std::min(offered,w.cap);
        // Before the second pickup, the first cohort can only grow from the
        // already visited population. The two affine limits have opposite
        // slopes at fixed total population. Their mean is another upper bound.
        const Wide ideal = Wide(offered)*people;
        const Wide middle = (Wide(offered)+first_supply+second_max)*people/2;
        const Wide first_cohort = (Wide(offered)+first_supply)*people - Wide(offered)*present;
        return std::min({ideal,middle,first_cohort});
    }

    void search(PrefixKey state,int tracked_mask,int time,int population,I total) {
        if ((++count&0x3fff)==0) check_time();
        const int from=state.last, pending=state.pending;
        const auto point=at(time,from);
        if (!pending && from>=w.nd && w.ret[from]<=time && Wide(total)*population>w.cutoff) {
            ++allocations;
            const Wide bound=fairness(state,population,population,total);
            if (bound>w.cutoff) {
                winning_score=bound;
                winning_pop=population;
                winning_time=w.limit-time+w.ret[from];
                throw Better{};
            }
        }
        const int future=value(tracked_mask,pending,time,from);
        if (future<0 || Wide(total)*(population+future)<=w.cutoff) {
            ++table_cut;
            return;
        }
        const Mask agencies=reachable[pending*layer+point]&~state.visited;
        const int geo=travel_population_upper(agencies,cardinality[pending*layer+point],time,pending);
        const int possible=population+std::min(geo,future);
        if (Wide(total)*possible<=w.cutoff) {
            ++reach_cut;
            return;
        }
        if (fairness(state,population,possible,total)<=w.cutoff) {
            ++equity_cut;
            return;
        }
        if (seen.size()<1500000) {
            auto [prior,inserted]=seen.try_emplace(state,time);
            if (!inserted) {
                if (prior->second>=time) {
                    ++dominance;
                    return;
                }
                prior->second=time;
            }
        }
        for (unsigned choices=pending;choices;choices&=choices-1) {
            const int v=std::countr_zero(choices);
            const int cost=w.cost[from][v];
            if (cost>time) continue;
            PrefixKey next=state;
            next.pending^=1<<v;
            next.last=v;
            next.before_second=population;
            path.push_back(v);
            search(next,tracked_mask,time-cost,population,total);
            path.pop_back();
        }
        for (Mask choices=agencies; choices; choices&=choices-1) {
            const int v=w.nd+std::countr_zero(choices), cost=w.cost[from][v];
            if (cost>time) continue;
            PrefixKey next=state;
            next.last=v;
            next.visited|=Mask{1}<<(v-w.nd);
            path.push_back(v);
            search(next,tracked_mask|tag(v),time-cost,population+w.pop[v],total);
            path.pop_back();
        }
    }
};

int main(int argc,char** argv) {
    try {
        if (argc!=3) throw std::runtime_error("Usage: certify tracked-count seconds < workbook.txt");
        Workbook w;
        const int tracked=std::stoi(argv[1]);
        const double budget=std::stod(argv[2]);
        if (tracked<0 || tracked>7 || tracked>w.na || !(budget>0 && budget<=3600)) {
            throw std::runtime_error("Invalid tracked count or runtime");
        }
        const auto start=Clock::now();
        const auto end=start+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(budget));
        const auto elapsed=[&] { return std::chrono::duration<double>(Clock::now()-start).count(); };
        Audit proof(w,tracked,end);
        std::cout << "target_numerator " << decimal(w.cutoff) << "\ntracked " << tracked << std::endl;
        try {
            proof.prepare_walks();
            proof.prepare_reach();
            std::cout << "table_seconds " << elapsed() << std::endl;
            for (int subset=1;subset<(1<<w.nd);++subset) for (int first=0;first<w.nd;++first) {
                if (!(subset&(1<<first)) || w.start[first]>w.limit) continue;
                int pending=subset^(1<<first), time=w.limit-w.start[first];
                int population=proof.value(0,pending,time,first);
                proof.root_upper=std::max(proof.root_upper,Wide(w.offered[subset])*std::max(population,0));
                if (population>=0) {
                    std::cout << "root " << subset << ' ' << w.sites[first] << ' ' << population
                              << ' ' << decimal(Wide(w.offered[subset])*population) << '\n';
                }
            }
            std::cout << "initial_upper_numerator " << decimal(proof.root_upper) << std::endl;
            for (int subset=(1<<w.nd)-1;subset>0;--subset) for (int first=0;first<w.nd;++first) {
                proof.check_time();
                if (!(subset&(1<<first)) || w.start[first]>w.limit) continue;
                const int pending=subset^(1<<first), time=w.limit-w.start[first];
                const int population=proof.value(0,pending,time,first);
                if (population<0 || Wide(w.offered[subset])*population<=w.cutoff) continue;
                proof.seen.clear();
                proof.path={first};
                proof.first_supply=w.supply[first];
                ++proof.ctx;
                const auto earlier=proof.count;
                PrefixKey root{};
                root.pending=pending;
                root.last=first;
                proof.search(root,0,time,0,w.offered[subset]);
                std::cout << "covered_context " << subset << ' ' << w.sites[first] << ' '
                          << proof.count-earlier << std::endl;
            }
            proof.check_time();
            std::cout << "status certified\nstrict_upper_numerator " << decimal(w.cutoff) << '\n';
        } catch (const Expired&) {
            std::cout << "status time_limit\n";
        } catch (const Better&) {
            std::cout << "status counterexample\nscore_numerator " << decimal(proof.winning_score)
                      << "\ncovered " << proof.winning_pop << "\ntime " << proof.winning_time
                      << "\nsource_route 0";
            for (int v:proof.path) std::cout << ' ' << w.sites[v];
            std::cout << " 0\n";
        }
        std::cout << "contexts_opened " << proof.ctx << "\nexpanded " << proof.count
                  << "\nraw_table_cut " << proof.table_cut << "\ntime_population_cut " << proof.reach_cut
                  << "\nequity_cut " << proof.equity_cut << "\ndominance " << proof.dominance
                  << "\nfull_allocation_checks " << proof.allocations << "\nseconds " << elapsed() << '\n';
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << '\n';
        return 2;
    }
}
