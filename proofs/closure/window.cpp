// Keep the independently audited reverse-direction rules and add bounded
// masks, a population-only relaxation, and a two-level residual frontier.
#define main base_audit_main
#include "reverse.cpp"
#undef main

using Mask = uint64_t;
constexpr int table_limit=3;
constexpr Mask site_bit(int v) { return Mask{1}<<v; }

struct Summary {
    Int stock=0,pop=0;
    int agencies=0,donors=0;
    std::array<int,table_limit> track{};
    std::array<Int,table_limit> prices{};

    void include(const Summary& rhs) {
        stock+=rhs.stock; pop+=rhs.pop; agencies+=rhs.agencies; donors|=rhs.donors;
        for (int i=0;i<table_limit;++i) {
            track[i]|=rhs.track[i]; prices[i]+=rhs.prices[i];
        }
    }
};

struct WindowCertificate : Certificate {
    using Certificate::Certificate;
    std::vector<Mask> reachable;
    std::array<std::array<Summary,256>,8> pieces{};
    uint64_t population_cuts=0;
    uint64_t tasks_closed=0,tasks_open=0;
    int task_first=0,task_second=0;

    void population_table(int tracked) {
        std::vector<int> candidates=p.by_pop;
        std::sort(candidates.begin(),candidates.end(),[&](int a,int b){
            return std::pair{p.population[a],a}>std::pair{p.population[b],b};
        });
        std::array<Int,sites> prices{};
        Int pivot=p.population[candidates[candidates.size()/2]];
        for (int v:candidates) prices[v]=std::max<Int>(0,p.population[v]-pivot);
        candidates.resize(std::min<size_t>(tracked,candidates.size()));
        tables.emplace_back(p,0,1,candidates,prices);
        if (!tables.back().fill(end)) {tables.pop_back(); expired=true;}
    }

    void build_summaries() {
        const int stride=p.limit+1;
        reachable.resize(size_t(p.n)*stride);
        for (int last=1;last<p.n;++last) {
            for (int v=1;v<p.n;++v) {
                int time=p.shortest[last][v]+p.shortest[v][0];
                for (int t=time;t<=p.limit;++t) reachable[size_t(last)*stride+t]|=site_bit(v);
            }
        }
        std::array<Summary,64> values{};
        for (int v=1;v<p.n;++v) {
            auto& item=values[v];
            item.stock=p.supply[v]; item.pop=p.population[v]; item.agencies=p.population[v]>0;
            item.donors=donor_bit[v];
            for (size_t i=0;i<tables.size();++i) {
                item.track[i]=tables[i].bit[v]; item.prices[i]=tables[i].price[v];
            }
        }
        for (size_t block=0;block<pieces.size();++block) {
            for (unsigned mask=1;mask<256;++mask) {
                auto& row=pieces[block][mask];
                row=pieces[block][mask&(mask-1)];
                row.include(values[8*block+__builtin_ctz(mask)]);
            }
        }
    }

    Summary summarize(Mask mask) const {
        Summary value;
        for (size_t block=0;mask;++block,mask>>=8) {
            value.include(pieces[block][mask&255]);
        }
        return value;
    }

    Int packing(Mask mask,int time,const std::vector<int>& order,
                const std::array<Int,sites>& value,const std::array<int,sites>& charge) const {
        Int result=0;
        for (int v:order) if (mask&site_bit(v)) {
            if (charge[v]>time) return result+(time*value[v])/charge[v];
            result+=value[v];
            time-=charge[v];
        }
        return result;
    }

    bool could_improve(Int total,Int a,Int b,Int supply,Int pop) const {
        const Wide required=Wide(p.goal)+1;
        if (Wide(total)*total<4*Wide(a)*b*required) return false;
        if (2*Wide(a)*supply<=total && Wide(supply)*(total-Wide(a)*supply)<b*required)
            return false;
        if (2*Wide(b)*pop<=total && Wide(pop)*(total-Wide(b)*pop)<a*required)
            return false;
        return true;
    }

    bool splits_closed(Mask mask,const std::array<Int,table_limit>& totals,size_t key,
                       int lo,int hi,int rem,Int covered,Int served,Int flow_cap,Int pop_cap) const {
        Int extra=donor_packing[key+hi],flow=std::min(flow_cap,served+extra);
        Int pop=std::min(pop_cap,covered+packing(mask,rem-lo,served_first,p.population,p.service));
        if (std::min(flow*pop,flow*covered+extra*pop)<=p.goal) return true;
        for (size_t i=0;i<tables.size();++i) {
            const auto& table=tables[i];
            if (table.a && !could_improve(totals[i],table.a,table.b,flow,pop)) return true;
        }
        if (lo==hi) return false;
        int mid=(lo+hi)/2;
        return splits_closed(mask,totals,key,lo,mid,rem,covered,served,flow_cap,pop_cap) &&
               splits_closed(mask,totals,key,mid+1,hi,rem,covered,served,flow_cap,pop_cap);
    }

    Int active_bound(Mask seen,int last,int previous,int rem,Int served,Int load,Int covered,
                     Int parent,bool key=false) {
        ++bounds;
        int available=rem-p.away[last]-p.into[0];
        if (available<0 || p.shortest[last][0]>rem) return impossible;
        Mask future=reachable[size_t(last)*(p.limit+1)+rem]&~seen;
        auto candidates=summarize(future);
        size_t donor_index=donor_masks ? donor_key(last,candidates.donors)*(p.limit+1) : 0;
        Int extra_stock=candidates.stock;
        if (donor_masks) extra_stock=std::min<Int>(extra_stock,donor_packing[donor_index+rem]);
        Int flow=served+std::min(extra_stock,load+p.capacity*candidates.agencies);
        Int pop=covered+candidates.pop;
        Int upper=std::min({parent,flow*pop,flow*covered+extra_stock*pop});
        if (upper<=p.goal) return upper;

        std::array<Int,table_limit> totals{};
        for (size_t i=0;i<tables.size();++i) {
            const auto& t=tables[i];
            Int z=t.get(rem,candidates.track[i],last,previous).first;
            if (z<0) return impossible;
            totals[i]=t.a*served+t.b*covered+z+candidates.prices[i];
            if (!t.a) {
                pop=std::min(pop,totals[i]);
                upper=std::min(upper,flow*pop);
                if (upper<=p.goal) {++population_cuts; return upper;}
            } else if (!could_improve(totals[i],t.a,t.b,flow,pop)) {
                ++table_rejections;
                return p.goal;
            } else if (key) {
                upper=balance_bound(upper,totals[i],t.a,t.b,flow,pop);
            }
        }
        extra_stock=std::min(extra_stock,packing(future,available,p.by_stock,p.supply,p.charge));
        pop=std::min(pop,covered+packing(future,available,p.by_pop,p.population,p.charge));
        flow=std::min(flow,served+extra_stock);
        upper=std::min({upper,flow*pop,flow*covered+extra_stock*pop});
        if (upper<=p.goal) return upper;
        for (size_t i=0;i<tables.size();++i) {
            const auto& t=tables[i];
            if (!t.a) continue;
            if (!could_improve(totals[i],t.a,t.b,flow,pop)) return p.goal;
            if (key) upper=balance_bound(upper,totals[i],t.a,t.b,flow,pop);
        }
        if (donor_masks && splits_closed(future,totals,donor_index,0,rem,rem,covered,served,flow,pop)) {
            ++donor_time_cuts;
            return p.goal;
        }
        return upper;
    }

    struct State {
        int last,previous,rem;
        Int served,load,covered,upper;
        Mask seen;
    };

    State step(const State& node,int v,bool key) {
        const Int dropped=std::min(node.load,p.supply[v]);
        const Int load=p.population[v] ? p.capacity : node.load-dropped;
        const Int covered=node.covered+p.population[v];
        int rem=node.rem-p.cost[node.last][v];
        Mask seen=node.seen|site_bit(v);
        const Int upper=active_bound(seen,v,node.last,rem,node.served+dropped,load,covered,node.upper,key);
        return {v,node.last,rem,node.served+dropped,load,covered,upper,seen};
    }

    void explore(const State& n) {
        if ((++nodes&127)==0 && Clock::now()>=end) {expired=true; return;}
        if (p.supply[n.last] && p.cost[n.last][0]<=n.rem && n.served*n.covered>p.goal) {
            ++leaves;
            Int score=exact_order(n.covered);
            highest_route=std::max(highest_route,score);
            if (score>p.goal) {apparent_counterexample.assign(route.rbegin(),route.rend()); return;}
        }
        for (int v:p.next[n.last]) {
            if (expired || !apparent_counterexample.empty()) return;
            if (p.cost[n.last][v]>n.rem) break;
            if (n.seen&site_bit(v)) continue;
            if (p.shortest[v][0]+p.cost[n.last][v]>n.rem) {++time_rejections; continue;}
            auto child=step(n,v,false);
            if (child.upper>p.goal) {
                route.push_back(v);
                explore(child);
                route.pop_back();
            }
        }
    }

    // External multipliers only suggest relaxations. Rebuild each complete
    // resource table with integer arithmetic on the original arc data.
    void read_tables() {
        int count;
        if (!(std::cin >> count) || count < 1 || count > table_limit)
            throw std::runtime_error("Invalid number of supplied tables");
        prepare_donors();
        for (int i = 0; i < count; ++i) {
            int a, b, k;
            if (!(std::cin >> a >> b >> k) || a < 1 || a > 10000 ||
                b < 1 || b > 10000 || k < 0 || k > 9)
                throw std::runtime_error("Invalid supplied table dimensions");
            std::vector<int> tracked(k);
            Mask seen = 0;
            for (int& v : tracked) {
                if (!(std::cin >> v) || v < 1 || v >= p.n || (seen & site_bit(v)))
                    throw std::runtime_error("Invalid tracked station");
                seen |= site_bit(v);
            }
            std::array<Int,sites> prices{};
            for (int v = 1; v < p.n; ++v) {
                if (!(std::cin >> prices[v]) || prices[v] < 0 ||
                    prices[v] > a*p.supply[v] + b*p.population[v])
                    throw std::runtime_error("Invalid station price");
            }
            tables.emplace_back(p, a, b, tracked, prices);
            if (!tables.back().fill(end)) { expired = true; return; }
        }
    }

    void verify(int tracked,int pop_track,bool supplied) {
        upper=p.stock*p.people;
        if (supplied) read_tables();
        else {
            prepare(tracked);
            if (expired) return;
            population_table(pop_track);
        }
        if (expired) return;
        build_summaries();
        std::vector<State> tasks;
        for (int v=1;v<p.n;++v) if (p.population[v] && p.cost[0][v]<=p.limit) {
            if (Clock::now()>=end) {expired=true; return;}
            State initial{v,0,p.limit-p.cost[0][v],0,p.capacity,p.population[v],upper,site_bit(v)};
            initial.upper=active_bound(initial.seen,v,0,initial.rem,0,p.capacity,initial.covered,upper,true);
            if (initial.upper<=p.goal) {++root_closed; continue;}
            bool any=false;
            for (int w:p.next[v]) {
                if (p.cost[v][w]>initial.rem) break;
                auto child=step(initial,w,true);
                if (child.upper>p.goal) {tasks.push_back(child); any=true;}
            }
            if (any) ++root_unresolved; else ++root_closed;
        }
        std::sort(tasks.begin(),tasks.end(),[](const State& a,const State& b){
            return std::tuple{a.upper,a.rem,a.previous,a.last}>std::tuple{b.upper,b.rem,b.previous,b.last};
        });
        tasks_open=tasks.size();
        std::cout<<"formed_tasks "<<tasks_open<<'\n'<<std::flush;
        auto last_report=Clock::now();
        for (const State& n:tasks) {
            active_task(n);
            if (Clock::now()>=end) {expired=true; return;}
            explore(n);
            if (expired || !apparent_counterexample.empty()) return;
            ++tasks_closed;
            --tasks_open;
            auto now=Clock::now();
            if ((tasks_closed&31)==0 || now-last_report>std::chrono::seconds(3)) {
                std::cout<<"frontier "<<tasks_closed<<' '<<tasks_open<<' '<<upper<<' '<<nodes<<'\n'<<std::flush;
                last_report=now;
            }
        }
        root_closed+=root_unresolved;
        root_unresolved=0;
        upper=p.goal;
    }

    void active_task(const State& n) {
        upper=n.upper;
        task_first=n.previous; task_second=n.last;
        route={n.previous,n.last};
    }

    void window_report() const {
        report();
        std::cout<<"population_cuts "<<population_cuts<<"\nclosed_tasks "<<tasks_closed
                 <<"\nunresolved_tasks "<<tasks_open<<"\nactive_task "<<task_first<<' '<<task_second<<'\n';
    }
};

int main(int argc,char** argv) {
    if (argc!=4 && argc!=5) throw std::runtime_error("Usage: window seconds weighted_tracked population_tracked [prices]");
    const bool supplied = argc == 5;
    if (supplied && std::string(argv[4]) != "prices") throw std::runtime_error("Unknown mode");
    const double seconds=std::stod(argv[1]);
    const int tracked=std::stoi(argv[2]),pop_tracked=std::stoi(argv[3]);
    Problem p;
    p.read();
    if (p.n>64 || p.people>1'000'000 || p.stock>1'000'000 || p.capacity>1'000'000 ||
        p.witness_pop>p.people || p.witness_flow>p.stock ||
        !(seconds>0 && std::isfinite(seconds)) || tracked<1 || tracked>9 || pop_tracked<0 || pop_tracked>7)
        throw std::runtime_error("Unsupported audit domain");
    WindowCertificate proof(p,seconds);
    proof.verify(tracked,pop_tracked,supplied);
    proof.window_report();
    if (!proof.apparent_counterexample.empty()) return 1;
    return proof.expired ? 2 : 0;
}
