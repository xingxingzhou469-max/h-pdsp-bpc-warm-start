#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <vector>

using Int = int64_t;
using Wide = __int128_t;
using Clock = std::chrono::steady_clock;
constexpr int sites = 111;
constexpr Int impossible = -1;

struct Problem {
    int n{}, limit{};
    Int capacity{}, goal{}, witness_pop{}, witness_flow{}, people{}, stock{};
    std::array<Int,sites> population{}, supply{};
    std::array<int,sites> service{};
    std::array<std::array<int,sites>,sites> travel{}, cost{}, shortest{};
    std::array<int,sites> away{}, into{}, charge{};
    std::array<std::vector<int>,sites> next;
    std::vector<int> by_pop, by_stock;

    void read() {
        if (!(std::cin>>n>>capacity>>limit>>goal>>witness_pop>>witness_flow) || n<3 || n>sites ||
            limit<1 || limit>1600 || capacity<=0 || goal<0 || !witness_pop || !witness_flow)
            throw std::runtime_error("Unsupported audit input");
        for (int i=0;i<n;++i) {
            std::cin>>supply[i];
            supply[i]=std::min(supply[i],capacity);
            stock+=supply[i];
        }
        for (int i=0;i<n;++i) {std::cin>>population[i]; people+=population[i];}
        for (int i=0;i<n;++i) std::cin>>service[i];
        for (int i=0;i<n;++i) for (int j=0;j<n;++j) std::cin>>travel[i][j];
        if (!std::cin || service[0] || population[0] || supply[0])
            throw std::runtime_error("Invalid integer input");
        for (int i=0;i<n;++i) {
            away[i]=limit+1;
            for (int j=0;j<n;++j) {
                if (travel[j][i]<0 || service[j]<0 || (j>0 && (supply[j]>0)==(population[j]>0)))
                    throw std::runtime_error("Invalid time or role");
                cost[i][j]=std::min(limit+1, travel[j][i]+service[j]);
                shortest[i][j]=i==j ? 0 : cost[i][j];
                if (i!=j) {
                    away[i]=std::min(away[i],cost[i][j]);
                    if (j>0 && cost[i][j]<=limit) next[i].push_back(j);
                }
            }
            std::sort(next[i].begin(),next[i].end(),[&](int a,int b){
                return std::pair{cost[i][a],a}<std::pair{cost[i][b],b};
            });
        }
        for (int j=0;j<n;++j) {
            into[j]=limit+1;
            for (int i=0;i<n;++i) if (i!=j) into[j]=std::min(into[j],cost[i][j]-away[i]);
            charge[j]=away[j]+into[j];
            if (supply[j]) by_stock.push_back(j);
            if (population[j]) by_pop.push_back(j);
            for (int i=1;i<n;++i) if (i!=j && j && cost[i][j]<=0)
                throw std::runtime_error("Reverse walk needs positive internal time steps");
        }
        for (int k=0;k<n;++k) for (int i=0;i<n;++i) for (int j=0;j<n;++j)
            shortest[i][j]=std::min(shortest[i][j],shortest[i][k]+shortest[k][j]);
        auto density=[&](const auto& quantity) {
            return [&quantity,this](int a,int b) {
                Wide va=Wide(quantity[a])*charge[b],vb=Wide(quantity[b])*charge[a];
                return va!=vb ? va>vb : a<b;
            };
        };
        std::sort(by_stock.begin(),by_stock.end(),density(supply));
        std::sort(by_pop.begin(),by_pop.end(),density(population));
    }
};

struct Record {
    Int best=impossible, alternate=impossible;
    int next=-1, other=-1;
    void add(Int value,int v) {
        if (value>best) {
            alternate=best; other=next;
            best=value; next=v;
        } else if (value>alternate) {
            alternate=value; other=v;
        }
    }
};

struct RewardTable {
    const Problem& p;
    Int a,b;
    int masks=1;
    std::array<int,sites> bit{};
    std::array<Int,sites> reward{}, price{};
    std::vector<Record> data;

    RewardTable(const Problem& p_,Int a_,Int b_,const std::vector<int>& tracked,
                const std::array<Int,sites>& prices)
        :p(p_),a(a_),b(b_),price(prices) {
        for (int v:tracked) {bit[v]=masks; masks*=2; price[v]=0;}
        for (int v=1;v<p.n;++v) {
            reward[v]=a*p.supply[v]+b*p.population[v];
            if (price[v]<0 || price[v]>reward[v]) throw std::runtime_error("Invalid visit price");
        }
        data.resize(size_t(p.limit+1)*masks*p.n);
    }

    Record& at(int left,int mask,int last) {return data[(size_t(left)*masks+mask)*p.n+last];}
    const Record& at(int left,int mask,int last) const {return data[(size_t(left)*masks+mask)*p.n+last];}

    std::pair<Int,int> get(int left,int mask,int last,int prev) const {
        const auto& record=at(left,mask,last);
        return record.next==prev ? std::pair{record.alternate,record.other} : std::pair{record.best,record.next};
    }

    bool fill(Clock::time_point end) {
        for (int t=0;t<=p.limit;++t) {
            if (Clock::now()>=end) return false;
            for (int mask=0;mask<masks;++mask) for (int last=1;last<p.n;++last) {
                auto& r=at(t,mask,last);
                if (p.supply[last] && p.cost[last][0]<=t) r.add(0,0);
                for (int v:p.next[last]) {
                    int arc=p.cost[last][v];
                    if (arc>t) break;
                    if (bit[v] && !(mask&bit[v])) continue;
                    auto [z,_]=get(t-arc,mask^bit[v],v,last);
                    if (z>=0) r.add(z+reward[v]-price[v],v);
                }
            }
        }
        return true;
    }

    std::vector<int> trace(int left,int mask,int last) const {
        std::vector<int> path{last};
        int prev=0;
        while (true) {
            auto [x,v]=get(left,mask,last,prev);
            if (x<0 || v<=0) return path;
            path.push_back(v);
            left-=p.cost[last][v]; mask^=bit[v]; prev=last; last=v;
        }
    }
};

struct Certificate {
    const Problem& p;
    Clock::time_point end;
    std::vector<RewardTable> tables;
    std::array<bool,sites> used{};
    std::vector<int> route, apparent_counterexample;
    uint64_t nodes=0, bounds=0, table_rejections=0, leaves=0, time_rejections=0;
    uint64_t root_closed=0, root_unresolved=0;
    bool expired=false;
    Int upper=0, highest_route=0;
    std::array<int,sites> donor_bit{};
    std::vector<int> served_first;
    std::vector<uint16_t> donor_packing;
    int donor_masks=0;
    uint64_t donor_time_cuts=0;

    Certificate(const Problem& case_,double budget):p(case_),end(Clock::now()+
        std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(budget))) {}

    size_t donor_key(int last,int mask) const { return size_t(last)*donor_masks+mask; }

    void prepare_donors() {
        if (p.by_stock.size()>11 || p.stock>65535) return;
        auto raw=p.travel;
        for (int i=0;i<p.n;++i) for (int j=0;j<p.n;++j) raw[i][j]=p.travel[j][i];
        for (int k=0;k<p.n;++k) for (int i=0;i<p.n;++i) for (int j=0;j<p.n;++j)
            raw[i][j]=std::min(raw[i][j],raw[i][k]+raw[k][j]);
        const int count=int(p.by_stock.size()), stride=p.limit+1;
        donor_masks=1<<count;
        for (int i=0;i<count;++i) donor_bit[p.by_stock[i]]=1<<i;
        std::vector<int> duration(size_t(p.n)*donor_masks,stride);
        std::vector<int> supply(donor_masks);
        donor_packing.resize(size_t(p.n)*donor_masks*stride);
        for (int v=0;v<p.n;++v) duration[donor_key(v,0)]=std::min(stride,raw[v][0]);
        for (int mask=1;mask<donor_masks;++mask) {
            int first=__builtin_ctz(mask), lower=mask&(mask-1);
            supply[mask]=supply[lower]+p.supply[p.by_stock[first]];
            for (int v=0;v<p.n;++v) {
                const auto key=donor_key(v,mask);
                auto& time=duration[key];
                for (int bits=mask;bits;bits&=bits-1) {
                    int i=__builtin_ctz(bits), next=p.by_stock[i];
                    time=std::min(time,raw[v][next]+p.service[next]+duration[donor_key(next,mask^(1<<i))]);
                }
                if (time<stride) donor_packing[key*stride+time]=supply[mask];
            }
        }
        for (int v=1;v<p.n;++v) {
            if (Clock::now()>=end) {expired=true; return;}
            for (int i=0;i<count;++i) for (int mask=1;mask<donor_masks;++mask) if (mask&(1<<i)) {
                size_t to=donor_key(v,mask)*stride, from=donor_key(v,mask^(1<<i))*stride;
                for (int t=0;t<stride;++t)
                    donor_packing[to+t]=std::max(donor_packing[to+t],donor_packing[from+t]);
            }
            for (int mask=1;mask<donor_masks;++mask) {
                size_t key=donor_key(v,mask)*stride;
                for (int t=1;t<stride;++t)
                    donor_packing[key+t]=std::max(donor_packing[key+t],donor_packing[key+t-1]);
            }
        }
        served_first=p.by_pop;
        std::sort(served_first.begin(),served_first.end(),[&](int a,int b) {
            Wide va=Wide(p.population[a])*p.service[b],vb=Wide(p.population[b])*p.service[a];
            return va!=vb ? va>vb : a<b;
        });
    }

    void prepare(int track_count) {
        const Int d=std::gcd(p.witness_pop,p.witness_flow);
        const Int a=p.witness_pop/d,b=p.witness_flow/d;
        std::vector<Int> weights;
        for (int v=1;v<p.n;++v) weights.push_back(a*p.supply[v]+b*p.population[v]);
        std::sort(weights.begin(),weights.end());
        const Int pivot=weights[(2*weights.size())/3];
        std::array<Int,sites> base{};
        for (int v=1;v<p.n;++v) base[v]=std::max<Int>(0,a*p.supply[v]+b*p.population[v]-pivot);
        tables.emplace_back(p,a,b,std::vector<int>{},base);
        if (!tables.back().fill(end)) {tables.pop_back(); expired=true; return;}
        if (!track_count) return;

        std::array<int,sites> frequency{};
        for (int v=1;v<p.n;++v) if (p.population[v] && p.cost[0][v]<=p.limit) {
            int left=p.limit-p.cost[0][v];
            const auto& first=tables.front();
            if (first.get(left,0,v,0).first<0) continue;
            std::array<int,sites> repeats{};
            for (int w:first.trace(left,0,v)) if (++repeats[w]>1) ++frequency[w];
        }
        std::vector<int> order(p.n-1);
        std::iota(order.begin(),order.end(),1);
        std::sort(order.begin(),order.end(),[&](int u,int v){
            return std::tuple{frequency[u],tables.front().reward[u],-u}>
                   std::tuple{frequency[v],tables.front().reward[v],-v};
        });
        order.resize(std::min<int>(track_count,order.size()));
        std::array<Int,sites> second{};
        for (int v=1;v<p.n;++v)
            second[v]=std::min(tables.front().reward[v],base[v]+std::min(2,frequency[v])*pivot/3);
        tables.emplace_back(p,a,b,order,second);
        if (!tables.back().fill(end)) {tables.pop_back(); expired=true; return;}
        prepare_donors();
    }

    Int packing(const std::array<bool,sites>& allowed,int time,
                const std::vector<int>& order,const std::array<Int,sites>& value) const {
        Int total=0;
        for (int v:order) if (allowed[v]) {
            const int weight=p.charge[v];
            if (weight>time) return total+Int(Wide(time)*value[v]/weight);
            total+=value[v]; time-=weight;
        }
        return total;
    }

    Int balance_bound(Int upper, Int total, Int a, Int b, Int supply, Int pop) const {
        upper=Int(std::min(Wide(upper),Wide(total)*total/(4*Wide(a)*b)));
        if (2*Wide(a)*supply<=total)
            upper=Int(std::min(Wide(upper),Wide(supply)*(total-Wide(a)*supply)/b));
        if (2*Wide(b)*pop<=total)
            upper=Int(std::min(Wide(upper),Wide(pop)*(total-Wide(b)*pop)/a));
        return upper;
    }

    // Charge exact shortest visits to the donor subset plus its service times.
    // Charge agency service separately, with an enclosing interval for donor time.
    bool donor_interval(const std::array<bool,sites>& allowed,const std::array<Int,2>& totals,
                        size_t key,int lo,int hi,int rem,Int covered,Int served,
                        Int flow_cap,Int pop_cap) const {
        Int remaining_stock=donor_packing[key+hi];
        Int flow=std::min(flow_cap,served+remaining_stock);
        Int agency_budget=rem-lo, pop=covered;
        for (int v:served_first) if (allowed[v]) {
            if (p.service[v]>agency_budget) {
                pop+=Int(Wide(agency_budget)*p.population[v]/p.service[v]);
                break;
            }
            pop+=p.population[v]; agency_budget-=p.service[v];
        }
        pop=std::min(pop,pop_cap);
        Int ub=std::min(flow*pop,flow*covered+remaining_stock*pop);
        for (size_t i=0;i<tables.size() && ub>p.goal;++i)
            ub=balance_bound(ub,totals[i],tables[i].a,tables[i].b,flow,pop);
        if (ub<=p.goal) return true;
        if (lo==hi) return false;
        int middle=(lo+hi)/2;
        return donor_interval(allowed,totals,key,lo,middle,rem,covered,served,flow_cap,pop_cap) &&
               donor_interval(allowed,totals,key,middle+1,hi,rem,covered,served,flow_cap,pop_cap);
    }

    Int bound(int last,int prev,int rem,Int served,Int load,Int covered,Int parent) {
        ++bounds;
        if (p.shortest[last][0]>rem) return impossible;
        int available_time=rem-p.away[last]-p.into[0];
        if (available_time<0) return impossible;
        Int others=0, future_stock=0, future_pop=0;
        std::array<bool,sites> allowed{};
        std::array<int,2> bits{};
        std::array<Int,2> price{}, totals{};
        int possible_donors=0;
        for (int v=1;v<p.n;++v) if (!used[v] && p.shortest[last][v]+p.shortest[v][0]<=rem) {
            allowed[v]=true;
            possible_donors|=donor_bit[v];
            future_stock+=p.supply[v]; future_pop+=p.population[v]; others+=p.population[v]>0;
            for (size_t i=0;i<tables.size();++i) {
                bits[i]|=tables[i].bit[v];
                price[i]+=tables[i].price[v];
            }
        }
        Int flow=served+std::min(future_stock,load+p.capacity*others);
        Int pop=covered+future_pop;
        Int upper=std::min(parent,flow*pop);
        if (upper<=p.goal) return upper;
        for (size_t i=0;i<tables.size();++i) {
            const auto& t=tables[i];
            Int z=t.get(rem,bits[i],last,prev).first;
            if (z<0) return impossible;
            totals[i]=t.a*served+t.b*covered+z+price[i];
            upper=balance_bound(upper,totals[i],t.a,t.b,flow,pop);
            if (upper<=p.goal) {++table_rejections; return upper;}
        }
        future_stock=std::min(future_stock,packing(allowed,available_time,p.by_stock,p.supply));
        future_pop=std::min(future_pop,packing(allowed,available_time,p.by_pop,p.population));
        flow=std::min(flow,served+future_stock); pop=covered+future_pop;
        upper=std::min({upper,flow*pop,flow*covered+future_stock*pop});
        for (size_t i=0;i<tables.size() && upper>p.goal;++i) {
            const auto& t=tables[i];
            upper=balance_bound(upper,totals[i],t.a,t.b,flow,pop);
        }
        if (donor_masks && upper>p.goal && donor_interval(allowed,totals,
                donor_key(last,possible_donors)*(p.limit+1),0,rem,rem,covered,served,flow,pop)) {
            ++donor_time_cuts;
            return p.goal;
        }
        return upper;
    }

    Int exact_order(Int covered) {
        // Coalesce only consecutive delivery stops. Their total can be split
        // by population without changing any load outside the delivery run.
        std::vector<Int> stock, pop;
        Int group_stock=0;
        for (auto it=route.rbegin();it!=route.rend();++it) {
            int v=*it;
            if (p.supply[v]) group_stock=std::min(p.capacity,group_stock+p.supply[v]);
            else {
                if (group_stock) {stock.push_back(group_stock); pop.push_back(0); group_stock=0;}
                pop.back()+=p.population[v];
            }
        }
        if (pop.size()>24) throw std::runtime_error("Residual route rank table is too large");
        size_t count=size_t{1}<<pop.size();
        std::vector<Int> rank(count),mass(count);
        for (size_t mask=1;mask<count;++mask) {
            if ((mask&4095)==0 && Clock::now()>=end) {expired=true; return 0;}
            Int load=0;
            for (size_t i=0;i<pop.size();++i) {
                load=std::min(p.capacity,load+stock[i]);
                if (mask&(size_t{1}<<i)) {rank[mask]+=load; mass[mask]+=pop[i]; load=0;}
            }
        }
        Int result=0, part=covered;
        const size_t full=count-1;
        for (size_t fixed=0;fixed!=full;) {
            size_t remaining=full^fixed, chosen=remaining;
            Int qty=rank[full]-rank[fixed], weight=mass[remaining];
            for (size_t mask=remaining;mask;mask=(mask-1)&remaining) {
                Int dq=rank[fixed|mask]-rank[fixed];
                if (Wide(dq)*weight<Wide(qty)*mass[mask] ||
                    (Wide(dq)*weight==Wide(qty)*mass[mask] && mass[mask]>weight)) {
                    qty=dq; weight=mass[mask]; chosen=mask;
                }
            }
            part-=weight; result+=qty*(weight+2*part); fixed|=chosen;
        }
        return result;
    }

    void visit(int last,int rem,Int served,Int load,Int covered,Int parent) {
        if ((++nodes&127)==0 && Clock::now()>=end) {expired=true; return;}
        if (p.supply[last] && p.cost[last][0]<=rem && served*covered>p.goal) {
            ++leaves;
            Int objective=exact_order(covered);
            highest_route=std::max(highest_route,objective);
            if (objective>p.goal) {
                apparent_counterexample.assign(route.rbegin(),route.rend());
                return;
            }
        }
        for (int v:p.next[last]) {
            if (expired || !apparent_counterexample.empty()) return;
            const int next_time=rem-p.cost[last][v];
            if (next_time<0) break;
            if (used[v]) continue;
            if (p.shortest[v][0]>next_time) {++time_rejections; continue;}
            Int reverse_drop=std::min(load,p.supply[v]);
            Int new_load=p.population[v] ? p.capacity : load-reverse_drop;
            used[v]=true;
            Int ub=bound(v,last,next_time,served+reverse_drop,new_load,covered+p.population[v],parent);
            if (ub>p.goal) {
                route.push_back(v);
                visit(v,next_time,served+reverse_drop,new_load,covered+p.population[v],ub);
                route.pop_back();
            }
            used[v]=false;
        }
    }

    void run(int track_count) {
        upper=p.stock*p.people;
        prepare(track_count);
        if (expired) return;
        std::vector<std::pair<Int,int>> roots;
        for (int v=1;v<p.n;++v) if (p.population[v] && p.cost[0][v]<=p.limit) {
            used[v]=true;
            Int ub=bound(v,0,p.limit-p.cost[0][v],0,p.capacity,p.population[v],upper);
            used[v]=false;
            if (ub>p.goal) roots.emplace_back(ub,v);
            else ++root_closed;
        }
        std::sort(roots.begin(),roots.end(),std::greater<>{});
        root_unresolved=roots.size();
        for (auto [ub,v]:roots) {
            upper=ub;
            std::cout<<"root_open "<<v<<' '<<ub<<' '<<nodes<<'\n'<<std::flush;
            if (Clock::now()>=end) {expired=true; return;}
            route.push_back(v); used[v]=true;
            visit(v,p.limit-p.cost[0][v],0,p.capacity,p.population[v],ub);
            used[v]=false; route.pop_back();
            if (expired || !apparent_counterexample.empty()) return;
            ++root_closed; --root_unresolved;
            std::cout<<"root_closed "<<v<<' '<<ub<<' '<<nodes<<'\n'<<std::flush;
        }
        upper=p.goal;
    }

    void report() const {
        std::cout<<"goal "<<p.goal<<"\nupper "<<upper<<"\nexpanded "<<nodes
                 <<"\nbound_calls "<<bounds<<"\nwalk_rejections "<<table_rejections
                 <<"\ntime_rejections "<<time_rejections<<"\nroute_audits "<<leaves
                 <<"\nmax_audited "<<highest_route<<"\nclosed_roots "<<root_closed
                 <<"\nunresolved_roots "<<root_unresolved<<"\ntables "<<tables.size()<<'\n';
        std::cout<<"donor_time_cuts "<<donor_time_cuts<<'\n';
        for (const auto& table:tables) {
            std::cout<<"table "<<table.a<<' '<<table.b<<' '<<table.masks;
            for (int v=1;v<p.n;++v) if (table.bit[v]) std::cout<<' '<<v;
            std::cout<<'\n';
        }
        if (!apparent_counterexample.empty()) {
            std::cout<<"status counterexample_candidate\nroute";
            for (int v:apparent_counterexample) std::cout<<' '<<v;
            std::cout<<'\n';
        } else std::cout<<(expired ? "status timeout\n" : "status strict_upper_bound\n");
    }
};

int main(int argc,char** argv) {
    if (argc!=3) throw std::runtime_error("Usage: reverse_proof seconds tracked");
    double seconds=std::stod(argv[1]);
    int tracked=std::stoi(argv[2]);
    if (!(seconds>0 && std::isfinite(seconds)) || tracked<0 || tracked>7)
        throw std::runtime_error("Invalid audit budget");
    Problem p;
    p.read();
    Certificate proof(p,seconds);
    proof.run(tracked);
    proof.report();
    if (!proof.apparent_counterexample.empty()) return 1;
    return proof.expired ? 2 : 0;
}
