// Demand-driven population walk bounds with mandatory, elementary suppliers.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
constexpr int impossible = -1;
constexpr int uncomputed = -2;
constexpr size_t max_cells = 25000000;

struct Arc {
    int to, time, bit;
};

struct Cell {
    int best = uncomputed;
    int second = impossible;
    int16_t next = -1;
    int16_t other = -1;
};

struct Table {
    int n, horizon, donor_mask, track_mask;
    bool no_return;
    size_t plane_size;
    std::vector<std::unique_ptr<Cell[]>> planes;
    std::vector<int> times, finish, prizes, bits;
    std::vector<std::vector<Arc>> outgoing;
    Clock::time_point stop;
    uint64_t calls = 0, computed = 0, allocated = 0;
    bool expired = false;

    Table(int nodes, int limit, int donors, int tracked, bool forbid_return, const int* matrix,
          const int* needs, const int* optional, const int* returns, const int* rewards)
        : n(nodes), horizon(limit), donor_mask((1 << donors) - 1),
          track_mask(((1 << tracked) - 1) << donors), no_return(forbid_return),
          plane_size(size_t(horizon+1)*n), planes(size_t(1) << (donors+tracked)),
          times(matrix,matrix+size_t(n)*n), finish(returns,returns+n),
          prizes(rewards,rewards+n), bits(n), outgoing(n) {
        for (int v=1; v<n; ++v) bits[v] = needs[v] | (optional[v] << donors);
        for (int u=1; u<n; ++u) {
            for (int v=1; v<n; ++v) {
                int time = times[u*n+v];
                if (u!=v && time<=horizon) outgoing[u].push_back({v,time,bits[v]});
            }
            std::sort(outgoing[u].begin(), outgoing[u].end(), [](const Arc& a, const Arc& b) {
                return a.time != b.time ? a.time < b.time : a.to < b.to;
            });
        }
    }

    static void insert(Cell& cell, int value, int next) {
        if (value>cell.best) {
            cell.second=cell.best;
            cell.other=cell.next;
            cell.best=value;
            cell.next=static_cast<int16_t>(next);
        } else if (value>cell.second) {
            cell.second=value;
            cell.other=static_cast<int16_t>(next);
        }
    }

    Cell& cell(int mask, int left, int last) {
        auto& plane=planes[mask];
        if (!plane) {
            plane=std::make_unique<Cell[]>(plane_size);
            ++allocated;
        }
        return plane[size_t(left)*n+last];
    }

    const Cell& cached(int mask, int left, int last) const {
        return planes[mask][size_t(left)*n+last];
    }

    int selected_value(const Cell& entry, int previous) const {
        return no_return && entry.next==previous ? entry.second : entry.best;
    }

    int recurse(int mask, int left, int last, int previous) {
        if (expired || ((calls++ & 255)==0 && Clock::now()>=stop)) {
            expired=true;
            return impossible;
        }
        Cell& entry=cell(mask,left,last);
        if (entry.best!=uncomputed) return selected_value(entry,previous);
        Cell candidate;
        candidate.best=impossible;
        if (!(mask & donor_mask) && finish[last]<=left) insert(candidate,0,0);
        for (const Arc& arc:outgoing[last]) {
            if (arc.time>left) break;
            if ((mask & arc.bit)!=arc.bit) continue;
            int tail=recurse(mask ^ arc.bit,left-arc.time,arc.to,last);
            if (expired) return impossible;
            if (tail!=impossible) insert(candidate,prizes[arc.to]+tail,arc.to);
        }
        entry=candidate;
        ++computed;
        return selected_value(entry,previous);
    }

    int query(int mask, int left, int last, int previous, double seconds, int& result) {
        expired=seconds<=0;
        calls=0;
        stop=Clock::now()+std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double>(std::max(seconds,0.0)));
        result=recurse(mask,left,last,previous);
        return expired || Clock::now()>=stop ? 0 : 1;
    }

    int trace(int mask, int left, int last, int previous, int* route, int capacity) const {
        if (!planes[mask]) return -2;
        int length=0;
        while (true) {
            const Cell& entry=cached(mask,left,last);
            if (selected_value(entry,previous)<0) return -1;
            int next=no_return && entry.next==previous ? entry.other : entry.next;
            if (!next) return length;
            if (next<0 || length>=capacity) return -2;
            route[length++]=next;
            left-=times[last*n+next];
            mask^=bits[next];
            previous=last;
            last=next;
        }
    }

    bool valid_state(int mask, int left, int last, int previous) const {
        return mask>=0 && mask<int(planes.size()) && left>=0 && left<=horizon &&
               last>0 && last<n && previous>=-1 && previous<n && !(mask & bits[last]);
    }
};

using I64 = int64_t;
using Wide = __int128_t;
using Mask = uint64_t;
using ExactWitness = void (*)(int,const int*,I64*);
using ExactCheckpoint = void (*)(I64,uint64_t,uint64_t,int*);

struct Group {
    I64 supply=0, people=0;
};

struct Node {
    Mask seen=0;
    int mask=0, left=0, last=0, previous=0, groups=0;
    I64 people=0, unloaded=0, onboard=0, future=0, credit=0, upper=0;
    Group runs[5];
};

struct Search {
    const Table& table;
    const I64 *population, *supply;
    std::vector<I64> prices;
    int agencies=0;
    int use_equity;
    I64 capacity, flow_cap, unit, total_people, incumbent, root_upper, active_upper;
    bool expired=false, broken=false;
    uint64_t probes=0, expanded=0, evaluated=0;
    Clock::time_point stop, next_event;
    ExactWitness witness;
    ExactCheckpoint checkpoint;
    std::vector<int> route;
    std::vector<Mask> agency_bit, reachable;
    std::vector<int> outgoing_cost, incoming_cost, charge, pop_order, price_order;

    Search(const Table& source,const I64* person,const I64* stock,const int* correction,
           I64 cap,I64 flow,I64 scale,I64 known,I64 upper,int equity,double seconds,
           ExactWitness report,ExactCheckpoint progress)
        : table(source),population(person),supply(stock),prices(correction,correction+source.n),
          use_equity(equity),capacity(cap),flow_cap(flow),unit(scale),total_people(0),
          incumbent(known),root_upper(upper),active_upper(upper),
          stop(Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(
              std::max(0.0,seconds)))),next_event(Clock::now()),witness(report),checkpoint(progress),
          agency_bit(source.n),reachable(size_t(source.n)*(source.horizon+1)),
          outgoing_cost(source.n,source.horizon+1),incoming_cost(source.n,source.horizon+1),
          charge(source.n) {
        auto shortest=table.times;
        for(int i=0;i<table.n;++i) {
            shortest[i*table.n+i]=0;
            for(int j=0;j<table.n;++j) if(i!=j)
                outgoing_cost[i]=std::min(outgoing_cost[i],table.times[i*table.n+j]);
        }
        for(int j=0;j<table.n;++j) {
            for(int i=0;i<table.n;++i) if(i!=j)
                incoming_cost[j]=std::min(incoming_cost[j],table.times[i*table.n+j]-outgoing_cost[i]);
            charge[j]=outgoing_cost[j]+incoming_cost[j];
        }
        for(int k=0;k<table.n;++k) for(int i=0;i<table.n;++i) for(int j=0;j<table.n;++j) {
            int& current=shortest[i*table.n+j];
            current=std::min(current,shortest[i*table.n+k]+shortest[k*table.n+j]);
        }
        for(int v=1;v<table.n;++v) if(population[v]) {
            total_people+=population[v];
            agency_bit[v]=Mask{1}<<agencies++;
            pop_order.push_back(v);
            if(prices[v]) price_order.push_back(v);
            for(int i=1;i<table.n;++i) {
                int effective=shortest[i*table.n+v]+shortest[v*table.n];
                for(int left=effective;left<=table.horizon;++left)
                    reachable[size_t(i)*(table.horizon+1)+left]|=agency_bit[v];
            }
        }
        auto by_density=[&](int a,int b,const I64* value) {
            Wide aa=Wide(value[a])*charge[b], bb=Wide(value[b])*charge[a];
            return aa!=bb ? aa>bb : a<b;
        };
        std::sort(pop_order.begin(),pop_order.end(),[&](int a,int b){return by_density(a,b,population);});
        std::sort(price_order.begin(),price_order.end(),[&](int a,int b){return by_density(a,b,prices.data());});
    }

    I64 packing(Mask allowed,int budget,const std::vector<int>& order,const I64* values) const {
        I64 result=0;
        for(int v:order) if(allowed & agency_bit[v]) {
            if(charge[v]>budget) {
                result+=I64(Wide(values[v])*budget/charge[v]);
                break;
            }
            result+=values[v];
            budget-=charge[v];
        }
        return result;
    }

    // Chronological supply deadlines; dispensing at the virtual future agency
    // may use all future stock and inventory carried out of the known prefix.
    I64 chain(const Node& node,I64 future_people,I64 flow) const {
        Group hull[6];
        int count=0, ready=node.groups;
        Wide future_stock=node.future;
        if(node.groups && !node.runs[node.groups-1].people) {
            future_stock+=node.runs[node.groups-1].supply;
            --ready;
        }
        I64 rest=flow, tail=node.people+future_people;
        auto add=[&](I64 stock,I64 people) {
            I64 available=std::min(stock,rest);
            rest-=available;
            hull[count++]={available,people};
            while(count>1 && Wide(hull[count-2].supply)*hull[count-1].people >
                  Wide(hull[count-1].supply)*hull[count-2].people) {
                hull[count-2].supply+=hull[count-1].supply;
                hull[count-2].people+=hull[count-1].people;
                --count;
            }
        };
        for(int i=0;i<ready;++i) add(node.runs[i].supply,node.runs[i].people);
        if(future_people) add(I64(std::min<Wide>(future_stock,flow)),future_people);
        Wide upper=0;
        for(int i=0;i<count;++i) {
            tail-=hull[i].people;
            upper+=Wide(hull[i].supply)*(hull[i].people+2*Wide(tail));
        }
        return I64(upper);
    }

    I64 bound(const Node& node) {
        if(!table.planes[node.mask]) {broken=true;return -1;}
        const Cell& label=table.cached(node.mask,node.left,node.last);
        if(label.best==uncomputed) {broken=true;return -1;}
        int reduced=table.selected_value(label,node.previous);
        if(reduced<0) return -1;
        I64 flow=I64(std::min<Wide>({flow_cap,Wide(node.unloaded)+node.onboard+node.future,
                                   Wide(node.unloaded)+Wide(capacity)*(agencies-__builtin_popcountll(node.seen))}));
        I64 potential=I64(std::min<Wide>(total_people-node.people,Wide(reduced+node.credit)*unit));
        I64 cheap=flow*(node.people+potential);
        if(cheap<=incumbent) return cheap;
        if(use_equity && chain(node,potential,flow)<=incumbent) return incumbent;
        int budget=node.left-outgoing_cost[node.last]-incoming_cost[0];
        for(int v=1;v<table.n;++v) if((table.bits[v]&table.donor_mask)&node.mask) budget-=charge[v];
        if(budget<0) return -1;
        Mask allowed=reachable[size_t(node.last)*(table.horizon+1)+node.left]&~node.seen;
        I64 pop=packing(allowed,budget,pop_order,population);
        I64 correction=packing(allowed,budget,price_order,prices.data());
        I64 remaining=I64(std::min<Wide>({potential,pop,Wide(reduced+correction)*unit}));
        I64 upper=flow*(node.people+remaining);
        return use_equity && upper>incumbent ? std::min(upper,chain(node,remaining,flow)) : upper;
    }

    Node child(const Node& parent, const Arc& edge) const {
        Node node=parent;
        int v=edge.to;
        node.mask^=edge.bit;
        node.left-=edge.time;
        node.previous=parent.last;
        node.last=v;
        if(supply[v]) {
            node.onboard=I64(std::min<Wide>(capacity,Wide(node.onboard)+supply[v]));
            node.future-=supply[v];
            if(node.runs[node.groups-1].people) node.runs[node.groups++]={supply[v],0};
            else {
                auto& active=node.runs[node.groups-1].supply;
                active=I64(std::min<Wide>(capacity,Wide(active)+supply[v]));
            }
        } else {
            node.unloaded+=node.onboard;
            node.onboard=0;
            node.people+=population[v];
            node.credit-=prices[v];
            node.seen|=agency_bit[v];
            node.runs[node.groups-1].people+=population[v];
        }
        return node;
    }

    bool stopped() {
        if(expired || broken) return true;
        if((++probes&255)!=0) return false;
        auto now=Clock::now();
        expired=now>=stop;
        if(!expired && now>=next_event) {
            int proceed=0;
            checkpoint(std::max(incumbent,active_upper),expanded,evaluated,&proceed);
            expired=proceed!=1;
            next_event=now+std::chrono::milliseconds(300);
        }
        return expired;
    }

    void consider(const Node& node) {
        if((node.mask&table.donor_mask) || table.finish[node.last]>node.left) return;
        I64 bound=node.unloaded*node.people;
        if(use_equity) bound=std::min(bound,chain(node,0,node.unloaded));
        if(bound<=incumbent) return;
        ++evaluated;
        I64 observed=-2;
        witness(int(route.size()),route.data(),&observed);
        if(observed < -1) {expired=true;return;}
        incumbent=std::max(incumbent,observed);
    }

    int children(const Node& node, Node* result) {
        int count=0;
        for(const Arc& edge:table.outgoing[node.last]) {
            if(edge.time>node.left) break;
            if((edge.bit&node.mask)!=edge.bit || (agency_bit[edge.to]&node.seen)) continue;
            Node next=child(node,edge);
            next.upper=std::min(node.upper,bound(next));
            if(broken) break;
            if(next.upper>incumbent) result[count++]=next;
        }
        std::sort(result,result+count,[](const Node& a,const Node& b) {
            return a.upper!=b.upper ? a.upper>b.upper : a.left>b.left;
        });
        return count;
    }

    void visit(const Node& node) {
        if(node.upper<=incumbent || stopped()) return;
        ++expanded;
        consider(node);
        if(node.upper<=incumbent || stopped()) return;
        Node next[69];
        int count=children(node,next);
        for(int i=0;i<count && !expired && !broken && next[i].upper>incumbent;++i) {
            route.push_back(next[i].last);
            visit(next[i]);
            route.pop_back();
        }
    }

    struct Task {Node node;std::vector<int> prefix;};

    void split(const Node& node,std::vector<Task>& tasks) {
        if(node.upper<=incumbent || stopped()) return;
        ++expanded;
        consider(node);
        if(node.upper<=incumbent || stopped()) return;
        if(route.size()>=3) {
            tasks.push_back({node,route});
            return;
        }
        Node next[69];
        int count=children(node,next);
        for(int i=0;i<count && !expired && !broken && next[i].upper>incumbent;++i) {
            route.push_back(next[i].last);
            split(next[i],tasks);
            route.pop_back();
        }
    }

    I64 run(int mask,int first,int left) {
        Node initial;
        initial.mask=mask;
        initial.left=left;
        initial.last=first;
        initial.onboard=supply[first];
        initial.runs[0]={supply[first],0};
        initial.groups=1;
        for(int v=1;v<table.n;++v) {
            if((table.bits[v]&table.donor_mask)&mask) initial.future+=supply[v];
            initial.credit+=prices[v];
        }
        if(Clock::now()>=stop) {
            expired=true;
            return std::max(incumbent,root_upper);
        }
        initial.upper=std::min(root_upper,bound(initial));
        active_upper=initial.upper;
        std::vector<Task> tasks;
        route.push_back(first);
        split(initial,tasks);
        std::sort(tasks.begin(),tasks.end(),[](const Task& a,const Task& b) {
            return a.node.upper>b.node.upper;
        });
        for(const Task& task:tasks) {
            if(expired || broken || task.node.upper<=incumbent) break;
            active_upper=task.node.upper;
            route=task.prefix;
            visit(task.node);
        }
        expired=expired || Clock::now()>=stop;
        return expired || broken ? std::max(incumbent,active_upper) : incumbent;
    }
};

bool one_bit(int value) {
    return value>0 && !(value & (value-1));
}
}

extern "C" {
void* hpdp_mandatory_create(int n, int horizon, int donors, int tracked, int no_return,
                            const int* times, const int* donor_bits, const int* tracked_bits,
                            const int* finish, const int* rewards, int* status) noexcept {
    *status=-1;
    if (n<3 || n>120 || horizon<0 || horizon>4096 || donors<1 || donors>5 ||
        tracked<0 || tracked>8 ||
        size_t(n)*(horizon+1)*(size_t(1) << (donors+tracked))>max_cells)
        return nullptr;
    int donor_seen=0, track_seen=0, maximum=0, agencies=0;
    for (int v=1; v<n; ++v) {
        int mandatory=donor_bits[v], optional=tracked_bits[v];
        if (rewards[v]<0 || finish[v]<0 || (mandatory && optional)) return nullptr;
        if (mandatory) {
            if (!one_bit(mandatory) || mandatory>=(1<<donors) || (donor_seen & mandatory) ||
                rewards[v] || finish[v]<=horizon)
                return nullptr;
            donor_seen|=mandatory;
        } else {
            ++agencies;
        }
        if (optional) {
            if (!one_bit(optional) || optional>=(1<<tracked) || (track_seen & optional)) return nullptr;
            track_seen|=optional;
        }
        maximum=std::max(maximum,rewards[v]);
        for (int u=1; u<n; ++u) {
            int time=times[u*n+v];
            if (u!=v && (time<0 || (!time && !mandatory && !optional))) {
                *status=-3;
                return nullptr;
            }
        }
    }
    if (!agencies || donor_seen!=(1<<donors)-1 || track_seen!=(1<<tracked)-1 ||
        int64_t(maximum)*(horizon+donors+tracked+1)>=std::numeric_limits<int>::max())
        return nullptr;
    try {
        auto* table=new Table(n,horizon,donors,tracked,no_return!=0,times,donor_bits,
                              tracked_bits,finish,rewards);
        *status=1;
        return table;
    } catch (const std::bad_alloc&) {*status=-2;return nullptr;}
}

void hpdp_mandatory_free(void* handle) noexcept {
    delete static_cast<Table*>(handle);
}

int hpdp_mandatory_query(void* handle, int mask, int left, int last, int previous,
                         double seconds, int* output) noexcept {
    auto* table=static_cast<Table*>(handle);
    if (!table || !table->valid_state(mask,left,last,previous)) return -1;
    try {
        return table->query(mask,left,last,previous,seconds,*output);
    } catch (const std::bad_alloc&) {return -2;}
}

int hpdp_mandatory_trace(void* handle, int mask, int left, int last, int previous,
                         int* output, int capacity) noexcept {
    auto* table=static_cast<Table*>(handle);
    if (!table || !table->valid_state(mask,left,last,previous)) return -2;
    return table->trace(mask,left,last,previous,output,capacity);
}

int hpdp_mandatory_exact_v2(void* handle, const int64_t* population, const int64_t* supply,
                         const int* prices, int mask, int first, int left, int64_t capacity,
                         int64_t flow, int64_t unit, int64_t incumbent, int64_t root_upper,
                         int equity, double seconds, ExactWitness witness, ExactCheckpoint checkpoint,
                         int64_t* upper, uint64_t* expanded, uint64_t* evaluated) noexcept {
    const auto* table=static_cast<Table*>(handle);
    if(!table || !table->valid_state(mask,left,first,0) || !table->planes[mask] || table->n>69 ||
       !(table->bits[first]&table->donor_mask) || capacity<=0 || flow<=0 || unit<=0 ||
       incumbent<-1 || root_upper<0 || !witness || !checkpoint) return -1;
    Wide total_people=0,total_stock=0;
    int count=0;
    for(int v=1;v<table->n;++v) {
        if(population[v]<0 || supply[v]<0 || (population[v]>0)==(supply[v]>0) || supply[v]>capacity ||
           prices[v]<0 || (population[v] && (table->bits[v]&table->donor_mask)) ||
           (supply[v] && !(table->bits[v]&table->donor_mask))) return -1;
        total_people+=population[v];
        total_stock+=supply[v];
        count+=population[v]>0;
    }
    if(count>63 || total_people>std::numeric_limits<I64>::max() ||
       total_people*flow>std::numeric_limits<I64>::max() ||
       total_stock>std::numeric_limits<I64>::max() || total_people*total_stock>std::numeric_limits<I64>::max())
        return -1;
    try {
        Search s(*table,population,supply,prices,capacity,flow,unit,incumbent,root_upper,
                 equity,seconds,witness,checkpoint);
        *upper=s.run(mask,first,left);
        *expanded=s.expanded;
        *evaluated=s.evaluated;
        return s.broken ? -3 : s.expired ? 0 : 1;
    } catch (const std::bad_alloc&) {return -2;}
}

uint64_t hpdp_mandatory_state_count(void* handle) noexcept {
    return static_cast<Table*>(handle)->computed;
}

uint64_t hpdp_mandatory_plane_count(void* handle) noexcept {
    return static_cast<Table*>(handle)->allocated;
}
}
