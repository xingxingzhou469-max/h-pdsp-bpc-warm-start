// Resource-constrained walk relaxation. Each cell stores the two best
// different next stops so an immediate return to the previous stop is excluded.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <new>
#include <unordered_map>
#include <vector>

namespace {
constexpr int impossible = -1000000000;
using Clock = std::chrono::steady_clock;

struct Arc {
    int destination, time, bit;
};

struct Cell {
    int best = impossible;
    int second = impossible;
    int8_t first_stop = -1;
    int8_t second_stop = -1;
};

struct Table {
    int n, limit, masks;
    bool forbid_return;
    std::vector<Cell> labels;
    std::vector<int> bits, weights, return_time, costs;
    std::vector<std::vector<Arc>> outgoing;

    Table(int nodes, int horizon, int count, bool no_return, const int* times,
          const int* prizes, const int* track, const int* inner, const int* finish)
        : n(nodes), limit(horizon), masks(1 << count), forbid_return(no_return),
          labels(size_t(masks) * (limit + 1) * n), bits(track,track+n),
          weights(prizes,prizes+n), return_time(finish,finish+n),
          costs(times,times+size_t(n)*n), outgoing(n) {
        for (int i=0; i<n; ++i) {
            for (int j=1; j<n; ++j) {
                int time=costs[i*n+j];
                if (inner[j] && i!=j && time<=limit) {
                    outgoing[i].push_back({j,time,bits[j]});
                }
            }
            std::sort(outgoing[i].begin(),outgoing[i].end(),
                      [](const Arc& a,const Arc& b){return a.time < b.time;});
        }
    }

    size_t index(int mask,int t,int last) const {
        return (size_t(mask)*(limit+1)+t)*n+last;
    }

    int value(int mask,int t,int last,int previous) const {
        const Cell& c=labels[index(mask,t,last)];
        return forbid_return && previous>=0 && c.first_stop==previous ? c.second : c.best;
    }

    bool build(double seconds) {
        auto stop=Clock::now()+std::chrono::duration<double>(seconds);
        for (int mask=0;mask<masks;++mask) {
            for (int t=0;t<=limit;++t) {
                if ((t&7)==0 && Clock::now()>=stop) return false;
                for (int i=1;i<n;++i) {
                    Cell c;
                    if (return_time[i]<=t) {c.best=0;c.first_stop=0;}
                    for (const Arc& arc:outgoing[i]) {
                        if (arc.time>t) break;
                        if ((mask & arc.bit)!=arc.bit) continue;
                        int tail=value(mask^arc.bit,t-arc.time,arc.destination,i);
                        if (tail==impossible) continue;
                        int score=weights[arc.destination]+tail;
                        if (score>c.best) {
                            c.second=c.best;
                            c.second_stop=c.first_stop;
                            c.best=score;
                            c.first_stop=arc.destination;
                        } else if (score>c.second) {
                            c.second=score;
                            c.second_stop=arc.destination;
                        }
                    }
                    labels[index(mask,t,i)]=c;
                }
            }
        }
        return true;
    }

    int trace(int mask,int t,int last,int previous,int* route) const {
        if (value(mask,t,last,previous)==impossible) return -1;
        int length=0;
        for (;;) {
            const Cell& c=labels[index(mask,t,last)];
            int next=forbid_return && previous>=0 && c.first_stop==previous ? c.second_stop : c.first_stop;
            if (next==0) return length;
            if (next<0 || length>=limit) return -2;
            route[length++]=next;
            t-=costs[last*n+next];
            mask^=bits[next];
            previous=last;
            last=next;
        }
    }
};

using Mask=uint64_t;
using Witness=void (*)(int,const int*,int,int*);
using Checkpoint=void (*)(int,int64_t,int*);

struct Prefix {
    Mask visited;
    int last,previous,left,available,people,upper;
};

struct Elementary {
    const Table& table;
    const int* population;
    const int* prices;
    int incumbent;
    int active_bound=0;
    int64_t expanded=0;
    Clock::time_point stop, next_save;
    bool expired=false;
    Witness witness;
    Checkpoint checkpoint;
    std::vector<int> path;
    std::vector<Mask> reachable;
    std::vector<int> out_price, in_price, charge, population_order, correction_order;
    std::vector<std::unordered_map<Mask,int>> most_time;
    size_t dominance_entries=0;

    Elementary(const Table& tab,const int* people,const int* correction,int first_score,
               double seconds,Witness save_witness,Checkpoint save_checkpoint)
        : table(tab),population(people),prices(correction),incumbent(first_score),
          stop(Clock::now()+std::chrono::duration_cast<Clock::duration>(
                           std::chrono::duration<double>(std::max(0.0,seconds)))),
          next_save(Clock::now()),witness(save_witness),checkpoint(save_checkpoint),
          reachable(size_t(tab.n)*(tab.limit+1)),
          out_price(tab.n,tab.limit+1),in_price(tab.n,tab.limit+1),
          charge(tab.n),most_time(tab.n) {
        int n=tab.n;
        auto shortest=table.costs;
        for (int i=0;i<n;++i) {
            shortest[i*n+i]=0;
            for (int j=0;j<n;++j) if(i!=j) {
                out_price[i]=std::min(out_price[i],table.costs[i*n+j]);
            }
        }
        for (int j=0;j<n;++j) for (int i=0;i<n;++i) if(i!=j) {
            in_price[j]=std::min(in_price[j],table.costs[i*n+j]-out_price[i]);
        }
        for (int k=0;k<n;++k) for (int i=0;i<n;++i) for(int j=0;j<n;++j) {
            shortest[i*n+j]=std::min(shortest[i*n+j],shortest[i*n+k]+shortest[k*n+j]);
        }
        for(int v=1;v<n;++v) if(population[v]>0) {
            charge[v]=out_price[v]+in_price[v];
            population_order.push_back(v);
            if(prices[v]>0) correction_order.push_back(v);
            for(int i=1;i<n;++i) {
                int least=shortest[i*n+v]+shortest[v*n];
                for(int left=least;left<=table.limit;++left)
                    reachable[i*(table.limit+1)+left]|=Mask{1}<<v;
            }
        }
        auto sort_prices=[&](auto& order,const int* reward) {
            std::sort(order.begin(),order.end(),[&](int a,int b) {
                auto left=int64_t(reward[a])*charge[b], right=int64_t(reward[b])*charge[a];
                return left==right ? a<b : left>right;
            });
        };
        sort_prices(population_order,population);
        sort_prices(correction_order,prices);
    }

    int packing(Mask allowed,int budget,const std::vector<int>& order,const int* prize) const {
        int upper=0;
        for (int v:order) if(allowed & (Mask{1}<<v)) {
            if(charge[v]>budget) {
                upper+=int(int64_t(budget)*prize[v]/charge[v]);
                break;
            }
            upper+=prize[v];
            budget-=charge[v];
        }
        return upper;
    }

    int bound(const Prefix& node) const {
        int reduced=table.value(node.available,node.left,node.last,node.previous);
        int budget=node.left-out_price[node.last]-in_price[0];
        if(reduced==impossible || budget<0) return -1;
        Mask allowed=reachable[node.last*(table.limit+1)+node.left]&~node.visited;
        int population_cap=packing(allowed,budget,population_order,population);
        int correction=packing(allowed,budget,correction_order,prices);
        return node.people+std::min(population_cap,reduced+correction);
    }

    bool expired_now() {
        if ((expanded&255)!=0) return expired;
        auto now=Clock::now();
        expired|=now>=stop;
        if(!expired && now>=next_save) {
            int proceed=0;
            checkpoint(active_bound,expanded,&proceed);
            expired=proceed!=1;
            next_save=now+std::chrono::milliseconds(400);
        }
        return expired;
    }

    void visit(const Prefix& node) {
        if(node.upper<=incumbent || expired_now()) return;
        ++expanded;
        auto& previous=most_time[node.last];
        auto found=previous.find(node.visited);
        if(found!=previous.end()) {
            if(found->second>=node.left) return;
            found->second=node.left;
        } else if(dominance_entries<300000) {
            previous.emplace(node.visited,node.left);
            ++dominance_entries;
        }
        if(node.people>incumbent && table.return_time[node.last]<=node.left) {
            int verified=-1;
            witness(int(path.size()),path.data(),node.people,&verified);
            if(verified!=node.people) {expired=true;return;}
            incumbent=verified;
        }
        // At most 63 nondepot sites are admitted by the C ABI below.
        Prefix children[64];
        int count=0;
        for(const Arc& arc:table.outgoing[node.last]) {
            if(arc.time>node.left) break;
            int v=arc.destination;
            Mask bit=Mask{1}<<v;
            if(node.visited&bit) continue;
            Prefix child{node.visited|bit,v,node.last,node.left-arc.time,
                         node.available^arc.bit,node.people+population[v],0};
            child.upper=std::min(node.upper,bound(child));
            if(child.upper>incumbent) children[count++]=child;
        }
        std::sort(children,children+count,[](const Prefix& a,const Prefix& b) {
            return a.upper!=b.upper ? a.upper>b.upper : a.left>b.left;
        });
        for(int i=0;i<count && !expired;++i) {
            path.push_back(children[i].last);
            visit(children[i]);
            path.pop_back();
        }
    }

    struct Task {Prefix prefix; std::vector<int> route;};

    void split(const Prefix& node,std::vector<Task>& tasks) {
        if(expired || node.upper<=incumbent) return;
        ++expanded;
        if(expired_now()) return;
        if(node.people>incumbent && table.return_time[node.last]<=node.left) {
            int verified=-1;
            witness(int(path.size()),path.data(),node.people,&verified);
            if(verified!=node.people) {expired=true;return;}
            incumbent=verified;
        }
        if(path.size()>=3) {
            tasks.push_back({node,path});
            return;
        }
        for(const Arc& arc:table.outgoing[node.last]) {
            if(arc.time>node.left) break;
            int v=arc.destination;
            Mask bit=Mask{1}<<v;
            if(node.visited&bit) continue;
            Prefix child{node.visited|bit,v,node.last,node.left-arc.time,
                         node.available^arc.bit,node.people+population[v],0};
            child.upper=std::min(node.upper,bound(child));
            path.push_back(v);
            split(child,tasks);
            path.pop_back();
        }
    }

    int search(int first,int time_left,int root_upper) {
        Prefix root{Mask{1}<<first,first,0,time_left,table.masks-1,0,root_upper};
        root.upper=std::min(root_upper,bound(root));
        active_bound=root.upper;
        expired=Clock::now()>=stop;
        if(expired) return std::max(incumbent,active_bound);
        std::vector<Task> tasks;
        path.push_back(first);
        split(root,tasks);
        std::sort(tasks.begin(),tasks.end(),[](const Task& a,const Task& b) {
            return a.prefix.upper>b.prefix.upper;
        });
        for(const Task& task:tasks) {
            if(expired || task.prefix.upper<=incumbent) break;
            active_bound=task.prefix.upper;
            path=task.route;
            visit(task.prefix);
        }
        expired|=Clock::now()>=stop;
        return expired ? std::max(incumbent,active_bound) : incumbent;
    }
};
}

extern "C" {
void* hpdp_walk_build(int n,int limit,int count,int forbid_return,const int* times,
                      const int* prizes,const int* bits,const int* inner,const int* finish,
                      double seconds,int* status) {
    *status=-1;
    if(n<3 || n>120 || limit<1 || limit>4096 || count<0 || count>10) return nullptr;
    int highest=0;
    for(int i=0;i<n;++i) {
        if(prizes[i]<0 || bits[i]<0 || bits[i]>=(1<<count) ||
           (bits[i] && (bits[i]&(bits[i]-1)))) return nullptr;
        highest=std::max(highest,prizes[i]);
        for(int j=1;j<n;++j) if(i>0 && i!=j && inner[j] && times[i*n+j]<=0) return nullptr;
    }
    if(int64_t(highest)*(limit+1)>=std::numeric_limits<int>::max()/2) return nullptr;
    Table* tab=nullptr;
    try {
        tab=new Table(n,limit,count,forbid_return!=0,times,prizes,bits,inner,finish);
        if(!tab->build(seconds)) {delete tab;*status=0;return nullptr;}
        *status=1;
        return tab;
    } catch (const std::bad_alloc&) {delete tab;*status=-2;return nullptr;}
}
void hpdp_walk_free(void* ptr) {delete static_cast<Table*>(ptr);}
int hpdp_walk_value(void* ptr,int mask,int t,int last,int previous) {
    return static_cast<Table*>(ptr)->value(mask,t,last,previous);
}
int hpdp_walk_trace(void* ptr,int mask,int t,int last,int previous,int* route) {
    return static_cast<Table*>(ptr)->trace(mask,t,last,previous,route);
}
int hpdp_walk_elementary_v2(void* ptr,const int* population,const int* correction,int first,
                         int remaining,int initial,int root_upper,double seconds,
                         Witness witness,Checkpoint checkpoint,int* upper,int64_t* expanded) {
    const auto& table=*static_cast<Table*>(ptr);
    if(table.n>64 || first<=0 || first>=table.n || remaining<0 || remaining>table.limit) return -1;
    try {
        Elementary search(table,population,correction,initial,seconds,witness,checkpoint);
        *upper=search.search(first,remaining,root_upper);
        *expanded=search.expanded;
        return search.expired ? 0 : 1;
    } catch (const std::bad_alloc&) {return -2;}
}
}
