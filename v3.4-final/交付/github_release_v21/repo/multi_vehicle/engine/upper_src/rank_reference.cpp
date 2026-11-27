// Instance-specific exhaustive upper certificate for at most two vehicles.
// Input integers are dimension checked in the Python front end. The entire
// proof (route times, rank, slopes, upper comparisons) uses integer arithmetic.
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <limits>
#include <utility>
#include <vector>

using namespace std;
using i64 = int64_t;
using Level = pair<int,int>;  // total flow, total population
using Chain = vector<Level>;
constexpr int N = 20;
static int n, cap, limit_time, sup[N+1], pop[N+1], service[N+1], tr[N+1][N+1], tail[N+1];
struct Group { int supply, people; };
struct Bucket {
    map<Chain, vector<int>> signatures;
    int mask=0, people=0, flow_max=0;
    i64 routes=0;
};
static map<int,Bucket> buckets;
static vector<int> path;
static i64 emitted=0, prefixes=0;
static vector<i64> histogram(N+1);

static bool leq_rate(const Level &a, const Level &b) {
    return i64(a.first)*b.second <= i64(b.first)*a.second;
}

static Chain certificate() {
    // A group aggregates consecutive pickups, then consecutive deliveries.
    // Such a run can unload its entire inventory at the first selected agency.
    vector<Group> g;
    int pending=0;
    for (int v : path) {
        if (sup[v]) pending += sup[v];
        else {
            if (pending) {
                g.push_back({min(cap,pending),0});
                pending=0;
            }
            g.back().people += pop[v];
        }
    }
    int m = int(g.size()), full = (1<<m)-1;
    vector<int> rank(full+1), people(full+1);
    for (int s=1; s<=full; ++s) {
        int stock=0;
        for (int j=0; j<m; ++j) {
            stock=min(cap,stock+g[j].supply);
            if (s&(1<<j)) {
                rank[s]+=stock;
                stock=0;
                people[s]+=g[j].people;
            }
        }
    }
    // Each prefix is a valid upper bound even without any decomposition
    // theorem. Minimum residual slopes produce increasing levels that are
    // tight for the candidate allocation. Longest tied prefix is deterministic.
    int fixed=0;
    Chain chain;
    while (fixed != full) {
        int rem=full^fixed, chosen=0;
        Level best;
        for (int add=rem; add; add=(add-1)&rem) {
            Level cand{rank[fixed|add]-rank[fixed],people[add]};
            if (!chosen || !leq_rate(best,cand) ||
                (leq_rate(cand,best) && people[add]>people[chosen])) {
                chosen=add; best=cand;
            }
        }
        if (!chain.empty() && !leq_rate(chain.back(),best))
            throw runtime_error("rank prefix slopes not increasing");
        fixed |= chosen;
        if (!chain.empty() && leq_rate(best,chain.back())) {
            chain.back().first += best.first;
            chain.back().second += best.second;
        } else chain.push_back(best);
    }
    return chain;
}

static void visit(int last, int elapsed, int mask, int people) {
    ++prefixes;
    if (pop[last] && elapsed+tr[last][0]<=limit_time) {
        ++emitted; ++histogram[path.size()];
        Chain c=certificate();
        int total=0;
        for (const auto &l:c) total+=l.first;
        auto &b=buckets[mask];
        ++b.routes;
        b.mask=mask; b.people=people; b.flow_max=max(b.flow_max,total);
        b.signatures.emplace(c,path);
    }
    for (int v=1; v<=n; ++v) {
        int bit=1<<(v-1);
        if ((mask&bit) || (!sup[v] && !pop[v])) continue;
        int arr=elapsed+tr[last][v]+service[v];
        if (arr+tail[v]>limit_time) continue;
        path.push_back(v);
        visit(v,arr,mask|bit,people+pop[v]);
        path.pop_back();
    }
}

// P times the upper bound for the two chains. In the sorted joint order,
// block with (q,p) gets integer coefficient 2*(higher population)+p.
// Equal slopes can be ordered either way. Each route prefix has a known
// upper on the flow; these coefficients are nonincreasing on that route.
static i64 pair_upper(const Chain &a, const Chain &b,int people) {
    size_t i=0,j=0;
    i64 ans=0;
    while (i<a.size() || j<b.size()) {
        const auto &l = j==b.size() || (i<a.size() && leq_rate(a[i],b[j])) ? a[i++] : b[j++];
        people-=l.second;
        ans+=i64(l.first)*(l.second+2*people);
    }
    return ans;
}

// Diagnostic entry point only. The caller supplies already checked routes.
// Compare each analytic upper directly against an independent joint flow LP.
static void audit_fixed_pairs() {
    int count;
    cin >> count;
    while(count--) {
        Chain chains[2];
        int people=0;
        for(int k=0;k<2;++k) {
            int len; cin>>len;
            path.resize(len);
            for(int &v:path) { cin>>v; people+=pop[v]; }
            chains[k]=certificate();
        }
        cout<<pair_upper(chains[0],chains[1],people)<<"\n";
    }
}

int main(int argc, char **argv) {
    if (!(cin>>n>>cap>>limit_time) || n>N || n<1) return 2;
    for (int *arr : {sup,pop,service}) for (int i=0;i<=n;++i) cin>>arr[i];
    for (int i=0;i<=n;++i) for(int j=0;j<=n;++j) cin>>tr[i][j];
    if (!cin) return 2;
    if (argc==2 && string(argv[1])=="--fixed-pairs") {
        audit_fixed_pairs();
        return 0;
    }
    if (argc!=1) return 2;
    // Exact lower bound with directed, non-metric times, allowing revisits.
    for(int i=1;i<=n;++i) tail[i]=pop[i]?tr[i][0]:1000000000;
    for(int k=0;k<n;++k) for(int i=1;i<=n;++i) for(int j=1;j<=n;++j)
        if (i!=j) tail[i]=min(tail[i],tr[i][j]+service[j]+tail[j]);
    buckets[0].signatures[{}]={};
    buckets[0].routes=1;
    for (int s=1;s<=n;++s) if(sup[s]) {
        int t=tr[0][s]+service[s];
        if(t+tail[s]<=limit_time) {
            path={s};
            visit(s,t,1<<(s-1),0);
        }
    }
    vector<const Bucket*> list;
    i64 sig=0;
    for(auto &it:buckets) { list.push_back(&it.second); sig+=it.second.signatures.size(); }
    sort(list.begin(),list.end(),[](const Bucket *a,const Bucket *b){
        i64 ax=i64(a->people)*a->flow_max, bx=i64(b->people)*b->flow_max;
        return ax!=bx ? ax>bx : a->mask<b->mask;
    });
    i64 best=-1, disjoint=0, group_pruned=0, pair_scored=0, fleet_combinations=0;
    i64 max_single=0, max_jensen=0;
    vector<int> ra,rb;
    for(size_t i=0;i<list.size();++i) {
        const auto &a=*list[i];
        for(size_t j=i;j<list.size();++j) {
            const auto &b=*list[j];
            if(a.mask&b.mask) continue;
            ++disjoint;
            if (a.routes > (numeric_limits<i64>::max()-fleet_combinations)/b.routes)
                throw overflow_error("diagnostic fleet count exceeds int64");
            fleet_combinations+=a.routes*b.routes;
            // Scalar group bound: within each route min is bounded by both
            // arguments. The cross term is bounded by both group totals.
            i64 coarse=i64(a.people)*a.flow_max + i64(b.people)*b.flow_max
                +2*min(i64(a.people)*b.flow_max,i64(b.people)*a.flow_max);
            max_jensen=max(max_jensen,coarse);
            if(coarse<=best) { ++group_pruned; continue; }
            for(const auto &x:a.signatures) for(const auto &y:b.signatures) {
                ++pair_scored;
                i64 val=pair_upper(x.first,y.first,a.people+b.people);
                if (val>coarse) throw runtime_error("coarse bound underestimated chain value");
                if(val>best) { best=val;ra=x.second;rb=y.second; }
            }
        }
        for(const auto &x:a.signatures)
            max_single=max(max_single,pair_upper(x.first,{},a.people));
    }
    cout<<"upper_num "<<best<<"\nenumerated_nonempty_routes "<<emitted
        <<"\ndfs_prefixes "<<prefixes<<"\nmasks_including_idle "<<list.size()
        <<"\nchain_signatures_including_idle "<<sig<<"\ndisjoint_mask_pairs_including_idle "<<disjoint
        <<"\ndisjoint_route_pairs_including_idle "<<fleet_combinations
        <<"\ncoarse_pruned_mask_pairs "<<group_pruned<<"\nchain_pairs_evaluated "<<pair_scored
        <<"\ncoarse_max_num "<<max_jensen<<"\nsingle_max_num "<<max_single<<"\n";
    for(const auto &r:{ra,rb}) { cout<<"route"; for(int v:r) cout<<" "<<v; cout<<"\n"; }
    cout<<"route_counts_by_length"; for(int i=0;i<=n;++i) cout<<" "<<histogram[i]; cout<<"\n";
}
