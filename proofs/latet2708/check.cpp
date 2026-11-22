// Exhaustive upper-bound replay for the normalized Latet 2708 instance.
// Tail relaxation and prefix bounds adapted from the independent Latet 2808 checker.
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using Quantity = std::int64_t;
using Score = __int128_t;
using AgencyMask = std::uint64_t;

struct Successors {
    int node = -1;
    int largest = -2;
    int other = -1;

    void add(int next, int score) {
        if (score > largest) {
            other = largest;
            largest = score;
            node = next;
        } else {
            other = std::max(other, score);
        }
    }
};

class BoundCheck {
    static constexpr int donor_count = 5;
    static constexpr int masks = 1 << donor_count;
    int size, horizon, first, suppliers, tag_size;
    Quantity capacity;
    Score threshold = 0;
    int population_total = 0, correction = 0, donor_flags = 0;
    std::vector<int> people, prices, tag;
    std::vector<Quantity> stock;
    std::vector<AgencyMask> agency;
    std::vector<std::vector<int>> duration, next_nodes;
    std::vector<Successors> walk;
    struct Boundary { int population; Quantity before, after; };
    std::vector<Boundary> boundaries;
    std::uint64_t states = 0, prefixes = 0, unresolved = 0;

    int reward(int last, int previous, int remaining, int flags) {
        auto& options = walk[(std::size_t(remaining) * size + last) * (1 << tag_size) + flags];
        if (options.largest == -2) {
            options.largest = -1;
            ++states;
            if (people[last] && !(donor_flags & flags) && duration[last][0] <= remaining) {
                options.add(0, 0);
            }
            for (int to : next_nodes[last]) {
                int step = duration[last][to];
                int flag = tag[to];
                if (step > remaining || (flag && !(flag & flags))) continue;
                int continuation = reward(to, last, remaining - step, flags ^ flag);
                if (continuation >= 0) options.add(to, continuation + people[to] - prices[to]);
            }
        }
        return previous == options.node ? options.other : options.largest;
    }

    bool above(int population, Quantity flow) const {
        // Total delivery bound and both signs of each before/after disparity.
        if (Score(population) * flow <= threshold) return false;
        for (const auto& split : boundaries) {
            if (Score(population - split.population) * flow + Score(population) * split.before <= threshold ||
                Score(split.population) * flow + Score(population) * split.after <= threshold) return false;
        }
        return true;
    }

    void visit(int last, int previous, int remaining, int flags, AgencyMask seen, int covered,
               int unspent_price, Quantity delivered_cap, Quantity processed_cap, Quantity unvisited_stock) {
        int tail = reward(last, previous, remaining, flags);
        if (tail < 0 || !above(std::min(population_total, covered + unspent_price + tail),
                              processed_cap + unvisited_stock)) return;
        ++prefixes;
        if (people[last] && !(flags & donor_flags) && duration[last][0] <= remaining &&
            above(covered, delivered_cap)) {
            ++unresolved;
        }
        for (int to : next_nodes[last]) {
            int cost = duration[last][to];
            int flag = tag[to];
            if (cost > remaining || (agency[to] & seen) || (flag && !(flags & flag))) continue;
            if (stock[to]) {
                Quantity future = unvisited_stock - stock[to];
                boundaries.push_back({covered, delivered_cap, capacity + future});
                visit(to, last, remaining - cost, flags ^ flag, seen, covered, unspent_price,
                      delivered_cap, std::min(processed_cap + stock[to], delivered_cap + capacity), future);
                boundaries.pop_back();
            } else {
                visit(to, last, remaining - cost, flags ^ flag, seen | agency[to], covered + people[to],
                      unspent_price - prices[to], processed_cap, processed_cap, unvisited_stock);
            }
        }
    }

public:
    void run() {
        std::string cutoff;
        int track;
        std::cin >> size >> horizon >> first >> suppliers >> track >> capacity >> cutoff;
        if (!std::cin || size != 68 || horizon != 540 || capacity != 12000000000000000LL ||
            first < 1 || first > donor_count || suppliers <= 0 || suppliers >= masks ||
            !(suppliers & (1 << (first - 1))) || track < 0 || track > 3 ||
            cutoff.empty() || cutoff.size() > 30) {
            throw std::runtime_error("unexpected certificate domain");
        }
        for (char digit : cutoff) {
            if (digit < '0' || digit > '9') throw std::runtime_error("bad cutoff");
            threshold = threshold * 10 + (digit - '0');
        }
        people.resize(size);
        prices.resize(size);
        stock.resize(size);
        tag.resize(size);
        agency.resize(size);
        duration.assign(size, std::vector<int>(size));
        next_nodes.resize(size);
        for (int& value : people) std::cin >> value;
        for (int& price : prices) std::cin >> price;
        if (people[0] || prices[0]) throw std::runtime_error("active depot");
        for (Quantity& available : stock) std::cin >> available;
        for (auto& durations : duration) for (int& step : durations) std::cin >> step;
        Quantity total = 0;
        tag_size = 0;
        for (int v = 1; v < size; ++v) {
            if (people[v] < 0 || people[v] > 5000 || (v <= donor_count) == (people[v] > 0) ||
                prices[v] < 0 || prices[v] > people[v] || stock[v] < 0 || stock[v] > capacity ||
                (v <= donor_count) != (stock[v] > 0)) throw std::runtime_error("invalid model or price");
            correction += prices[v];
            population_total += people[v];
            if (v <= donor_count && (suppliers & (1 << (v - 1)))) {
                total += stock[v];
                if (v != first) tag[v] = 1 << tag_size++;
            } else if (v > donor_count) {
                agency[v] = AgencyMask(1) << (v - donor_count - 1);
            }
        }
        donor_flags = (1 << tag_size) - 1;
        for (int i = 0; i < track; ++i) {
            int v;
            std::cin >> v;
            if (!std::cin || v <= donor_count || v >= size || prices[v] || tag[v]) throw std::runtime_error("bad track");
            tag[v] = 1 << tag_size++;
        }
        if (!std::cin || population_total != 32660 || stock[0]) throw std::runtime_error("invalid model");
        for (int u = 0; u < size; ++u) for (int v = 0; v < size; ++v) {
            int time = duration[u][v];
            if (time < 0 || time > horizon + 1 || (u != v && v > 0 && time < 20))
                throw std::runtime_error("unexpected time or zero length step");
            if (u > 0 && v > 0 && v != first && v != u && (people[v] || tag[v])) next_nodes[u].push_back(v);
        }
        walk.resize(std::size_t(horizon + 1) * size * (1 << tag_size));
        boundaries.reserve(donor_count);
        int flags = (1 << tag_size) - 1;
        int remaining = horizon - duration[0][first];
        int tail = remaining < 0 ? -1 : reward(first, 0, remaining, flags);
        int pop_upper = tail < 0 ? -1 : std::min(population_total, tail + correction);
        if (pop_upper >= 0 && Score(pop_upper) * total > threshold) {
            visit(first, 0, remaining, flags, 0, 0, correction, 0, stock[first], total - stock[first]);
        }
        std::cout << pop_upper << ' ' << states << ' ' << prefixes << ' ' << unresolved << '\n';
    }
};

int main() {
    try {
        BoundCheck{}.run();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
