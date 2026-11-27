#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace v3 {

using Route = std::vector<int>;
struct Level { double rate, people; };
using Levels = std::vector<Level>;

inline bool rate_order(Level a, Level b) { return a.rate < b.rate; }

inline double lorentz(const Levels& sorted, double total_people) {
    double remaining = 0;
    for (const auto level : sorted) remaining += level.people;
    double result = 0;
    for (auto [rate, people] : sorted) {
        remaining -= people;
        result += rate * people * (people + 2 * remaining);
    }
    return result / total_people;
}

// Cross term is 2/P times the integrated product of two survival curves.
struct Background {
    Levels levels;
    std::vector<double> people{0}, quantity{0};

    void finish() {
        std::sort(levels.begin(), levels.end(), rate_order);
        people.reserve(levels.size() + 1);
        quantity.reserve(levels.size() + 1);
        for (const auto [rate, size] : levels) {
            people.push_back(people.back() + size);
            quantity.push_back(quantity.back() + size * rate);
        }
    }

    double cross(const Levels& other, double total_people) const {
        double result = 0;
        std::size_t i = 0;
        for (const auto [rate, size] : other) {
            while (i < levels.size() && levels[i].rate <= rate) ++i;
            result += size * (quantity[i] + rate * (people.back() - people[i]));
        }
        return 2 * result / total_people;
    }
};

class Profile {
    struct Group { double stock, people; };
    struct Prefix { double stock, people; };

    const std::vector<double>& supply;
    const std::vector<double>& population;
    double capacity;
    std::vector<Group> groups;
    std::vector<Prefix> hull;
    std::vector<unsigned char> predecessor;

    // Two-state chord cut over the pickup/delivery chain. Record predecessor
    // sides for traceback so the number of groups never depends on word size.
    bool cut(const std::vector<unsigned char>& fixed,
             const std::vector<unsigned char>& allowed, double rate, double target,
             std::vector<unsigned char>& chosen) {
        constexpr double inf = std::numeric_limits<double>::infinity();
        double outside = 0, inside = 0;
        predecessor.resize(2 * groups.size());
        for (std::size_t i = 0; i < groups.size(); ++i) {
            const auto [stock, people] = groups[i];
            for (int sink = 0; sink < 2; ++sink) {
                const bool from_in = inside + capacity < outside;
                const bool keep_in = inside <= outside;
                const double next_out = (from_in ? inside + capacity : outside) + (sink ? 0 : stock);
                double next_in = std::min(inside, outside);
                if (sink && fixed[i]) next_in = inf;
                else if (sink && allowed[i]) next_in += rate * people;
                predecessor[2 * i + sink] = (from_in ? 1 : 0) | (keep_in ? 2 : 0);
                outside = next_out;
                inside = next_in;
            }
        }
        const double error = 1e-13 * std::abs(target);
        if (std::min(outside, inside) >= target - error) return false;
        int side = (inside < outside);
        for (std::size_t j = predecessor.size(); j-- > 0;) {
            if (j % 2) chosen[j / 2] = !side && allowed[j / 2];
            side = (predecessor[j] >> side) & 1;
        }
        return true;
    }

    void split(const std::vector<unsigned char>& left,
               const std::vector<unsigned char>& right, Prefix lo, Prefix hi,
               int count, Levels& result) {
        const double people = hi.people - lo.people;
        if (!(people > 0)) return;
        const double rate = std::max(0.0, hi.stock - lo.stock) / people;
        if (count > 1) {
            std::vector<unsigned char> middle(groups.size());
            if (cut(left, right, rate, hi.stock, middle)) {
                Prefix mid{};
                int added = 0;
                double load = 0;
                for (std::size_t i = 0; i < groups.size(); ++i) {
                    load = std::min(capacity, load + groups[i].stock);
                    if (!middle[i]) continue;
                    mid.stock += load;
                    mid.people += groups[i].people;
                    load = 0;
                    added += !left[i];
                }
                if (0 < added && added < count && lo.people < mid.people && mid.people < hi.people) {
                    split(left, middle, lo, mid, added, result);
                    split(middle, right, mid, hi, count - added, result);
                    return;
                }
            }
        }
        result.push_back({rate, people});
    }

public:
    Profile(const std::vector<double>& s, const std::vector<double>& p, double cap)
        : supply(s), population(p), capacity(cap) {}

    void set_capacity(double cap) { capacity = cap; }

    void operator()(const Route& route, Levels& result) {
        result.clear();
        if (route.empty() || capacity <= 0) return;
        groups.clear();
        double pending = 0;
        Prefix total{};
        for (int site : route) {
            if (supply[site] > 0) {
                pending = std::min(capacity, pending + supply[site]);
            } else {
                if (pending > 0) {
                    groups.push_back({pending, 0});
                    total.stock += pending;
                    pending = 0;
                }
                groups.back().people += population[site];
                total.people += population[site];
            }
        }
        if (groups.size() == 1) {
            result.push_back({total.stock / total.people, total.people});
        } else if (total.stock <= capacity) {
            hull.clear();
            hull.push_back({});
            Prefix point{};
            for (const auto group : groups) {
                point.stock += group.stock;
                point.people += group.people;
                while (hull.size() > 1) {
                    const auto a = hull[hull.size() - 2], b = hull.back();
                    if ((b.stock - a.stock) * (point.people - b.people)
                        < (point.stock - b.stock) * (b.people - a.people)) break;
                    hull.pop_back();
                }
                hull.push_back(point);
            }
            for (std::size_t i = 1; i < hull.size(); ++i) {
                const auto a = hull[i - 1], b = hull[i];
                if (b.people > a.people)
                    result.push_back({(b.stock - a.stock) / (b.people - a.people), b.people - a.people});
            }
        } else {
            split(std::vector<unsigned char>(groups.size(), 0),
                  std::vector<unsigned char>(groups.size(), 1),
                  {}, total, static_cast<int>(groups.size()), result);
        }
        std::sort(result.begin(), result.end(), rate_order);
    }
};

}  // namespace v3
