    // The beam prices a path with a feasible capped-rate delivery along it.
    // The price is the integral of a fixed population layer curve; accepted
    // routes are rescored with canonical water filling, independent of this cap.
    struct StationMask {
        std::array<std::uint64_t, 8> words{};

        void add(int site) {
            const auto index = static_cast<unsigned>(site - 1);
            words[index / 64] |= std::uint64_t{1} << (index % 64);
        }
        bool contains(int site) const {
            const auto index = static_cast<unsigned>(site - 1);
            return (words[index / 64] >> (index % 64)) & 1;
        }
        bool intersects(const StationMask& other) const {
            for (std::size_t i = 0; i < words.size(); ++i)
                if (words[i] & other.words[i]) return true;
            return false;
        }
        bool operator==(const StationMask&) const = default;
    };

    StationMask mask(const Route& route) const {
        StationMask result;
        for (int site : route) result.add(site);
        return result;
    }

    std::vector<double> pricing_exit(const Route& allowed) const {
        // Backwards shortest eligible return via an agency; no triangle
        // inequality or shortcut is applied to the original directed costs.
        const double infinity = std::numeric_limits<double>::infinity();
        std::vector<double> distance(stride, infinity);
        std::vector<bool> settled(stride);
        for (int site : allowed)
            if (!pickup(site)) distance[site] = edge(site, 0);
        for (std::size_t step = 0; step < allowed.size() && !done(); ++step) {
            int tail = 0;
            for (int site : allowed)
                if (!settled[site] && distance[site] < distance[tail]) tail = site;
            if (!tail || !fits(distance[tail])) break;
            settled[tail] = true;
            for (int site : allowed)
                if (!settled[site])
                    distance[site] = std::min(distance[site], edge(site, tail) + distance[tail]);
        }
        return distance;
    }

    struct BeamLabel {
        Route route;
        StationMask visited;
        double elapsed = 0, load = 0, prize = 0;
    };

    struct BeamMove {
        int parent, site;
        double elapsed, load, prize, key;
    };

    std::vector<Column> price_routes(const std::vector<int>& owner, const Background& bg,
                                     const Column& anchor, int a, int b, int depth_limit) {
        Route allowed;
        for (int node = 1; node <= n; ++node)
            if (owner[node] < 0 || owner[node] == a || owner[node] == b) allowed.push_back(node);
        auto to_exit = pricing_exit(allowed);
        double reference_people = 0, reference_stock = 0;
        for (const auto level : anchor.levels) {
            reference_people += level.people;
            reference_stock += level.people * level.rate;
        }
        if (reference_stock <= 0) {
            reference_people = total_people / std::max(2, 2 * k);
            reference_stock = std::min(capacity, 1.0);
        }

        constexpr std::array<double, 8> rate_bands{0.43, 0.63, 0.80, 1.0, 1.12, 1.37, 1.70, 2.25};
        const double target_rate = (reference_stock / reference_people) * rate_bands[pick(rate_bands.size())];
        const double price_at_zero = reference_people + bg.people.back();
        const double weighted_mark = target_rate * reference_people * price_at_zero;
        const double delay_price = (0.10 + 0.37 * uniform()) * weighted_mark;
        const double stock_credit = 0.40 + 0.30 * uniform();
        std::vector<double> preference(stride), demand(stride);
        for (int site : allowed) {
            preference[site] = 0.93 + 0.14 * uniform();
            demand[site] = std::min(capacity, target_rate * people[site]);
        }
        // At an agency, all provisional deliveries are capped by target_rate.
        // These prices use the survival curve of the retained routes exactly.
        auto priced_delivery = [&](int site, double quantity) {
            const double rate = quantity / people[site];
            const auto first_above = std::upper_bound(bg.levels.begin(), bg.levels.end(), rate,
                [](double r, Level level) { return r < level.rate; });
            const auto i = static_cast<std::size_t>(first_above - bg.levels.begin());
            return preference[site] * (reference_people * quantity + people[site] *
                (bg.quantity[i] + rate * (bg.people.back() - bg.people[i])));
        };

        std::vector<BeamLabel> layer(1), next;
        std::vector<BeamMove> proposals;
        const int width = n >= 64 ? 30 : 22;
        const int terminal_limit = 7;
        std::vector<Column> terminals;
        std::vector<StationMask> terminal_masks;
        std::vector<double> terminal_merit;
        proposals.reserve(width * allowed.size());
        for (int depth = 0; depth < depth_limit && !layer.empty() && !done(); ++depth) {
            proposals.clear();
            for (int parent = 0; parent < static_cast<int>(layer.size()); ++parent) {
                if (done()) break;
                const auto& label = layer[parent];
                const int predecessor = depth ? label.route.back() : 0;
                for (int site : allowed) {
                    if ((depth == 0 && !pickup(site)) || label.visited.contains(site)) continue;
                    const double elapsed = label.elapsed + edge(predecessor, site);
                    if (!fits(elapsed + to_exit[site])) continue;
                    double load = label.load, prize = label.prize;
                    if (pickup(site)) load = std::min(capacity, load + supply[site]);
                    else {
                        const double unload = std::min(load, demand[site]);
                        if (unload > 0) prize += priced_delivery(site, unload);
                        load -= unload;
                    }
                    const double credit = stock_credit * price_at_zero *
                                          std::min(load, reference_stock);
                    const double time_fraction = limit > 0
                        ? (elapsed + 0.65 * to_exit[site]) / limit : 0;
                    proposals.push_back({parent, site, elapsed, load, prize,
                                         prize + credit - delay_price * time_fraction});
                }
            }
            const auto preferred = [](const BeamMove& x, const BeamMove& y) {
                if (x.key != y.key) return x.key > y.key;
                if (x.elapsed != y.elapsed) return x.elapsed < y.elapsed;
                if (x.site != y.site) return x.site < y.site;
                return x.parent < y.parent;
            };
            const auto shortlist = std::min(proposals.size(), static_cast<std::size_t>(5 * width));
            std::partial_sort(proposals.begin(), proposals.begin() + shortlist, proposals.end(), preferred);
            next.clear();
            std::vector<std::array<unsigned, 4>> counts(stride);
            for (std::size_t index = 0; index < shortlist && !done(); ++index) {
                const auto& move = proposals[index];
                const int bucket = std::min(3, static_cast<int>(4 * (move.load / capacity)));
                if (counts[move.site][bucket]++ >= 2) continue;
                auto label = layer[move.parent];
                label.route.push_back(move.site);
                label.visited.add(move.site);
                label.elapsed = move.elapsed;
                label.load = move.load;
                label.prize = move.prize;
                if (next.size() < 12 && !pickup(move.site) && move.prize > 0
                    && fits(move.elapsed + edge(move.site, 0))) {
                    Column candidate{label.route, {}, 0, 0};
                    update(candidate);
                    const double score = contribution(candidate, bg);
                    auto it = std::find(terminal_masks.begin(), terminal_masks.end(), label.visited);
                    unsigned slot = it - terminal_masks.begin();
                    if (it == terminal_masks.end() && terminal_masks.size() == terminal_limit) {
                        slot = std::min_element(terminal_merit.begin(), terminal_merit.end()) - terminal_merit.begin();
                    }
                    if (slot == terminal_masks.size()) {
                        terminals.push_back(std::move(candidate));
                        terminal_masks.push_back(label.visited);
                        terminal_merit.push_back(score);
                    } else if (better(score, -candidate.elapsed, terminal_merit[slot], -terminals[slot].elapsed)) {
                        terminals[slot] = std::move(candidate);
                        terminal_masks[slot] = label.visited;
                        terminal_merit[slot] = score;
                    }
                }
                next.push_back(std::move(label));
                if (next.size() == static_cast<std::size_t>(width)) break;
            }
            layer.swap(next);
        }
        std::sort(terminals.begin(), terminals.end(), [&](const Column& x, const Column& y) {
            const double p = contribution(x, bg), q = contribution(y, bg);
            return p == q ? x.route < y.route : p > q;
        });
        return terminals;
    }

    bool reprice(Fleet& current, std::uint64_t iteration) {
        const int a = pick(k);
        const int b = k > 1 && (iteration / 5) % 3 == 0 ? (a + 1 + pick(k - 1)) % k : -1;
        const auto base = occupancy(current);
        int last_size = 0;
        for (const auto& column : current)
            last_size = std::max(last_size, static_cast<int>(column.route.size()));
        const int depth = std::min(n, std::max(24, last_size + 10));
        auto leaders = price_routes(base, background(current, a, b), current[a], a, b, depth);
        bool improved = false;
        for (const auto& lead : leaders) {
            if (done()) break;
            Fleet trial = current;
            trial[a] = lead;
            if (b < 0 || !mask(lead.route).intersects(mask(current[b].route))) {
                const double value = measure(trial);
                archive(trial, value);
                improved |= record(trial, value, "fair-route-beam");
            }
            if (b < 0) continue;
            trial[b] = {};
            auto used = occupancy(trial);
            auto followers = price_routes(used, background(trial, b), current[b], b, -1, depth);
            for (const auto& follow : followers) {
                if (done()) break;
                trial[b] = follow;
                const double value = measure(trial);
                archive(trial, value);
                improved |= record(trial, value, "fair-paired-beam");
            }
            // Three different leaders suffice to provide conditional alternatives.
            if (&lead == &leaders[std::min<std::size_t>(2, leaders.size() - 1)]) break;
        }
        if (improved) current = best;
        return improved;
    }

    bool compatible_master(Fleet& current) {
        if (k < 2 || best_value <= 0 || pool.size() < 4) return false;
        struct Option {
            Column column;
            Background shadow;
            StationMask used;
            double solo = 0, envelope = 0;
        };
        std::vector<Option> options;
        options.reserve(pool.size() + k + 1);
        auto offer_route = [&](const Column& col) {
            Background shadow;
            shadow.levels = col.levels;
            shadow.finish();
            options.push_back({col, std::move(shadow), mask(col.route)});
        };
        offer_route({});
        for (const auto& col : best) if (!col.route.empty()) offer_route(col);
        for (const auto& [key, entry] : pool) {
            (void)key;
            if (done()) return false;
            Column col{entry.route, {}, 0, 0};
            update(col);
            offer_route(col);
        }

        const Fleet seed = best;
        Fleet candidate = seed;
        double incumbent = measure(seed);
        for (int a = 0; a < k && !done(); ++a) {
            for (int b = a + 1; b < k && !done(); ++b) {
                StationMask reserved;
                for (int v = 0; v < k; ++v) {
                    if (v == a || v == b) continue;
                    for (int site : seed[v].route) reserved.add(site);
                }
                const auto retained = background(seed, a, b);
                const double fixed = lorentz(retained.levels, total_people);
                std::vector<Option*> usable;
                for (auto& op : options) {
                    if (op.used.intersects(reserved)) continue;
                    op.solo = contribution(op.column, retained);
                    // The pair cross reward is at most self_a + self_b.
                    op.envelope = op.solo + op.column.self;
                    usable.push_back(&op);
                }
                std::sort(usable.begin(), usable.end(), [](const Option* x, const Option* y) {
                    return x->envelope == y->envelope ? x->column.route < y->column.route
                                                       : x->envelope > y->envelope;
                });
                for (std::size_t i = 0; i < usable.size() && !done(); ++i) {
                    const auto& x = *usable[i];
                    if (fixed + 2 * x.envelope <= incumbent) break;
                    for (std::size_t j = i; j < usable.size(); ++j) {
                        const auto& y = *usable[j];
                        if (fixed + x.envelope + y.envelope <= incumbent) break;
                        if (x.used.intersects(y.used)) continue;
                        const double value = fixed + x.solo + y.solo
                                           + x.shadow.cross(y.column.levels, total_people);
                        if (value <= incumbent) continue;
                        incumbent = value;
                        candidate = seed;
                        candidate[a] = x.column;
                        candidate[b] = y.column;
                    }
                }
            }
        }
        if (!record(candidate, measure(candidate), "exact-pair-pool")) return false;
        current = best;
        return true;
    }
