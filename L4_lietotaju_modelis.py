"""
L4_lietotaju_modelis.py
Lietotāja adaptīvā interfeisa programmatūra, 4. lekcija.

Paraugkods nodevumam N2: no notikumu žurnāla līdz lietotāju klasēm.
Piemēra sistēma: adaptīvs filtru interfeiss datu tabulai.

Izmanto tikai Python standarta bibliotēku (Python 3.9 vai jaunāks).

Palaišana:
    python L4_lietotaju_modelis.py              # izmanto L4_piemera_dati.csv
    python L4_lietotaju_modelis.py --generate   # no jauna ģenerē sintētisko žurnālu

Kas jāpielāgo savam projektam:
    1. GENERATE_GROUPS un generate_log()  -> jūsu sistēmas notikumi un lietotāju grupas
    2. FEATURES un compute_features()      -> jūsu 4 līdz 6 pazīmes
    3. STEREOTYPES                         -> jūsu stereotipu prototipi
"""

import csv
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

SEED = 42
DATA_FILE = Path(__file__).with_name("L4_piemera_dati.csv")

# ---------------------------------------------------------------------------
# 1. Sintētiskā žurnāla ģenerēšana
# ---------------------------------------------------------------------------
# Katrai grupai: lietotāju skaits un uzvedības parametri vienam filtram.
#   time      vidējais filtra sastādīšanas laiks, s
#   p_error   varbūtība, ka filtrā būs kļūda
#   p_tmpl    varbūtība, ka lietotājs izmantos saglabātu šablonu
#   p_help    varbūtība, ka lietotājs atvērs palīdzību
#   p_complex varbūtība, ka filtrs būs salikts (UN / VAI / iekavas)
GENERATE_GROUPS = {
    "iesacejs":     dict(n=10, time=48, p_error=0.30, p_tmpl=0.05, p_help=0.35, p_complex=0.10),
    "patstavigais": dict(n=12, time=30, p_error=0.14, p_tmpl=0.25, p_help=0.10, p_complex=0.35),
    "eksperts":     dict(n=8,  time=16, p_error=0.05, p_tmpl=0.55, p_help=0.02, p_complex=0.70),
}
SESSIONS_PER_USER = 4


def generate_log(path, seed=SEED):
    """Ģenerē žurnālu: user_id, session_id, timestamp, event_type, value."""
    rnd = random.Random(seed)
    rows = []
    uid = 0
    for group, g in GENERATE_GROUPS.items():
        for _ in range(g["n"]):
            uid += 1
            user = f"U{uid:02d}"
            ts = 1_788_000_000 + uid * 100_000
            # katram lietotājam sava individuālā novirze no grupas vidējā
            ind = {key: g[key] * rnd.uniform(0.6, 1.4) for key in ("time", "p_error", "p_tmpl", "p_help", "p_complex")}
            speed = rnd.uniform(0.0, 0.10)  # cik ātri lietotājs mācās
            for s in range(1, SESSIONS_PER_USER + 1):
                session = f"{user}_S{s}"
                # lietotājs sesiju no sesijas nedaudz uzlabojas
                learn = 1 - speed * (s - 1)
                for _ in range(rnd.randint(10, 15)):
                    ts += rnd.randint(20, 90)
                    if rnd.random() < ind["p_help"] * learn:
                        rows.append((user, session, ts, "help_open", 1))
                    if rnd.random() < ind["p_tmpl"]:
                        rows.append((user, session, ts, "template_used", 1))
                    if rnd.random() < ind["p_error"] * learn:
                        kind = "syntax_error" if rnd.random() < 0.6 else "semantic_error"
                        rows.append((user, session, ts, kind, 1))
                    if rnd.random() < ind["p_complex"]:
                        rows.append((user, session, ts, "complex_filter", 1))
                    t = max(4.0, rnd.gauss(ind["time"] * learn, ind["time"] * 0.2))
                    rows.append((user, session, ts, "filter_run", round(t, 1)))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["user_id", "session_id", "timestamp", "event_type", "value"])
        w.writerows(rows)
    return len(rows)


def read_log(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# 2. Pazīmju aprēķins
# ---------------------------------------------------------------------------
# (nosaukums, īss nosaukums, virziens: +1 liela vērtība = lielāka pieredze, -1 = mazāka)
FEATURES = [
    ("Vidējais filtra sastādīšanas laiks, s", "laiks", -1),
    ("Kļūdas uz 10 filtriem", "kļūdas", -1),
    ("Šablonu izmantošana, %", "šabloni", +1),
    ("Palīdzības atvēršana uz 10 filtriem", "palīdzība", -1),
    ("Salikto filtru īpatsvars, %", "salikti", +1),
]


def compute_features(events):
    """Atgriež pazīmju vektoru no viena lietotāja (vai vienas sesijas) notikumiem."""
    times = [float(e["value"]) for e in events if e["event_type"] == "filter_run"]
    n = len(times) or 1
    cnt = defaultdict(int)
    for e in events:
        cnt[e["event_type"]] += 1
    return [
        sum(times) / n,
        10 * (cnt["syntax_error"] + cnt["semantic_error"]) / n,
        100 * cnt["template_used"] / n,
        10 * cnt["help_open"] / n,
        100 * cnt["complex_filter"] / n,
    ]


def features_by_user(log):
    per_user = defaultdict(list)
    for e in log:
        per_user[e["user_id"]].append(e)
    return {u: compute_features(ev) for u, ev in sorted(per_user.items())}


def features_by_session(log, user):
    per_session = defaultdict(list)
    for e in log:
        if e["user_id"] == user:
            per_session[e["session_id"]].append(e)
    return [compute_features(ev) for _, ev in sorted(per_session.items())]


# ---------------------------------------------------------------------------
# 3. Normalizācija
# ---------------------------------------------------------------------------
def minmax_params(vectors):
    cols = list(zip(*vectors))
    return [min(c) for c in cols], [max(c) for c in cols]


def normalize(x, mins, maxs):
    """Min-max normalizācija; vērtības ārpus diapazona ierobežo līdz [0, 1]."""
    out = []
    for v, lo, hi in zip(x, mins, maxs):
        z = (v - lo) / (hi - lo) if hi > lo else 0.0
        out.append(min(1.0, max(0.0, z)))
    return out


# ---------------------------------------------------------------------------
# 4. Stereotipi un k-vidējo
# ---------------------------------------------------------------------------
def dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# Prototipi normalizētajā telpā, pazīmju secība kā FEATURES
STEREOTYPES = {
    "Iesācējs":     [1.0, 1.0, 0.0, 1.0, 0.0],
    "Patstāvīgais": [0.5, 0.5, 0.5, 0.5, 0.5],
    "Eksperts":     [0.0, 0.0, 1.0, 0.0, 1.0],
}


def nearest(x, centers):
    """Atgriež tuvākā centra nosaukumu (vai indeksu) un attālumu."""
    items = centers.items() if isinstance(centers, dict) else enumerate(centers)
    return min(((k, dist(x, c)) for k, c in items), key=lambda t: t[1])


def kmeans(points, k, rnd, max_iter=100):
    centers = [list(p) for p in rnd.sample(points, k)]
    labels = [0] * len(points)
    for _ in range(max_iter):
        new = [nearest(p, centers)[0] for p in points]
        if new == labels and _ > 0:
            break
        labels = new
        for j in range(k):
            members = [p for p, l in zip(points, labels) if l == j]
            if members:
                centers[j] = [sum(c) / len(members) for c in zip(*members)]
    inertia = sum(dist(p, centers[l]) ** 2 for p, l in zip(points, labels))
    return labels, centers, inertia


def best_kmeans(points, k, runs=10, seed=SEED):
    """k-vidējo ar `runs` dažādiem sākuma centriem; atgriež rezultātu ar mazāko iekšējo novirzi."""
    rnd = random.Random(seed + k)
    return min((kmeans(points, k, rnd) for _ in range(runs)), key=lambda r: r[2])


def silhouette(points, labels):
    k = max(labels) + 1
    scores = []
    for i, p in enumerate(points):
        own = [dist(p, q) for j, q in enumerate(points) if labels[j] == labels[i] and j != i]
        if not own:
            scores.append(0.0)
            continue
        a = sum(own) / len(own)
        b = min(
            sum(dist(p, q) for j, q in enumerate(points) if labels[j] == c)
            / max(1, sum(1 for l in labels if l == c))
            for c in range(k) if c != labels[i]
        )
        scores.append((b - a) / max(a, b))
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# 5. Modelis laikā: EMA un histerēze
# ---------------------------------------------------------------------------
def ema(values, alpha):
    m = values[0]
    out = [m]
    for x in values[1:]:
        m = alpha * x + (1 - alpha) * m
        out.append(m)
    return out


def hysteresis(session_vectors, centers, start, delta=0.1, n_required=2):
    """Klase mainās tikai tad, ja jaunā klase ir tuvāka vismaz par delta N sesijas pēc kārtas."""
    current, candidate, streak, history = start, None, 0, []
    for x in session_vectors:
        best, d_best = nearest(x, centers)
        d_cur = dist(x, centers[current])
        if best != current and d_best < d_cur - delta:
            streak = streak + 1 if best == candidate else 1
            candidate = best
            if streak >= n_required:
                current, candidate, streak = best, None, 0
        else:
            candidate, streak = None, 0
        history.append((best, current))
    return history


# ---------------------------------------------------------------------------
# Izvade
# ---------------------------------------------------------------------------
def fmt(v, d=2):
    return f"{v:.{d}f}".replace(".", ",")


def table(header, rows):
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    line = lambda r: "  ".join(str(c).ljust(w) for c, w in zip(r, widths))
    print(line(header))
    for r in rows:
        print(line(r))
    print()


def main():
    if "--generate" in sys.argv or not DATA_FILE.exists():
        n = generate_log(DATA_FILE)
        print(f"Ģenerēts žurnāls {DATA_FILE.name}: {n} notikumi, seed = {SEED}\n")

    log = read_log(DATA_FILE)
    feats = features_by_user(log)
    users = list(feats)
    sessions = {e["session_id"] for e in log}
    print(f"Žurnāls: {len(log)} notikumi, {len(users)} lietotāji, {len(sessions)} sesijas\n")

    short = [f[1] for f in FEATURES]
    print("== Pazīmes (pirmie 5 lietotāji) ==")
    table(["U"] + short, [[u] + [fmt(v, 1) for v in feats[u]] for u in users[:5]])

    mins, maxs = minmax_params(list(feats.values()))
    print("== Normalizācijas parametri (min-max) ==")
    table(["pazīme", "min", "max"], [[s, fmt(a, 1), fmt(b, 1)] for s, a, b in zip(short, mins, maxs)])

    X = {u: normalize(v, mins, maxs) for u, v in feats.items()}
    points = [X[u] for u in users]

    print("== Stereotipi ==")
    stereo = {u: nearest(X[u], STEREOTYPES)[0] for u in users}
    for name in STEREOTYPES:
        print(f"{name}: {sum(1 for c in stereo.values() if c == name)} lietotāji")
    print()

    print("== k-vidējo (10 palaišanas katram k) ==")
    results = {}
    rows = []
    for k in (2, 3, 4):
        labels, centers, inertia = best_kmeans(points, k)
        s = silhouette(points, labels)
        results[k] = (labels, centers)
        rows.append([k, fmt(s, 3), fmt(inertia, 2)])
    table(["k", "silueta koef.", "iekšējā novirze"], rows)

    k = 3
    labels, centers = results[k]
    # klasēm piešķir nosaukumus pēc tuvākā stereotipa
    names = {j: nearest(c, STEREOTYPES)[0] for j, c in enumerate(centers)}
    named_centers = {names[j]: c for j, c in enumerate(centers)}
    print(f"== Izvēlētās klases, k = {k} (centri normalizētajā telpā) ==")
    table(["klase", "n"] + short,
          [[names[j], labels.count(j)] + [fmt(v) for v in c] for j, c in enumerate(centers)])

    diff = [u for u, l in zip(users, labels) if names[l] != stereo[u]]
    print("== Atšķirības: stereotipi pret k-vidējo ==")
    table(["U", "stereotips", "k-vidējo"], [[u, stereo[u], names[labels[users.index(u)]]] for u in diff] or [["nav", "", ""]])

    print("== Jaunu lietotāju klasificēšana (tuvākais klases centrs) ==")
    rnd = random.Random(SEED + 100)
    test_rows = []
    for i, group in enumerate(["iesacejs", "patstavigais", "eksperts", "patstavigais", "eksperts"], 1):
        g = GENERATE_GROUPS[group]
        n = 12
        x = [
            max(4.0, rnd.gauss(g["time"], g["time"] * 0.2)),
            10 * sum(rnd.random() < g["p_error"] for _ in range(n)) / n,
            100 * sum(rnd.random() < g["p_tmpl"] for _ in range(n)) / n,
            10 * sum(rnd.random() < g["p_help"] for _ in range(n)) / n,
            100 * sum(rnd.random() < g["p_complex"] for _ in range(n)) / n,
        ]
        cls, d = nearest(normalize(x, mins, maxs), named_centers)
        test_rows.append([f"T{i:02d}"] + [fmt(v, 1) for v in x] + [cls, fmt(d)])
    table(["T"] + short + ["klase", "attālums"], test_rows)

    user = users[0]
    per_session = features_by_session(log, user)
    errs = [s[1] for s in per_session]
    print(f"== EMA pazīmei 'kļūdas', lietotājs {user} ==")
    e3, e6 = ema(errs, 0.3), ema(errs, 0.6)
    table(["sesija", "novērojums", "α = 0,3", "α = 0,6"],
          [[i + 1, fmt(o), fmt(a), fmt(b)] for i, (o, a, b) in enumerate(zip(errs, e3, e6))])

    print("== Histerēze δ = 0,1, N = 2 (lietotājs uz klašu robežas) ==")
    def changes(u):
        near = [nearest(normalize(v, mins, maxs), named_centers)[0] for v in features_by_session(log, u)]
        return sum(a != b for a, b in zip(near, near[1:]))
    border = max(users, key=changes)
    sess = [normalize(v, mins, maxs) for v in features_by_session(log, border)]
    start = nearest(sess[0], named_centers)[0]
    hist = hysteresis(sess, named_centers, start)
    table(["sesija", "tuvākā klase", "piešķirtā klase"],
          [[i + 1, b, c] for i, (b, c) in enumerate(hist)])
    print(f"Lietotājs {border}, sākuma klase {start}.")


if __name__ == "__main__":
    main()
