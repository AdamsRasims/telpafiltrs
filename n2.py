"""TelpaFiltrs N2. Python 3.9+, tikai standarta bibliotēka.

Algoritmu pamats: kursa L4_lietotaju_modelis.py; sava datu shēma,
pazīmes, ģenerators, atsevišķi testa dati un secīgs EMA/histerēzes piemērs.
"""
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEED = 231279
FEATURES = ['Laiks, s', 'Sintakses kļūdas / 10', 'Semantikas kļūdas / 10',
            'Salikti noteikumi, %', 'Atkārtota izmantošana, %']
NAMES = ['Iesācējs', 'Patstāvīgais', 'Eksperts']
PROTOTYPES = [[1, 1, 1, 0, 0], [.5, .5, .5, .5, .5], [0, 0, 0, 1, 1]]
# Vidējais aktīvais laiks un kļūdu, sarežģītības, atkārtotas izmantošanas varbūtības.
GROUPS = [(78, .29, .19, .12, .08), (46, .12, .09, .48, .40),
          (24, .035, .025, .85, .72)]
FIELDS = ['user_id', 'session_id', 'timestamp', 'attempt_id', 'event_type',
          'value', 'or_count', 'not_count', 'max_depth']


def make_session(rng, user, session, params):
    rows = []
    ts = 1790812800 + int(user[1:]) * 86400 + session * 7200
    for attempt in range(1, 13):
        key = f'{user}_S{session}_A{attempt:02d}'
        duration = round(min(180, max(5, rng.gauss(params[0], params[0] * .16))), 1)
        ts += math.ceil(duration) + 5
        def event(kind, value=1, or_count=0, not_count=0, depth=0):
            rows.append([user, f'{user}_S{session}', ts, key, kind, value,
                         or_count, not_count, depth])
        if rng.random() < params[1]:
            event('syntax_error')
        if rng.random() < params[2]:
            event('semantic_error')
        if rng.random() < params[4]:
            event('saved_rule_reused')
        advanced = rng.random() < params[3]
        mode = rng.randrange(3) if advanced else -1
        event('rule_submit', duration, int(mode == 0), int(mode == 1), 2 if mode == 2 else 0)
    return rows


def write_csv(name, rows):
    with (ROOT / name).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(FIELDS)
        writer.writerows(rows)


def generate():
    rng = random.Random(SEED)
    rows = []
    for index in range(36):
        base = GROUPS[index // 12]
        individual = [base[0] * rng.uniform(.78, 1.22)] + [min(.98, p * rng.uniform(.65, 1.35)) for p in base[1:]]
        for s in range(1, 5):
            params = individual.copy()
            params[0] *= 1 - .025 * (s - 1)
            rows.extend(make_session(rng, f'U{index+1:02d}', s, params))
    write_csv('zurnals.csv', rows)
    rng = random.Random(SEED + 1)
    rows = []
    test_params = [GROUPS[0], GROUPS[1], GROUPS[2], (58, .18, .13, .30, .25), (30, .05, .04, .68, .58)]
    for index, params in enumerate(test_params, 1):
        for s in range(1, 5):
            rows.extend(make_session(rng, f'T{index:02d}', s, params))
    write_csv('testa_zurnals.csv', rows)
    rng = random.Random(SEED + 2)
    rows = []
    # Atsevišķs sintētisks mācīšanās scenārijs pēc sākotnējām četrām sesijām.
    for s in range(5, 11):
        rows.extend(make_session(rng, 'U01', s, GROUPS[1]))
    write_csv('laika_zurnals.csv', rows)


def read(name):
    with (ROOT / name).open(encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def features(events):
    submits = [e for e in events if e['event_type'] == 'rule_submit']
    n = len(submits)
    if not n:
        raise ValueError('Nav rule_submit notikumu; pazīmes nav definētas.')
    count = Counter(e['event_type'] for e in events)
    complex_count = sum(int(e['or_count']) > 0 or int(e['not_count']) > 0 or int(e['max_depth']) >= 2 for e in submits)
    return [sum(float(e['value']) for e in submits) / n,
            10 * count['syntax_error'] / n, 10 * count['semantic_error'] / n,
            100 * complex_count / n, 100 * count['saved_rule_reused'] / n]


def grouped(rows, key):
    result = defaultdict(list)
    for row in rows:
        result[row[key]].append(row)
    return dict(sorted(result.items()))


def normalize(vector, lo, hi):
    return [min(1, max(0, (v-a)/(b-a))) if b > a else 0 for v, a, b in zip(vector, lo, hi)]


def distance(a, b):
    return math.sqrt(sum((x-y)**2 for x, y in zip(a, b)))


def nearest(point, centers):
    return min(range(len(centers)), key=lambda i: distance(point, centers[i]))


def kmeans(points, initial):
    centers = [points[i][:] for i in initial]
    old = None
    for _ in range(300):
        labels = [nearest(p, centers) for p in points]
        if labels == old:
            return labels, centers, sum(distance(p, centers[l])**2 for p, l in zip(points, labels))
        old = labels
        updated = []
        for i in range(len(centers)):
            members = [p for p, l in zip(points, labels) if l == i]
            if not members:
                return None  # Nederīgu mēģinājumu aizstāj ar jaunu inicializāciju.
            updated.append([sum(c)/len(c) for c in zip(*members)])
        centers = updated
    raise RuntimeError('k-vidējo metode nekonverģēja')


def best_kmeans(points, k):
    rng = random.Random(SEED + 100 + k)
    seen, runs = set(), []
    while len(runs) < 20:
        initial = tuple(sorted(rng.sample(range(len(points)), k)))
        if initial in seen:
            continue
        seen.add(initial)
        result = kmeans(points, initial)
        if result is not None:
            runs.append(result)
    return min(runs, key=lambda item: item[2])


def silhouette(points, labels):
    scores = []
    classes = sorted(set(labels))
    for i, p in enumerate(points):
        own = [distance(p, q) for j, q in enumerate(points) if labels[j] == labels[i] and i != j]
        if not own:
            scores.append(0)
            continue
        a = sum(own) / len(own)
        b = min(sum(distance(p, q) for q, l in zip(points, labels) if l == c) / labels.count(c)
                for c in classes if c != labels[i])
        scores.append((b-a)/max(a, b) if max(a, b) else 0)
    return sum(scores)/len(scores)


def validate(rows, min_users, min_sessions):
    users = grouped(rows, 'user_id')
    assert len(users) >= min_users
    for events in users.values():
        assert len(grouped(events, 'session_id')) >= min_sessions
        if min_sessions >= 4:
            assert sum(e['event_type'] == 'rule_submit' for e in events) >= 48, 'Klasificēšanai nepieciešami vismaz 48 mēģinājumi.'
    for attempt in grouped(rows, 'attempt_id').values():
        counts = Counter(e['event_type'] for e in attempt)
        assert counts['rule_submit'] == 1
        assert all(v <= 1 for v in counts.values())
        assert all(e['event_type'] in {'rule_submit','syntax_error','semantic_error','saved_rule_reused'} for e in attempt)
        submit = next(e for e in attempt if e['event_type'] == 'rule_submit')
        assert 5 <= float(submit['value']) <= 180


def calculate():
    train, test, future = read('zurnals.csv'), read('testa_zurnals.csv'), read('laika_zurnals.csv')
    validate(train, 36, 4)
    validate(test, 5, 4)
    validate(future, 1, 6)
    raw = {u: features(ev) for u, ev in grouped(train, 'user_id').items()}
    lo, hi = [min(c) for c in zip(*raw.values())], [max(c) for c in zip(*raw.values())]
    vectors = {u: normalize(v, lo, hi) for u, v in raw.items()}
    points = list(vectors.values())
    metrics, solutions = [], {}
    for k in (2, 3, 4):
        labels, centers, inertia = best_kmeans(points, k)
        solutions[k] = (labels, centers)
        metrics.append(dict(k=k, silhouette=silhouette(points, labels), inertia=inertia,
                            counts=[labels.count(i) for i in range(k)], runs=20))
    labels, centers = solutions[3]
    # Sakārto no lielākā līdz mazākajam laika/kļūdu un mazākajam sarežģītības rādītājam.
    order = sorted(range(3), key=lambda i: centers[i][0]+centers[i][1]+centers[i][2]-centers[i][3]-centers[i][4], reverse=True)
    centers = [centers[i] for i in order]
    labels = [order.index(l) for l in labels]
    stereo = [nearest(p, PROTOTYPES) for p in points]
    users = list(raw)
    differences = [dict(user=u, stereotype=NAMES[s], cluster=NAMES[l],
                        d_stereo=distance(vectors[u], PROTOTYPES[s]),
                        d_other=distance(vectors[u], PROTOTYPES[l]))
                   for u, s, l in zip(users, stereo, labels) if s != l]
    tests = []
    for u, ev in grouped(test, 'user_id').items():
        x = features(ev)
        z = normalize(x, lo, hi)
        label = nearest(z, centers)
        tests.append(dict(user=u, raw=x, normalized=z, cls=NAMES[label], distance=distance(z, centers[label])))
    # Pēc S4 fiksēti centri un normalizācija. Tālāki notikumi neietekmē apmācību.
    m03, m06 = raw['U01'][:], raw['U01'][:]
    current = labels[users.index('U01')]
    candidate, streak = None, 0
    timeline = []
    sessions = sorted(grouped(future, 'session_id').items(), key=lambda item: int(item[0].split('_S')[1]))
    for session, ev in sessions:
        observation = features(ev)
        m03 = [.3*x+.7*m for x, m in zip(observation, m03)]
        m06 = [.6*x+.4*m for x, m in zip(observation, m06)]
        z = normalize(m03, lo, hi)
        best = nearest(z, centers)
        old = current
        d_current, d_best = distance(z, centers[current]), distance(z, centers[best])
        if best != current and d_current-d_best > .1:
            streak = streak+1 if candidate == best else 1
            candidate = best
        else:
            candidate, streak = None, 0
        qualifying = streak
        if streak >= 2:
            current, candidate, streak = best, None, 0
        timeline.append(dict(session=session, observation=observation, ema03=m03[:], ema06=m06[:],
                             before=NAMES[old], nearest=NAMES[best], d_current=d_current,
                             d_best=d_best, advantage=d_current-d_best, streak=qualifying, after=NAMES[current]))
    return dict(seed=SEED, features=FEATURES, rows=len(train), users=len(raw), sessions=144,
                submits=sum(e['event_type']=='rule_submit' for e in train), raw=raw, lo=lo, hi=hi,
                prototypes=PROTOTYPES, stereotype_counts=[stereo.count(i) for i in range(3)],
                metrics=metrics, centers=centers, counts=[labels.count(i) for i in range(3)],
                assignments=dict(zip(users, [NAMES[l] for l in labels])), differences=differences,
                tests=tests, timeline=timeline, initial=raw['U01'], initial_class=NAMES[labels[0]])


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if '--generate' in sys.argv:
        generate()
    result = calculate()
    # Pilna precizitāte saglabāta aprēķinos un izvadē; atskaitē vērtības noapaļotas.
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
