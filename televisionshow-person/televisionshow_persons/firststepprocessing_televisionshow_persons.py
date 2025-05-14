import time, os, json, math, random, collections
from urllib.parse import unquote

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────
# 0. USER CONFIGURATION
# ─────────────────────────────────────────────────────────
DUMP_FILE          = "....dbpedia-2015-10.nt"
OUT_DIR            = "...pythonProject/ISWC2025/"
EDGE_CAP           = 70_000_000    # how many triples to read for PageRank
MAX_SUBJ_PER_CELL  = 1000          # cap subjects per (bin,quartile) cell
DAMPING            = 0.85
MAX_ITER           = 20

# ─────────────────────────────────────────────────────────
# 1. Harvest all Book & Writer URIs
# ─────────────────────────────────────────────────────────
def harvest_types():
    RDF   = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"
    TELEVISIONSHOW  = "<http://dbpedia.org/ontology/TelevisionShow>"
    PERSON  = "<http://dbpedia.org/ontology/Person>"
    televisionshows, persons = set(), set()
    t0 = time.time()
    with open(DUMP_FILE, "rt", encoding="utf-8", errors="ignore") as fh:
        for i, line in enumerate(fh,1):
            parts = line.rstrip(" .\n\t").split(None,2)
            if len(parts)!=3:
                continue
            s,p,o = parts
            if p==RDF:
                if   o==TELEVISIONSHOW:  televisionshows.add(s)
                elif o==PERSON: persons.add(s)
            if i%5_000_000==0:
                print(f"[types] {i:,} lines…")
    print(f"Harvested in {time.time()-t0:.1f}s → "
          f"Televisionshow={len(televisionshows):,}, Persons={len(persons):,}")
    return televisionshows, persons

# ─────────────────────────────────────────────────────────
# 2. Compute PageRank_in (incoming‐only)
# ─────────────────────────────────────────────────────────
def compute_pr_in():
    in_adj = collections.defaultdict(list)
    edges  = 0
    t0     = time.time()
    with open(DUMP_FILE, "rt", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            s,_,o = line.rstrip(" .\n\t").split(None,2)
            in_adj[o].append(s)
            edges += 1
            if edges >= EDGE_CAP:
                break
    print(f"In-adj built on {edges:,} triples in {(time.time()-t0)/60:.1f}m")

    nodes = list(in_adj.keys())
    idx   = {n:i for i,n in enumerate(nodes)}
    N     = len(nodes)

    outdeg = np.zeros(N, float)
    for srcs in in_adj.values():
        for src in srcs:
            j = idx.get(src)
            if j is not None:
                outdeg[j] += 1

    pr = np.full(N, 1/N, np.float64)
    for _ in range(MAX_ITER):
        new = np.zeros_like(pr)
        for tgt, srcs in in_adj.items():
            j = idx[tgt]
            for src in srcs:
                i = idx.get(src)
                if i is None or outdeg[i]==0:
                    continue
                new[j] += pr[i]/outdeg[i]
        pr = (1-DAMPING)/N + DAMPING*new

    pr_in = {nodes[i]: float(pr[i]) for i in range(N)}
    print("PageRank_in done")
    return pr_in

# ─────────────────────────────────────────────────────────
# 3. Patterns metadata (including avgOS & minOS)
# ─────────────────────────────────────────────────────────
def make_patterns_df():
    data = [
      # id   pred_local     freq   maxSO  avgSO  minSO
      ("P1","starring",      60547,    46,     4,     1),
      ("P2","director",     8321,     26,     1,     1),
      ("P3","executiveProducer",     7193,     11,     2,     1),
      ("P4","presenter",      6955,     25,     2,     1),
      ("P5","creator",          6755,     6,     1,     1),
      ("P6","voice",   6364,     31,     5,     1),
      ("P7","producer",   4764,     8,     1,     1),
      ("P8","author",        4612,     14,     1,     1),
      ("P9","composer",     2771,     10,     1,     1),
      ("P10","narrator",   1897,     9,     1,     1),
      ("P11","showJudge",         886,     15,     2,     1),
      ("P12","creativeDirector",    192,     4,     1,     1),
      ("P13", "storyEditor", 27, 3, 1, 1)
    ]
    df = pd.DataFrame(data, columns=["id","pred_local","sup","maxSO","avgSO","minSO"])
    ONTO = "http://dbpedia.org/ontology/"
    df["pred_iri"] = df.pred_local.apply(lambda s: f"<{ONTO}{s}>")
    return df

# ─────────────────────────────────────────────────────────
# 4. Extract & sample by (card_bin, PR_quartile) – drop empty
# ─────────────────────────────────────────────────────────
def extract_and_sample(televisionshows, persons, patterns, pr_in):
    pred2pid  = dict(zip(patterns.pred_iri, patterns.id))
    subj2objs = {pid:{} for pid in patterns.id}

    print("Collecting all (TelevisionShows→Person) pairs…")
    t0 = time.time()
    with open(DUMP_FILE, "rt", encoding="utf-8", errors="ignore") as fh:
        for i,line in enumerate(fh,1):
            if "dbpedia.org/ontology" not in line:
                continue
            s,p,o = line.rstrip(" .\n\t").split(None,2)
            pid   = pred2pid.get(p)
            if pid and s in televisionshows and o in persons:
                subj2objs[pid].setdefault(s, set()).add(o)
            if i%5_000_000==0:
                print(f"[pairs] {i:,} lines…")
    print(f"Done in {time.time()-t0:.1f}s")

    rows = []
    for _, pat in patterns.iterrows():
        pid    = pat.id
        sup    = pat.sup
        maxSO  = pat.maxSO
        avgSO  = pat.avgSO
        T      = math.ceil((avgSO + maxSO)/2)     # dynamic threshold
        mapping = subj2objs[pid]

        # 4.1 Cardinality buckets
        bins = {"single":[], "few":[], "many":[]}
        for s, objs in mapping.items():
            cnt = len(objs)
            if   cnt==1:    b="single"
            elif cnt<=T:    b="few"
            else:           b="many"
            bins[b].append(s)

        # 4.2 PR‐in quartiles per bucket
        for b, subjects in bins.items():
            if not subjects:
                continue
            subjects.sort(key=lambda u: pr_in.get(u,0.0))
            n    = len(subjects)
            size = math.ceil(n/4)
            quart2subs = {}
            for i in range(4):
                chunk = subjects[i*size:(i+1)*size]
                if chunk:
                    quart2subs[f"Q{i+1}"] = chunk

            # 4.3 sample and record
            for qlabel, subs in quart2subs.items():
                if len(subs) > MAX_SUBJ_PER_CELL:
                    subs = random.sample(subs, MAX_SUBJ_PER_CELL)
                pairs = [(s,o) for s in subs for o in mapping[s]]
                rows.append({
                  "pattern_id":  pid,
                  "pred_local":  pat.pred_local,
                  "sup":         sup,
                  "maxSO":       maxSO,
                  "avgSO":       avgSO,
                  "T_thresh":    T,
                  "card_bin":    b,
                  "pr_quartile": qlabel,
                  "n_subjects":  len(subs),
                  "n_pairs":     len(pairs),
                  "pairs":       pairs
                })

    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────
# 5. Generate QA from every subject (grouping all its objects)
# ─────────────────────────────────────────────────────────
def build_QA(df, patterns):
    from urllib.parse import unquote

    def nice(uri):
        return unquote(uri.strip("<>")\
                       .split("/resource/")[-1])\
                       .replace("_"," ")

    # singular vs. plural templates
    SINGULAR = {
      "starring":         'Who is the person starring at the televisionshow "{title}"?',
      "director":    'Who is the director of the televisionshow "{title}"?',
      "executiveProducer":    'Who is the executive producer of the televisionshow "{title}"?',
      "presenter":     'Who is the presenter of the televisionshow "{title}"?',
      "creator":         'Who is the creator of the televisionshow "{title}"?',
      "voice": 'Whose voice is in the televisionshow "{title}"?',
      "producer": 'Who is the producer of the televisionshow "{title}"?',
      "author":      'Who is the author of the televisionshow "{title}"?',
      "composer":   'Who is the composer of the televisionshow "{title}"?',
      "narrator":  'Who is the narrator of the televisionshow "{title}"?',
      "showJudge":        'Who is the show judge of the televisionshow "{title}"?',
      "creativeDirector":  'Who is the creative director of the televisionshow “{title}"?',
      "storyEditor":   'Who is the story editor of the televisionshow “{title}"?'
    }
    PLURAL = {
        "starring": 'Who are the people starring at the televisionshow "{title}"?',
        "director": 'Who are the directors of the televisionshow "{title}"?',
        "executiveProducer": 'Who are the executive producers of the televisionshow "{title}"?',
        "presenter": 'Who are the presenters of the televisionshow "{title}"?',
        "creator": 'Who are the creators of the televisionshow "{title}"?',
        "voice": 'Whose voices are in the televisionshow "{title}"?',
        "producer": 'Who are the producers of the televisionshow "{title}"?',
        "author": 'Who are the authors of the televisionshow "{title}"?',
        "composer": 'Who are the composers of the televisionshow "{title}"?',
        "narrator": 'Who are the narrators of the televisionshow "{title}"?',
        "showJudge": 'Who are the show judges of the televisionshow "{title}"?',
        "creativeDirector": 'Who are the creative directors of the televisionshow “{title}"?',
        "storyEditor": 'Who are the story editors of the televisionshow “{title}"?'
    }

    rows = []
    for _, row in df.iterrows():
        pid      = row.pattern_id
        pred     = row.pred_local
        card     = row.card_bin
        quart    = row.pr_quartile
        sup      = row.sup

        # group the flat list of (s,o) → subj→list of objects
        subj_map = {}
        for s, o in row.pairs:
            subj_map.setdefault(s, []).append(o)

        for s, obj_list in subj_map.items():
            title = nice(s)
            if len(obj_list) == 1:
                # singular
                q = SINGULAR[pred].format(title=title)
                a = nice(obj_list[0])
            else:
                # plural
                q = PLURAL[pred].format(title=title)
                a = ", ".join(nice(o) for o in obj_list)

            rows.append({
                "pattern_id":  pid,
                "predicate":   pred,
                "sup":         sup,
                "card_class":  card,
                "pr_quartile": quart,
                "question":    q,
                "answer":      a
            })

    qa_df = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, "televisionshow_persons_QA.csv")
    qa_df.to_csv(out_path, index=False)
    print(f"✓ Wrote {len(qa_df):,} QA pairs → {out_path}")
    return qa_df

# ─────────────────────────────────────────────────────────
# 6. SAVE to Parquet & CSV
# ─────────────────────────────────────────────────────────
def save(df):
    os.makedirs(OUT_DIR, exist_ok=True)
    df.drop(columns=["pairs"])\
      .to_parquet(os.path.join(OUT_DIR,"televisionshow_persons_by_bin_pr.parquet"),index=False)
    out = df.copy()
    out["pairs"] = out["pairs"].apply(json.dumps)
    out.to_csv(os.path.join(OUT_DIR,"televisionshow_persons_by_bin_pr.csv"), index=False)
    print("Saved patterns under", OUT_DIR)

# ─────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────
if __name__=="__main__":
    televisionshows, persons = harvest_types()
    pr_in          = compute_pr_in()
    patterns       = make_patterns_df()
    df             = extract_and_sample(televisionshows, persons, patterns, pr_in)

    # Show summary
    print(df[["pattern_id","card_bin","pr_quartile","n_subjects","n_pairs"]])
    save(df)
    qa_df = build_QA(df, patterns)
