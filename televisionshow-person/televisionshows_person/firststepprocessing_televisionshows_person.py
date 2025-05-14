import time
import os
import json
import math
import random
import collections
from urllib.parse import unquote

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────
# 0. USER CONFIGURATION
# ─────────────────────────────────────────────────────────
DUMP_FILE          = ".../dbpedia-2015-10.nt"
OUT_DIR            = ".../pythonProject/ISWC2025/"
EDGE_CAP           = 70_000_000    # how many triples to read for PageRank
MAX_SUBJ_PER_CELL  = 1000          # cap authors per (bin,quartile) cell
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
            if len(parts)!=3: continue
            s,p,o = parts
            if p==RDF:
                if   o==TELEVISIONSHOW:  televisionshows.add(s)
                elif o==PERSON: persons.add(s)
            if i%5_000_000==0:
                print(f"[types] {i:,} lines…")
    print(f"Harvested in {time.time()-t0:.1f}s → TELEVISIONSHOW={len(televisionshows):,}, PERSON={len(persons):,}")
    return televisionshows, persons

# ─────────────────────────────────────────────────────────
# 2. Compute PageRank_in (incoming‐only)
# ─────────────────────────────────────────────────────────
def compute_pr_in():
    in_adj, edges = collections.defaultdict(list), 0
    t0 = time.time()
    with open(DUMP_FILE, "rt", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            s,_,o = line.rstrip(" .\n\t").split(None,2)
            in_adj[o].append(s)
            edges += 1
            if edges >= EDGE_CAP: break
    print(f"In‐adj built on {edges:,} triples in {(time.time()-t0)/60:.1f}m")

    nodes = list(in_adj.keys())
    idx   = {n:i for i,n in enumerate(nodes)}
    N     = len(nodes)

    # compute out-degree for each node in this subgraph
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
# 3. Patterns metadata (including maxSO, avgSO, minSO)
# ─────────────────────────────────────────────────────────
def make_patterns_df():
    data = [
      # id     pred_local      sup   maxSO  avgSO  minSO
      ("P1","starring",      60547,    41,     2,     1),
      ("P2","director",     8321,     69,     3,     1),
      ("P3","executiveProducer",     7193,     156,     3,     1),
      ("P4","presenter",      6955,     33,     2,     1),
      ("P5","creator",          6755,     78,     2,     1),
      ("P6","voice",   6364,     145,     3,     1),
      ("P7","producer",   4764,     64,     2,     1),
      ("P8","author",        4612,     20,     2,     1),
      ("P9","composer",     2771,     101,     3,     1),
      ("P10","narrator",   1897,     25,     2,     1),
      ("P11","showJudge",         886,     10,     2,     1),
      ("P12","creativeDirector",    192,     64,     3,     1),
      ("P13", "storyEditor", 27, 1, 1, 1)
    ]
    df = pd.DataFrame(data, columns=["id","pred_local","sup","maxSO","avgSO","minSO"])
    ONTO = "http://dbpedia.org/ontology/"
    df["pred_iri"] = df.pred_local.map(lambda s: f"<{ONTO}{s}>")
    return df

# ─────────────────────────────────────────────────────────
# 4. Extract & sample by (card_class, PR_quartile) 
# ─────────────────────────────────────────────────────────
def extract_and_sample(televisionshows, persons, patterns, pr_in):
    pred2pid = dict(zip(patterns.pred_iri, patterns.id))
    pers2tvsh = {pid:{} for pid in patterns.id}

    print("Collecting all (Televisionshows→Person) pairs…")
    t0 = time.time()
    with open(DUMP_FILE, "rt", encoding="utf-8", errors="ignore") as fh:
        for i,line in enumerate(fh,1):
            if "dbpedia.org/ontology" not in line:
                continue
            s,p,o = line.rstrip(" .\n\t").split(None,2)
            pid   = pred2pid.get(p)
            if pid and s in televisionshows and o in persons:
                # invert: map writer o → book s
                pers2tvsh[pid].setdefault(o, set()).add(s)
            if i%5_000_000==0:
                print(f"[pairs] {i:,} lines…")
    print(f"Done in {time.time()-t0:.1f}s")

    rows = []
    for _, pat in patterns.iterrows():
        pid, sup, maxSO, avgSO, minSO = (
            pat.id, pat.sup, pat.maxSO, pat.avgSO, pat.minSO
        )
        T = math.ceil((avgSO + maxSO)/2)
        mapping = pers2tvsh[pid]

        # 4.1 Cardinality buckets over #televisionshows and person
        bins = {"single":[], "few":[], "many":[]}
        for auth, televisionshows in mapping.items():
            cnt = len(televisionshows)
            if   cnt == minSO:  b = "single"
            elif cnt <= T:      b = "few"
            else:               b = "many"
            bins[b].append(auth)

        # 4.2 PR‐in quartiles within each bucket
        for b, persons in bins.items():
            if not persons: continue
            persons.sort(key=lambda u: pr_in.get(u,0.0))
            n    = len(persons)
            size = math.ceil(n/4)
            quart2subs = {
              f"Q{i+1}": persons[i*size:(i+1)*size]
              for i in range(4)
            }
            # 4.3 sample and record
            for ql, chunk in quart2subs.items():
                if not chunk: continue
                if len(chunk) > MAX_SUBJ_PER_CELL:
                    chunk = random.sample(chunk, MAX_SUBJ_PER_CELL)
                pairs = [(auth, televisionshows) for auth in chunk for televisionshows in mapping[auth]]
                rows.append({
                  "pattern_id":  pid,
                  "pred_local":  pat.pred_local,
                  "sup":         sup,
                  "maxSO":       maxSO,
                  "avgSO":       avgSO,
                  "minSO":       minSO,
                  "T_thresh":    T,
                  "card_class":  b,
                  "pr_quartile": ql,
                  "n_subjects":  len(chunk),
                  "n_pairs":     len(pairs),
                  "pairs":       pairs
                })

    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────
# 5. Generate person‐centric QA from every televisionshow (grouping all its books)
# ─────────────────────────────────────────────────────────
def build_QA(df, patterns):
    def nice(uri):
        return unquote(uri.strip("<>")\
                       .split("/resource/")[-1])\
                       .replace("_"," ")

    SINGULAR = {
        "starring": "What television show did {author} star in?",
        "director": "What television show did {author} direct?",
        "executiveProducer": "What television show did {author} executive produce?",
        "presenter": "What television show did {author} present?",
        "creator": "What television show did {author} create?",
        "voice": "What television show did {author} provide voice work for?",
        "producer": "What television show did {author} produce?",
        "author": "What television show did {author} author?",
        "composer": "What television show did {author} compose music for?",
        "narrator": "What television show did {author} narrate?",
        "showJudge": "What television show did {author} judge?",
        "creativeDirector": "What television show did {author} serve as creative director for?",
        "storyEditor": "What television show did {author} edit the story for?"
    }

    PLURAL = {
        "starring": "What television shows did {author} star in?",
        "director": "What television shows did {author} direct?",
        "executiveProducer": "What television shows did {author} executive produce?",
        "presenter": "What television shows did {author} present?",
        "creator": "What television shows did {author} create?",
        "voice": "What television shows did {author} provide voice work for?",
        "producer": "What television shows did {author} produce?",
        "author": "What television shows did {author} author?",
        "composer": "What television shows did {author} compose music for?",
        "narrator": "What television shows did {author} narrate?",
        "showJudge": "What television shows did {author} judge?",
        "creativeDirector": "What television shows did {author} serve as creative director for?",
        "storyEditor": "What television shows did {author} edit the story for?"
    }

    rows = []
    for _, row in df.iterrows():
        pid   = row.pattern_id
        pred  = row.pred_local
        card  = row.card_class
        quart = row.pr_quartile
        sup   = row.sup

        # row.pairs are already (author,book)
        pers2tvsh = {}
        for pers, tvsh in row.pairs:
            pers2tvsh.setdefault(pers, []).append(tvsh)

        for pers, tvsh in pers2tvsh.items():
            a_nice = nice(pers)
            if len(tvsh) == 1:
                q = SINGULAR[pred].format(author=a_nice)
                a = nice(tvsh[0])
            else:
                q = PLURAL[pred].format(author=a_nice)
                a = ", ".join(nice(b) for b in tvsh)

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
    out = os.path.join(OUT_DIR, "tevelisionshows_person_centric_QA.csv")
    qa_df.to_csv(out, index=False)
    print(f"Wrote {len(qa_df):,} tevelisionshows_person-centric QA → {out}")
    return qa_df

# ─────────────────────────────────────────────────────────
# 6. Save pattern‐level slices
# ─────────────────────────────────────────────────────────
def save_slices(df):
    os.makedirs(OUT_DIR, exist_ok=True)
    df.drop(columns=["pairs"])\
      .to_parquet(os.path.join(OUT_DIR, "tevelisionshows_person_by_bin_pr.parquet"), index=False)
    js = df.copy(); js["pairs"] = js["pairs"].apply(json.dumps)
    js.to_csv(os.path.join(OUT_DIR, "tevelisionshows_person_by_bin_pr.csv"), index=False)
    print("✓ Saved pattern slices under", OUT_DIR)

# ─────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────
if __name__=="__main__":
    televisionshows, persons = harvest_types()
    pr_in          = compute_pr_in()
    patterns       = make_patterns_df()
    slices         = extract_and_sample(televisionshows, persons, patterns, pr_in)

    # show a quick summary
    print(slices[["pattern_id","card_class","pr_quartile","n_subjects","n_pairs"]])
    save_slices(slices)
    qa_df = build_QA(slices, patterns)
