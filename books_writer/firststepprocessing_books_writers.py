#!/usr/bin/env python3
# firststepprocessing.py

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
DUMP_FILE          = "/Users/USER/PycharmProjects/pythonProject/ISWC2025/dbpedia-2015-10.nt"
OUT_DIR            = "/Users/USER/PycharmProjects/pythonProject/ISWC2025/"
EDGE_CAP           = 70_000_000    # how many triples to read for PageRank
MAX_SUBJ_PER_CELL  = 1000          # cap authors per (bin,quartile) cell
DAMPING            = 0.85
MAX_ITER           = 20

# ─────────────────────────────────────────────────────────
# 1. Harvest all Book & Writer URIs
# ─────────────────────────────────────────────────────────
def harvest_types():
    RDF   = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"
    BOOK  = "<http://dbpedia.org/ontology/Book>"
    WRIT  = "<http://dbpedia.org/ontology/Writer>"
    books, writers = set(), set()
    t0 = time.time()
    with open(DUMP_FILE, "rt", encoding="utf-8", errors="ignore") as fh:
        for i, line in enumerate(fh,1):
            parts = line.rstrip(" .\n\t").split(None,2)
            if len(parts)!=3: continue
            s,p,o = parts
            if p==RDF:
                if   o==BOOK:  books.add(s)
                elif o==WRIT: writers.add(s)
            if i%5_000_000==0:
                print(f"[types] {i:,} lines…")
    print(f"Harvested in {time.time()-t0:.1f}s → Books={len(books):,}, Writers={len(writers):,}")
    return books, writers

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
      ("P1",   "author",      19810,    128,     4,     1),
      ("P2",   "illustrator",   369,     25,     2,     1),
      ("P3",   "coverArtist",   294,     21,     3,     1),
      ("P4",   "translator",    184,     17,     2,     1),
      ("P6",   "subsequentWork", 47,      3,     1,     1),
      ("P7",   "nonFictSubject", 42,      9,     2,     1),
      ("P8",   "publisher",      41,      4,     2,     1)
    ]
    df = pd.DataFrame(data, columns=["id","pred_local","sup","maxSO","avgSO","minSO"])
    ONTO = "http://dbpedia.org/ontology/"
    df["pred_iri"] = df.pred_local.map(lambda s: f"<{ONTO}{s}>")
    return df

# ─────────────────────────────────────────────────────────
# 4. Extract & sample by (card_class, PR_quartile) – drop empty
#    Now mapping **author → {books}** rather than book→authors
# ─────────────────────────────────────────────────────────
def extract_and_sample(books, writers, patterns, pr_in):
    pred2pid = dict(zip(patterns.pred_iri, patterns.id))
    auth2books = {pid:{} for pid in patterns.id}

    print("Collecting all (Book→Writer) pairs…")
    t0 = time.time()
    with open(DUMP_FILE, "rt", encoding="utf-8", errors="ignore") as fh:
        for i,line in enumerate(fh,1):
            if "dbpedia.org/ontology" not in line:
                continue
            s,p,o = line.rstrip(" .\n\t").split(None,2)
            pid   = pred2pid.get(p)
            if pid and s in books and o in writers:
                # invert: map writer o → book s
                auth2books[pid].setdefault(o, set()).add(s)
            if i%5_000_000==0:
                print(f"[pairs] {i:,} lines…")
    print(f"Done in {time.time()-t0:.1f}s")

    rows = []
    for _, pat in patterns.iterrows():
        pid, sup, maxSO, avgSO, minSO = (
            pat.id, pat.sup, pat.maxSO, pat.avgSO, pat.minSO
        )
        T = math.ceil((avgSO + maxSO)/2)
        mapping = auth2books[pid]

        # 4.1 Cardinality buckets over #books per author
        bins = {"single":[], "few":[], "many":[]}
        for auth, books in mapping.items():
            cnt = len(books)
            if   cnt == minSO:  b = "single"
            elif cnt <= T:      b = "few"
            else:               b = "many"
            bins[b].append(auth)

        # 4.2 PR‐in quartiles within each bucket
        for b, authors in bins.items():
            if not authors: continue
            authors.sort(key=lambda u: pr_in.get(u,0.0))
            n    = len(authors)
            size = math.ceil(n/4)
            quart2subs = {
              f"Q{i+1}": authors[i*size:(i+1)*size]
              for i in range(4)
            }
            # 4.3 sample and record
            for ql, chunk in quart2subs.items():
                if not chunk: continue
                if len(chunk) > MAX_SUBJ_PER_CELL:
                    chunk = random.sample(chunk, MAX_SUBJ_PER_CELL)
                pairs = [(auth, book) for auth in chunk for book in mapping[auth]]
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
# 5. Generate author‐centric QA from every author (grouping all its books)
# ─────────────────────────────────────────────────────────
def build_QA(df, patterns):
    def nice(uri):
        return unquote(uri.strip("<>")\
                       .split("/resource/")[-1])\
                       .replace("_"," ")

    SINGULAR = {
      "author":        'What book has {author} written?',
      "illustrator":   'What book has {author} illustrated?',
      "coverArtist":   'What book has {author} as the cover artist?',
      "translator":    'What book has {author} translated?',
      "subsequentWork":'What book is a subsequent work by {author}?',
      "nonFictSubject":'What non-fiction book is about {author}?',
      "publisher":     'What book has as publisher {author}?'
    }
    PLURAL = {
      "author":        'What books has {author} written?',
      "illustrator":   'What books has {author} illustrated?',
      "coverArtist":   'What books has {author} created the cover art for?',
      "translator":    'What books has {author} translated?',
      "subsequentWork":'What books are subsequent works by {author}?',
      "nonFictSubject":'What non-fiction books are about {author}?',
      "publisher":     'What books have as publisher {author}?'
    }

    rows = []
    for _, row in df.iterrows():
        pid   = row.pattern_id
        pred  = row.pred_local
        card  = row.card_class
        quart = row.pr_quartile
        sup   = row.sup

        # row.pairs are already (author,book)
        auth2books = {}
        for auth, book in row.pairs:
            auth2books.setdefault(auth, []).append(book)

        for auth, books in auth2books.items():
            a_nice = nice(auth)
            if len(books) == 1:
                q = SINGULAR[pred].format(author=a_nice)
                a = nice(books[0])
            else:
                q = PLURAL[pred].format(author=a_nice)
                a = ", ".join(nice(b) for b in books)

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
    out = os.path.join(OUT_DIR, "author_centric_QA.csv")
    qa_df.to_csv(out, index=False)
    print(f"✓ Wrote {len(qa_df):,} author-centric QA → {out}")
    return qa_df

# ─────────────────────────────────────────────────────────
# 6. Save pattern‐level slices
# ─────────────────────────────────────────────────────────
def save_slices(df):
    os.makedirs(OUT_DIR, exist_ok=True)
    df.drop(columns=["pairs"])\
      .to_parquet(os.path.join(OUT_DIR, "book_writer_by_bin_pr.parquet"), index=False)
    js = df.copy(); js["pairs"] = js["pairs"].apply(json.dumps)
    js.to_csv(os.path.join(OUT_DIR, "book_writer_by_bin_pr.csv"), index=False)
    print("✓ Saved pattern slices under", OUT_DIR)

# ─────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────
if __name__=="__main__":
    books, writers = harvest_types()
    pr_in          = compute_pr_in()
    patterns       = make_patterns_df()
    slices         = extract_and_sample(books, writers, patterns, pr_in)

    # show a quick summary
    print(slices[["pattern_id","card_class","pr_quartile","n_subjects","n_pairs"]])
    # save the sliced patterns
    save_slices(slices)
    # build and save author‐centric QA
    qa_df = build_QA(slices, patterns)
