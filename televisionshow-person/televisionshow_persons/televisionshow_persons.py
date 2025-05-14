!pip -q install scikit-learn rouge-score nltk sentence-transformers bitsandbytes transformers accelerate sentencepiece

from google.colab import files
import os, torch, subprocess, re
import pandas as pd
import numpy as np
from tqdm import tqdm
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from sklearn.linear_model import LinearRegression
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer, util

# ──────────────────────────────────────────────────────────
# 0. PREP
# ──────────────────────────────────────────────────────────
nltk.download('punkt', quiet=True)
smooth     = SmoothingFunction().method1
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

uploaded = files.upload()            # pick your qa_tasks.csv
ORIG_CSV  = next(iter(uploaded))
qa        = pd.read_csv(ORIG_CSV)
print(f"Loaded {len(qa):,} questions from {ORIG_CSV}")

# ╔══════════════════════════════════════╗
# ║ 1. LOGIN TO GATED REPO               ║
# ╚══════════════════════════════════════╝
from huggingface_hub import login
login("token-to-be-used")

# ──────────────────────────────────────────────────────────
# 1. MODEL SETUP
# ──────────────────────────────────────────────────────────
if not torch.cuda.is_available():
    raise RuntimeError("Please switch to a GPU runtime")

cuda_ver = re.search(r"CUDA Version: (\d+\.\d+)",
                     subprocess.check_output("nvidia-smi").decode()).group(1)
print(f"CUDA {cuda_ver} detected, loading in 8‐bit with offload")

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
bnb_cfg  = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True
)

tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
tok.pad_token_id = tok.eos_token_id

offload_dir = "/content/offload"
os.makedirs(offload_dir, exist_ok=True)
llm = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_cfg,
    device_map="auto",
    offload_folder=offload_dir,
    max_memory={0:"14GiB","cpu":"30GiB"}
)

pipe = pipeline(
    "text-generation",
    model=llm,
    tokenizer=tok,
    pad_token_id=tok.eos_token_id,
    max_new_tokens=32,
    temperature=0.0
)

print("Model ready →",
      pipe("Who wrote The Old Man and the Sea?\nAnswer:", return_full_text=False)[0]["generated_text"].strip())

# ──────────────────────────────────────────────────────────
# 2. SIMILARITY FUNCTIONS & THRESHOLDS
# ──────────────────────────────────────────────────────────
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def bleu_sim(pred, ref): return sentence_bleu([ref.split()], pred.split(), smoothing_function=smooth)
def bert_sim(pred, ref):
    return util.cos_sim(
      bert_model.encode(pred,convert_to_tensor=True),
      bert_model.encode(ref, convert_to_tensor=True)
    ).item()
def edit_sim(a, b):
    la, lb = len(a), len(b)
    if max(la, lb)==0: return 1.0
    dp = np.zeros((la+1, lb+1), int)
    for i in range(la+1): dp[i,0]=i
    for j in range(lb+1): dp[0,j]=j
    for i in range(1, la+1):
      for j in range(1, lb+1):
        cost = 0 if a[i-1]==b[j-1] else 1
        dp[i,j] = min(dp[i-1,j]+1, dp[i,j-1]+1, dp[i-1,j-1]+cost)
    return 1 - dp[la,lb]/max(la,lb)
def jaccard_sim(a, b):
    sa,sb = set(a.split()), set(b.split())
    return 1.0 if not (sa|sb) else len(sa&sb)/len(sa|sb)
def exact_match(p, g): return int(p.strip().lower()==g.strip().lower())

TH = {
  'exact_match': 1.00,   # only perfect match
  'edit_sim':    0.80,   # ≥ 0.80
  'jaccard_sim': 0.50,   # ≥ 0.50
  'rouge_l':     0.70,   # ≥ 0.70
  'bleu':        0.50,   # ≥ 0.50
  'bert_cosine': 0.85    # ≥ 0.85 (> 0.84)
}

# ──────────────────────────────────────────────────────────
# 3. PROMPTS
# ──────────────────────────────────────────────────────────
zero_tpl = (
  "Answer the following question. Respond with just the author name(s),\n"
  "or 'unsure' if unknown. Do not include any other text.\n\n"
  "Question: {q}\nAnswer:"
)
few_tpl = (
  "Answer the following question with just the author name(s), or 'unsure' if unknown.\n"
  "Do not include any other text.\n\n"
  "Question: Who is starring in the television show \"Threads\"?\nAnswer: Reece Dinsdale\n\n"
  "Question: Who are starring in the television show \"Then Churchill Said to Me\"?\nAnswer: Nicholas Courtney, Frankie Howerd\n\n"
  "Question: {q}\nAnswer:"
)
modes = {"zero": zero_tpl, "few": few_tpl}

# ──────────────────────────────────────────────────────────
# 4. RUN BOTH MODES, BATCH GENERATE, SCORE, SAVE & DOWNLOAD
# ──────────────────────────────────────────────────────────
for mode, tpl in modes.items():
    print(f"\n▶ MODE = {mode}\n" + "─"*50)

    # 4.1 Generate
    prompts, answers = [tpl.format(q=q) for q in qa.question], []
    bs = 8
    for i in tqdm(range(0,len(prompts),bs), desc=f"Generating {mode}"):
        batch = prompts[i:i+bs]
        outs  = pipe(batch, return_full_text=False, batch_size=len(batch))
        answers += [o[0]["generated_text"].split("\n",1)[0].strip() for o in outs]
    qa["model_answer"] = answers

    # 4.2 Score + verdicts per metric
    recs=[]
    for row in tqdm(qa.itertuples(), total=len(qa), desc=f"Scoring {mode}"):
        p,g = row.model_answer.strip(), row.answer.strip()
        em  = exact_match(p,g)
        ed  = edit_sim(p,g)
        ja  = jaccard_sim(p,g)
        rl  = scorer.score(g,p)['rougeL'].fmeasure
        b4  = bleu_sim(p,g)
        bS  = bert_sim(p,g)

        v = {}
        for m,val in [
             ('exact_match',em),
             ('edit_sim',   ed),
             ('jaccard_sim',ja),
             ('rouge_l',    rl),
             ('bleu',       b4),
             ('bert_cosine',bS)
        ]:
            if p.lower()=='unsure':
                v[f"verdict_{m}"] = 'unsure'
            else:
                v[f"verdict_{m}"] = (
                  'correct' if val>=TH[m]
                  else 'hallucination'
                )

        recs.append({
          **row._asdict(),
          **v,
          "exact_match": em,
          "edit_sim":    ed,
          "jaccard_sim": ja,
          "rouge_l":     rl,
          "bleu":        b4,
          "bert_cosine": bS
        })

    df = pd.DataFrame(recs)

    # 4.3 Save raw results
    parq = f"/content/results_{mode}.parquet"
    csvf = f"/content/{os.path.splitext(ORIG_CSV)[0]}_{mode}.csv"
    df.to_parquet(parq, index=False)
    df.to_csv(csvf,     index=False)
    print(f"  • Saved → {parq}\n           {csvf}")
    files.download(parq)
    files.download(csvf)

    # 4.4 Aggregate per‐pattern & per‐metric accuracy
    grp = df.groupby(["pattern_id","predicate","sup","card_class"])
    agg = {f"acc_{m}%": (f"verdict_{m}", lambda x,m=m: (x=="correct").mean()*100)
            for m in ['exact_match','edit_sim','jaccard_sim','rouge_l','bleu','bert_cosine']}
    agg["n_q"] = ("pattern_id","size")
    metrics = grp.agg(**agg).reset_index()

    metcsv = f"/content/{os.path.splitext(ORIG_CSV)[0]}_metrics_{mode}.csv"
    metrics.to_csv(metcsv,index=False)
    print(f"Pattern-card_class metrics → {metcsv}")
    files.download(metcsv)

    # 4.4-b By-quartile
    grp2 = df.groupby([
        "pattern_id","predicate","sup","card_class","pr_quartile"
    ])
    metrics_q = grp2.agg(**agg).reset_index()

    metqcsv = f"/content/{os.path.splitext(ORIG_CSV)[0]}_metrics_{mode}_by_quartile.csv"
    metrics_q.to_csv(metqcsv,index=False)
    print(f"Pattern-card_class-quartile metrics → {metqcsv}")
    files.download(metqcsv)

    # ──────────────────────────────────────────────────────────
    # 4.5 Compute & store Elasticities for every metric & group
    # ──────────────────────────────────────────────────────────
    elastic_rows = []
    for m in ['exact_match','edit_sim','jaccard_sim','rouge_l','bleu','bert_cosine']:
        col = f"acc_{m}%"
        for group_key, df_group in [
          ("single",   metrics[metrics.card_class=="single"]),
          ("few",      metrics[metrics.card_class=="few"]),
          ("many",     metrics[metrics.card_class=="many"]),
          ("few+many", metrics[metrics.card_class.isin(["few","many"])])
        ]:
            sub = df_group[df_group.n_q>=2]
            if len(sub)>=2:
                X = np.log10(sub.sup.values).reshape(-1,1)
                y = sub[col].values
                coef = LinearRegression().fit(X,y).coef_[0]
            else:
                coef = np.nan
            elastic_rows.append({
              "metric": m,
              "group":  group_key,
              "elasticity": coef
            })

    # write elasticity table
    eldf = pd.DataFrame(elastic_rows)
    el_file = f"/content/elasticity_{mode}.csv"
    eldf.to_csv(el_file, index=False)
    print(f"  • Elasticities saved → {el_file}")
    files.download(el_file)

print("\nAll done — check your browser downloads for the generated files.")