"""
Standalone verification of every statistic reported in the paper.

Dependencies: pandas, numpy, statsmodels (+ openpyxl / pdfplumber only if the
KBA series must be re-parsed). NO torch, NO sentence-transformers, NO
llama_cpp, so it is immune to the CUDA/NCCL breakage in Colab.

Covers audit findings 3.1, 3.2, 3.3 and 3.5. Finding 3.4 (three-seed
labelling determinism) needs the classifier and is not reproduced here.

Usage in Colab:

    GDELT_PATH   = '/content/drive/MyDrive/osint/gdelt/gdelt_fuel/gdelt_fuel_de_v2.csv'
    KBA_SHARES   = '/content/drive/MyDrive/osint/kba_shares.csv'   # optional
    exec(open('verify_values.py').read())

If KBA_SHARES does not exist, run build_kba_shares() once with your FZ8 /
FZ8.5 / FZ28 directories to create it.
"""
import numpy as np, pandas as pd, warnings
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen
warnings.filterwarnings("ignore")

SEED        = globals().get("SEED", 1234)
BOOTSTRAP_N = globals().get("BOOTSTRAP_N", 1000)
MAXLAG      = 4
FUEL_TAGS   = "FUELPRICES|ECON_GASOLINEPRICE|ECON_HEATINGOIL"

# =====================================================================
# 1. Discourse series, read in chunks so the 1 GB file never loads whole
# =====================================================================
def build_discourse(path, chunksize=50_000):
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0, 0.0, 0])
    ev_total = 0
    ev_month = defaultdict(int)
    n_rows = 0
    for ch in pd.read_csv(path, usecols=["DATE", "V2Themes", "V2Tone"],
                          dtype=str, chunksize=chunksize):
        n_rows += len(ch)
        th   = ch["V2Themes"].fillna("").str.upper()
        fuel = th.str.contains(FUEL_TAGS, regex=True)
        ev   = th.str.contains("ELECTRIC_VEHICLE", regex=False)
        ev_total += int(ev.sum())
        dt   = pd.to_datetime(ch["DATE"].str[:8], format="%Y%m%d", errors="coerce")
        tone = pd.to_numeric(ch["V2Tone"].str.split(",").str[0], errors="coerce")
        mon  = dt.dt.to_period("M").astype(str)
        for m, f, t, e in zip(mon, fuel, tone, ev):
            if m == "NaT":
                continue
            a = agg[m]; a[0] += 1
            if e: ev_month[m] += 1
            if f:
                a[1] += 1
                if pd.notna(t): a[2] += t; a[3] += 1
    rows = sorted(agg.items())
    fuel_share = pd.Series({m: v[1] / v[0] for m, v in rows}).sort_index()
    fuel_tone  = pd.Series({m: (v[2] / v[3] if v[3] else np.nan)
                            for m, v in rows}).sort_index()
    n_fuel = sum(v[1] for _, v in rows)
    return fuel_share, fuel_tone, n_rows, n_fuel, ev_total, ev_month

# =====================================================================
# 2. Statistics. Function bodies match the notebook's originals.
# =====================================================================
def lead_lag(a, b):
    idx = a.index.intersection(b.index)
    x, y = a[idx].astype(float), b[idx].astype(float)
    res = {}
    for lag in range(-6, 7):
        s = x.shift(lag); v = s.notna() & y.notna()
        res[lag] = s[v].corr(y[v]) if v.sum() > 5 else np.nan
    best = max(res, key=lambda k: abs(res[k]) if not np.isnan(res[k]) else 0)
    return res, best

def boot_ci(a, b, n_bootstrap=BOOTSTRAP_N):
    idx = a.index.intersection(b.index)
    m = pd.DataFrame({"x": a[idx].astype(float),
                      "y": b[idx].astype(float)}).dropna()
    A, B, n = m.x.values, m.y.values, len(m)
    lags = []
    for _ in range(n_bootstrap):
        j = np.random.choice(n, size=n, replace=True)
        d, k = A[j], B[j]; br, bl = -1, 0
        for lag in range(-6, 7):
            if lag > 0 and n > lag:        r = np.corrcoef(d[lag:], k[lag:])[0, 1]
            elif lag < 0 and n > abs(lag): r = np.corrcoef(d[:lag], k[:lag])[0, 1]
            else:                          r = np.corrcoef(d, k)[0, 1]
            if not np.isnan(r) and abs(r) > br: br, bl = abs(r), lag
        lags.append(bl)
    return np.percentile(lags, 2.5), np.percentile(lags, 97.5)

def granger(a, b, difference):
    idx = a.index.intersection(b.index)
    x, y = a[idx].astype(float), b[idx].astype(float)
    if difference: x, y = x.diff(), y.diff()
    g = pd.DataFrame({"kba": y, "disc": x}).dropna()
    r = grangercausalitytests(g[["kba", "disc"]], maxlag=MAXLAG, verbose=False)
    return [r[l][0]["ssr_ftest"][1] for l in range(1, MAXLAG + 1)], len(g)

# =====================================================================
# 3. Report
# =====================================================================
def main(gdelt_path, kba_shares_path):
    print("=" * 74)
    print("VERIFICATION OF REPORTED VALUES")
    print("=" * 74)

    fuel_share, fuel_tone, n_rows, n_fuel, ev_total, ev_month = \
        build_discourse(gdelt_path)

    import os
    if not os.path.exists(kba_shares_path):
        fz8  = globals().get("KBA_FZ8_DIR")
        fz28 = globals().get("KBA_FZ28_DIR")
        if fz8 and fz28:
            print(f"\n{kba_shares_path} not found. Parsing KBA releases from "
                  f"Drive:\n  FZ8  {fz8}\n  FZ28 {fz28}")
            build_kba_shares(fz8, fz28, out=kba_shares_path)
        else:
            raise FileNotFoundError(
                f"{kba_shares_path} not found and KBA_FZ8_DIR / KBA_FZ28_DIR "
                "are not defined. Set both, or supply a kba_shares.csv.")
    kba = pd.read_csv(kba_shares_path).set_index("period")
    kba_diesel, kba_bev, kba_hybrid = (kba.diesel_share, kba.bev_share,
                                       kba.hybrid_share)

    # ---- corpus + finding 3.5 -------------------------------------
    print(f"\n[corpus]  records {n_rows:,} | fuel-tagged {n_fuel:,} "
          f"({100*n_fuel/n_rows:.1f}%) | months {len(fuel_share)}")
    print(f"[3.5]     ELECTRIC_VEHICLE records: {ev_total:,}")
    print(f"[3.5]     months with >=1 such record: "
          f"{sum(1 for v in ev_month.values() if v)}")

    # ---- ADF -------------------------------------------------------
    print("\n[ADF on levels]")
    for nm, s in [("fuel_tag_share", fuel_share), ("fuel_record_tone", fuel_tone),
                  ("kba_diesel_share", kba_diesel), ("kba_bev_share", kba_bev),
                  ("kba_hybrid_share", kba_hybrid)]:
        st, p, *_ = adfuller(pd.Series(s).dropna().astype(float), autolag="AIC")
        print(f"  {nm:18s} ADF={st:7.3f}  p={p:.4f}  "
              f"{'stationary' if p < .05 else 'NON-stationary'}")

    pairs = [("Fuel-tag -> diesel", fuel_share, kba_diesel),
             ("Fuel-tag -> BEV",    fuel_share, kba_bev),
             ("Fuel-tag -> hybrid", fuel_share, kba_hybrid),
             ("Tone -> diesel",     fuel_tone,  kba_diesel)]

    # ---- Granger, both transformations -----------------------------
    print("\n[Granger ssr-F p-values]        n    L1     L2     L3     L4")
    lvl, dif = [], []
    for name, a, b in pairs:
        for mode in (False, True):
            ps, n = granger(a, b, mode)
            (dif if mode else lvl).extend(ps)
            tag = "diff  " if mode else "levels"
            print(f"  {name:20s} [{tag}] {n:3d}  " +
                  "  ".join(f"{p:.3f}" for p in ps))
    print(f"\n  smallest LEVELS      p = {min(lvl):.4f}")
    print(f"  smallest DIFFERENCED p = {min(dif):.4f}")
    print(f"  Bonferroni (16 tests)  = {0.05/16:.4f}")

    # ---- findings 3.2 and 3.3 --------------------------------------
    print("\n[3.2 / 3.3  differenced lead-lag and bootstrap]")
    np.random.seed(SEED)
    d = lambda s: s.diff().dropna()
    print(f"  {'pair':22s} {'lag':>4} {'r':>8}   95% CI")
    for name, a, b in pairs:
        res, best = lead_lag(d(a), d(b))
        lo, hi = boot_ci(d(a), d(b))
        print(f"  {name:22s} {best:+4d} {res[best]:+8.3f}   [{lo:+.1f}, {hi:+.1f}]")

    print("\n[levels lead-lag, for comparison]")
    for name, a, b in pairs:
        res, best = lead_lag(a, b)
        print(f"  {name:22s} {best:+4d} {res[best]:+8.3f}")

    # ---- finding 3.1 -----------------------------------------------
    print("\n[3.1  Johansen trace test, non-stationary outcomes only]")
    for name, ks in [("diesel share", kba_diesel), ("hybrid share", kba_hybrid)]:
        idx = fuel_share.index.intersection(ks.index)
        m = pd.DataFrame({"d": fuel_share[idx].astype(float),
                          "k": ks[idx].astype(float)}).dropna()
        j = coint_johansen(m.values, det_order=0, k_ar_diff=1)
        v = "reject H0" if j.lr1[0] > j.cvt[0, 1] else "cannot reject H0"
        print(f"  fuel-tag vs {name:14s} trace(r<=0)={j.lr1[0]:6.3f}  "
              f"95% crit={j.cvt[0,1]:6.3f}  -> {v}")

    print("\n" + "=" * 74)
    print("Compare against the paper. Deviations beyond the third decimal")
    print("should be reported so the manuscript can be corrected.")
    print("=" * 74)


def _fz8_pdf(filepath):
    """FZ 8 monthly PDF (2022). Mirrors parse_fz8_pdf in the notebook."""
    import re, os, pdfplumber
    fname = os.path.basename(filepath)
    m = re.search(r"(\d{4})(\d{2})", fname)
    year, month = (m.group(1), m.group(2)) if m else ("0000", "00")
    result = {"period": f"{year}-{month}"}

    def extract_nums(line):
        clean = re.sub(r"\s+r\s+", " ", line)
        tokens, nums, i = clean.split(), [], 0
        while i < len(tokens):
            if tokens[i].isdigit():
                if i + 1 < len(tokens) and re.match(r"^\d{3}$", tokens[i + 1]):
                    v = int(tokens[i] + tokens[i + 1])
                    if 100 <= v <= 999999:
                        nums.append(v); i += 2; continue
                v = int(tokens[i])
                if 100 <= v <= 999999:
                    nums.append(v)
            i += 1
        return nums

    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text or "Zusammen" not in text or "Elektro (BEV)" not in text \
               or "Hubraum" not in text:
                continue
            current = None
            for line in text.splitlines():
                ls = re.sub(r"[\u00a0\u202f\u2009\u2007\u2008\u200a\u0020\t]+",
                            " ", line).strip()
                if ls == "Benzin":  current = "benzin";  continue
                if ls == "Diesel":  current = "diesel";  continue
                nums = extract_nums(ls)
                if not nums:
                    continue
                if ls.startswith("Zusammen") and current in ("benzin", "diesel"):
                    result.setdefault(current, nums[0]); current = None
                elif ls.startswith("Elektro (BEV)"):
                    result.setdefault("bev", nums[0])
                elif ls.startswith("Hybrid"):
                    result.setdefault("hybrid", nums[0])
                elif ls.startswith("Insgesamt"):
                    result.setdefault("total", nums[0])
    return result


def _kba_xlsx(filepath, sheet_key="8.5"):
    """FZ 8.5 monthly workbook (2023 onward). Mirrors parse_kba_xlsx."""
    import re, os
    fname = os.path.basename(filepath)
    m = re.search(r"(\d{4})_?(\d{2})(?:_|\.|$)", fname)
    year, month = (m.group(1), m.group(2)) if m else ("0000", "00")
    result = {"period": f"{year}-{month}"}
    xl = pd.ExcelFile(filepath)
    target = next((s for s in xl.sheet_names if sheet_key in s), None)
    if target is None:
        return result
    sheet = pd.read_excel(filepath, sheet_name=target, header=None, dtype=str)

    def first_num(vals, lo, hi):
        for v in vals:
            try:
                x = int(str(v).replace(".", "").replace(",", ""))
                if lo <= x <= hi:
                    return x
            except (ValueError, TypeError):
                continue
        return None

    current = None
    for _, row in sheet.iterrows():
        vals = [str(x).strip() for x in row.values if str(x).strip() != "nan"]
        if not vals:
            continue
        first = vals[0]
        if first == "Benzin":   current = "benzin"
        elif first == "Diesel": current = "diesel"
        elif first == "Zusammen" and current in ("benzin", "diesel"):
            v = first_num(vals[1:], 1000, 999999)
            if v is not None: result.setdefault(current, v)
            current = None
        elif first == "Elektro (BEV)":
            v = first_num(vals[1:], 1000, 999999)
            if v is not None: result.setdefault("bev", v)
        elif first == "Hybrid":
            v = first_num(vals[1:], 1000, 999999)
            if v is not None: result.setdefault("hybrid", v)
        elif first == "Insgesamt":
            v = first_num(vals[1:], 1000, 9999999)
            if v is not None: result.setdefault("total", v)
    return result


def _fz28_xlsx(filepath):
    """FZ 28.1 workbook. Mirrors parse_fz28_xlsx in the notebook."""
    import re, os
    fname = os.path.basename(filepath)
    m = re.search(r"(\d{4})_(\d{2})", fname)
    year, month = (m.group(1), m.group(2)) if m else ("0000", "00")
    result = {"period": f"{year}-{month}"}
    xl = pd.ExcelFile(filepath)
    target = next((s for s in xl.sheet_names if "28.1" in s), None)
    if target is None:
        return result
    sheet = pd.read_excel(filepath, sheet_name=target, header=None, dtype=str)

    def safe_int(v):
        try:
            x = int(str(v).replace(".", "").replace(",", "").strip())
            return x if x > 0 else None
        except (ValueError, TypeError):
            return None

    for _, row in sheet.iterrows():
        vals = [str(x).strip() for x in row.values]
        if len(vals) < 11 or vals[1] != "Personenkraftwagen":
            continue
        if vals[2] == "nan":
            continue
        result["total"]  = safe_int(vals[2])
        result["bev"]    = safe_int(vals[7])
        result["hybrid"] = safe_int(vals[10])
        break
    return result


def build_kba_shares(fz8_dir, fz28_dir, out="kba_shares.csv"):
    """Parse the KBA releases straight from Drive and write the share series.

    fz8_dir  : holds the 2022 FZ 8 monthly PDFs and the 2023+ FZ 8.5 workbooks
    fz28_dir : holds the FZ 28 workbooks (battery-electric and hybrid counts)

    Needs pdfplumber and openpyxl only. No ML packages, so it is unaffected
    by the torch/NCCL problem.
    """
    import glob, os
    fz8_rows = []
    for f in sorted(glob.glob(os.path.join(fz8_dir, "**", "*"), recursive=True)):
        low = f.lower()
        try:
            if low.endswith(".pdf"):
                fz8_rows.append(_fz8_pdf(f))
            elif low.endswith((".xlsx", ".xlsm")):
                fz8_rows.append(_kba_xlsx(f))
        except Exception as e:
            print(f"  skipped {os.path.basename(f)}: {e}")
    fz28_rows = []
    for f in sorted(glob.glob(os.path.join(fz28_dir, "**", "*.xlsx"),
                              recursive=True)):
        try:
            fz28_rows.append(_fz28_xlsx(f))
        except Exception as e:
            print(f"  skipped {os.path.basename(f)}: {e}")

    fz8 = pd.DataFrame([r for r in fz8_rows if "diesel" in r and "total" in r])
    fz28 = pd.DataFrame([r for r in fz28_rows if "bev" in r and "total" in r])
    print(f"parsed {len(fz8)} FZ8/FZ8.5 months, {len(fz28)} FZ28 months")
    if fz8.empty or fz28.empty:
        raise RuntimeError("no months parsed; check the directory paths")

    fz8  = fz8.set_index("period").sort_index()
    fz28 = fz28.set_index("period").sort_index()
    out_df = pd.DataFrame({
        "diesel_share": fz8["diesel"].astype(float) / fz8["total"].astype(float),
        "bev_share":    fz28["bev"].astype(float)  / fz28["total"].astype(float),
        "hybrid_share": fz28["hybrid"].astype(float) / fz28["total"].astype(float),
    }).dropna()
    out_df.index.name = "period"
    out_df.to_csv(out)
    print(f"wrote {out} with {len(out_df)} months "
          f"({out_df.index.min()} to {out_df.index.max()})")
    return out_df


def _interactive():
    """True inside IPython/Jupyter, where sys.argv belongs to the kernel."""
    try:
        get_ipython  # noqa: F821
        return True
    except NameError:
        return False


if _interactive():
    # Executed via exec() in a notebook. Do not touch sys.argv, which holds
    # the kernel's own flags. Use the paths already defined in the session,
    # falling back to the working directory.
    _gp = globals().get("GDELT_PATH", "gdelt_fuel_de_v2.csv")
    _kp = globals().get("KBA_SHARES", globals().get("KBA_SHARES_PATH",
                                                    "kba_shares.csv"))
    print(f"notebook mode\n  GDELT_PATH = {_gp}\n  KBA_SHARES = {_kp}\n")
    main(_gp, _kp)
elif __name__ == "__main__":
    import sys
    gp = sys.argv[1] if len(sys.argv) > 1 else "gdelt_fuel_de_v2.csv"
    kp = sys.argv[2] if len(sys.argv) > 2 else "kba_shares.csv"
    main(gp, kp)
