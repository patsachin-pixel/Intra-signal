import re, xml.etree.ElementTree as ET
import numpy as np, pandas as pd, requests, yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

app = FastAPI()
POS = set("surge jump gain gains rally beat record upgrade order wins win profit growth strong bullish approval soars".split())
NEG = set("fall falls drop slump miss probe fraud raid downgrade loss weak bearish penalty default ban cut plunge".split())

ema = lambda s, n: s.ewm(span=n, adjust=False).mean()

def rsi(c, n=14):
    d = c.diff()
    g = d.clip(lower=0).ewm(alpha=1 / n).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1 / n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def load(sym):
    # DATA SOURCE: Yahoo (free, ~1-15 min delayed). Swap this one function for a
    # broker API (Zerodha Kite / Upstox / Angel SmartAPI / Dhan) for true real-time.
    df = yf.Ticker(sym + ".NS").history(period="5d", interval="5m")
    if len(df) < 60:
        raise HTTPException(404, f"No usable NSE data for {sym}")
    return df

def indicators(df):
    c, v = df.Close, df.Volume
    day = df[df.index.date == df.index[-1].date()]
    vwap = ((day.Close * day.Volume).cumsum() / day.Volume.cumsum()).iloc[-1]
    macd = ema(c, 12) - ema(c, 26); hist = macd - ema(macd, 9)
    tr = pd.concat([df.High - df.Low, (df.High - c.shift()).abs(), (df.Low - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14).mean().iloc[-1]
    r = rsi(c).iloc[-1]
    spike = v.iloc[-1] > 1.5 * v.tail(20).mean()
    up = c.iloc[-1] > df.Open.iloc[-1]
    votes = [
        ("EMA 9 vs 21", 1 if ema(c, 9).iloc[-1] > ema(c, 21).iloc[-1] else -1),
        (f"RSI {r:.0f}", 1 if 55 <= r < 70 else -1 if 30 < r < 45 else 0),
        ("MACD histogram", 1 if hist.iloc[-1] > 0 and hist.iloc[-1] > hist.iloc[-2] else -1 if hist.iloc[-1] < 0 and hist.iloc[-1] < hist.iloc[-2] else 0),
        (f"VWAP {vwap:.2f}", 1 if c.iloc[-1] > vwap else -1),
        ("Volume spike", (1 if up else -1) if spike else 0),
    ]
    return votes, float(atr)

def double(w, price, top):
    a = (w.High if top else w.Low).values
    v = (w.Low if top else w.High).values
    i = a.argmax() if top else a.argmin()
    m = np.abs(np.arange(len(a)) - i) >= 8
    if not m.any(): return False
    k = np.where(m)[0]
    j = k[a[m].argmax() if top else a[m].argmin()]
    lo, hi = sorted((i, j))
    if abs(a[i] - a[j]) / a[i] > 0.003: return False
    neck = v[lo:hi + 1].min() if top else v[lo:hi + 1].max()
    return price < neck if top else price > neck

def patterns(df):
    o, h, l, c = (df[k].values for k in ("Open", "High", "Low", "Close"))
    body = abs(c[-1] - o[-1]); lw = min(o[-1], c[-1]) - l[-1]; uw = h[-1] - max(o[-1], c[-1])
    out = []
    if c[-1] > o[-1] and c[-2] < o[-2] and c[-1] >= o[-2] and o[-1] <= c[-2]: out.append(("Bullish engulfing", 1))
    if c[-1] < o[-1] and c[-2] > o[-2] and c[-1] <= o[-2] and o[-1] >= c[-2]: out.append(("Bearish engulfing", -1))
    if lw > 2 * body and uw < body: out.append(("Hammer", 1))
    if uw > 2 * body and lw < body: out.append(("Shooting star", -1))
    if c[-1] > h[-21:-1].max(): out.append(("20-bar breakout", 2))
    if c[-1] < l[-21:-1].min(): out.append(("20-bar breakdown", -2))
    w = df.tail(60)
    if double(w, c[-1], True): out.append(("Double top confirmed", -2))
    if double(w, c[-1], False): out.append(("Double bottom confirmed", 2))
    return out

def news(sym):
    try:
        r = requests.get("https://news.google.com/rss/search", timeout=8,
                         params={"q": f"{sym} share NSE", "hl": "en-IN", "gl": "IN", "ceid": "IN:en"})
        items = []
        for it in ET.fromstring(r.content).iter("item"):
            t = it.findtext("title") or ""
            w = set(re.findall(r"[a-z]+", t.lower()))
            items.append({"title": t, "score": int(np.sign(len(w & POS) - len(w & NEG)))})
            if len(items) == 8: break
        return items
    except Exception:
        return []

@app.get("/api/analyze")
def analyze(symbol: str):
    sym = symbol.upper().replace(".NS", "").strip()
    df = load(sym)
    votes, atr = indicators(df)
    pats = patterns(df)
    nws = news(sym)
    ind = sum(s for _, s in votes)
    pat = max(-3, min(3, sum(s for _, s in pats)))
    nw = max(-3, min(3, sum(n["score"] for n in nws)))
    total = ind + pat + nw
    if total >= 4 and ind >= 2 and nw >= 0: sig = "BUY"
    elif total <= -4 and ind <= -2 and nw <= 0: sig = "SELL"
    else: sig = "WAIT"
    px = float(df.Close.iloc[-1]); d = 1 if sig == "BUY" else -1
    lv = None if sig == "WAIT" else {"entry": round(px, 2), "stop": round(px - d * 1.2 * atr, 2), "target": round(px + d * 1.8 * atr, 2)}
    return {"symbol": sym, "price": round(px, 2), "last_bar": df.index[-1].strftime("%d %b %H:%M"),
            "signal": sig, "score": {"indicators": int(ind), "patterns": int(pat), "news": int(nw), "total": int(total)},
            "votes": [{"name": n, "v": s} for n, s in votes], "patterns": [{"name": n, "v": s} for n, s in pats],
            "news": nws, "levels": lv}

import os
D = os.path.join(os.path.dirname(__file__), "static")
for _n in ["manifest.json", "sw.js", "icon-192.png", "icon-512.png"]:
    app.add_api_route("/" + _n, (lambda n=_n: FileResponse(os.path.join(D, n))), methods=["GET"])

@app.get("/")
def home():
    return FileResponse(os.path.join(D, "index.html"))
