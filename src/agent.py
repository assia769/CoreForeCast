"""Step 7 - LLM agent that EXPLAINS forecast errors. It never predicts.
Numbers come from the pipeline; the LLM only turns them into plain language."""
import json
import pandas as pd

def build_summary(sku, res_path="results/predictions.csv", best_path="results/wape_by_segment.csv"):
    res = pd.read_csv(res_path, parse_dates=["ds"])
    best = pd.read_csv(best_path).set_index("segment")["BEST"]
    g = res[res.unique_id == sku]
    segment = g.segment.iloc[0]
    model = best[segment]
    m = g[g.model == model].sort_values("ds")
    return {
        "sku": sku, "segment": segment, "model_used": model,
        "months": [d.strftime("%Y-%m") for d in m.ds],
        "forecast": [round(x, 1) for x in m.yhat],
        "actual": [int(x) for x in m.y],
        "total_forecast": round(m.yhat.sum(), 1), "total_actual": int(m.y.sum()),
    }

def build_prompt(summary):
    return ("You are an assistant for supply-chain planners working on remanufactured parts (cores).\n"
            "Explain the forecast result below in plain business language, 4 sentences max.\n"
            "Use ONLY the data provided. Do NOT invent causes; if a cause is unknown, say so.\n"
            "State whether we under- or over-forecast and what it means for core availability.\n\n"
            f"DATA: {json.dumps(summary)}")

def explain(summary, call_llm):
    """call_llm: any function(str) -> str (Claude API, DeepSeek, Ollama...)."""
    return call_llm(build_prompt(summary))

if __name__ == "__main__":
    s = build_summary("SKU_002")
    print(build_prompt(s))
    # Plug your LLM here, e.g. with Ollama:
    # import ollama
    # print(explain(s, lambda p: ollama.chat(model="llama3.1", messages=[{"role":"user","content":p}])["message"]["content"]))
