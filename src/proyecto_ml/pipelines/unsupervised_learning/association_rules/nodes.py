import pandas as pd
import numpy as np
from typing import Dict, Any

def _transactions(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    max_uniques = params.get("no_supervisado", {}).get("association", {}).get("max_uniques", 20)
    cats = [c for c in df.columns if df[c].dtype == object]
    cats = [c for c in cats if df[c].nunique() <= max_uniques]
    if not cats:
        return pd.DataFrame()
    T = pd.DataFrame(index=df.index)
    for c in cats:
        dummies = pd.get_dummies(df[c].astype(str), prefix=c)
        T = pd.concat([T, dummies], axis=1)
    T = T.astype(bool)
    return T

def run_apriori(dataset_con_features_temporales: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    T = _transactions(dataset_con_features_temporales, params)
    if T.empty:
        return pd.DataFrame(columns=["lhs", "rhs", "support", "confidence", "lift"])
    min_sup = params.get("no_supervisado", {}).get("association", {}).get("min_support", 0.05)
    n = len(T)
    sup_item = T.sum(axis=0) / n
    items = sup_item[sup_item >= min_sup].index.tolist()
    rules = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            A = items[i]; B = items[j]
            AB = (T[A] & T[B]).sum() / n
            if AB >= min_sup:
                conf_A_B = AB / sup_item[A]
                lift_A_B = conf_A_B / sup_item[B]
                rules.append({"lhs": A, "rhs": B, "support": AB, "confidence": conf_A_B, "lift": lift_A_B})
                conf_B_A = AB / sup_item[B]
                lift_B_A = conf_B_A / sup_item[A]
                rules.append({"lhs": B, "rhs": A, "support": AB, "confidence": conf_B_A, "lift": lift_B_A})
    df_rules = pd.DataFrame(rules)
    if not df_rules.empty:
        df_rules = df_rules.sort_values(["lift", "confidence", "support"], ascending=False)
    return df_rules

def run_fpgrowth(dataset_con_features_temporales: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    return run_apriori(dataset_con_features_temporales, params)