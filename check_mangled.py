import pandas as pd, re

c = pd.read_csv(r"results\hallucination_claims_s1_s3.csv")
c["sentence"] = c["sentence"].astype(str)

meta = c["sentence"].str.contains(
    r"(?:stated|mentioned|described|outlined|specified|noted|highlighted|emphasi[sz]ed)\s+in\s*,"
    r"|^(?:According to|As per|Per)\s*,"
    r"|\bin\s*,\s*which\b",
    case=False, regex=True)

print("total claims:", len(c))
print("mangled meta-sentences:", meta.sum())
print()
print(pd.crosstab(c[meta]["system"], c[meta]["label"]).to_string())
print()
print("share of each system's claims that are mangled:")
print((c[meta].groupby("system").size() / c.groupby("system").size()).round(3).to_string())
