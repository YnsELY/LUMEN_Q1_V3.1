"""Tableaux Markdown du rapport FORMIC-Ink, lus depuis results/ (et sealed/ après le gel). Usage : python -B tables_ink.py"""
import json, os, sys
R = "/workspace/LUMEN_Q1_V3.1/formic_inkling/results"; SE = "/workspace/work_ink/sealed"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ink_eval import boot_ratio
J = lambda p: json.load(open(p)) if os.path.exists(p) else None
fr = lambda x, d=4: ("%.*f" % (d, x)).replace(".", ",")
pc = lambda x, d=2: ("%+.*f %%" % (d, 100 * x)).replace(".", ",")
DOMS = ["agentic", "code", "math", "reasoning", "encyclopedic", "multilingual", "long", "instructions", "web"]
COMP = ["UD-IQ2_XXS", "UD-Q2_K_XL", "UD-IQ3_XXS"]


def main():
    fin = J(f"{R}/final_selected.json"); sel = J(f"{R}/selection.lock.json")
    rows = []
    for q in COMP:
        c = J(f"{R}/competitor_{q}.json")
        if c: rows.append((q, c["dev"], fin["competitors"][q]["conf"] if fin else None, c["load_report"]["tensor_bytes"]))
    for v in ("L0", "V1a", "V4", "V1b", "C1"):
        d = J(f"{R}/pi_dev_{v}.json"); i = J(f"{R}/variant_{v}.json")
        if d: rows.append(("Formic " + v, d, fin["conf"] if (fin and fin["selected"] == v) else None, i["info"]["text_bytes_real"]))
    iq2 = rows[0][1] if rows else None
    print("\n### P-I dev (54 fenêtres)\n\n| Modèle | octets texte | KL | p99 | top-1 | routage (8 experts communs) | top-1 expert changé | contre UD-IQ2_XXS [IC 95 %] |\n|---|---:|---:|---:|---:|---:|---:|---|")
    for n, d, c, b in rows:
        g = boot_ratio(d["per_window"], iq2["per_window"]) if iq2 and not n.startswith("UD-IQ2") else None
        print("| %s | %s | %s | %s | %s | %s | %s | %s |" % (n, f"{b:,}".replace(",", " "), fr(d["kl"]), fr(d["kl_p99"], 3), fr(100 * d["agree"], 2) + " %",
              fr(100 * d["routing"]["mean_overlap"], 2) + " %", fr(100 * d["routing"]["mean_top1_changed"], 2) + " %",
              ("%s [%s ; %s], %d/%d fenêtres" % (pc(g["gain"], 1), pc(g["ci95"][0], 1), pc(g["ci95"][1], 1), g["windows_won"], g["n"])) if g else ""))
    print("\n### Par domaine, P-I dev (KL)\n\n| Modèle | " + " | ".join(DOMS) + " |\n|---|" + "---:|" * len(DOMS))
    for n, d, c, b in rows: print("| %s | " % n + " | ".join(fr(d["by_domain"][x]["kl"]) if x in d["by_domain"] else "—" for x in DOMS) + " |")
    if fin:
        print("\n### P-I conf (52 fenêtres, scellée jusqu'au gel)\n\n| Modèle | KL | p99 | top-1 | routage | Formic contre ce modèle [IC 95 %] |\n|---|---:|---:|---:|---:|---|")
        for q in COMP:
            c = fin["competitors"][q]["conf"]; g = fin["competitors"][q]["gain_conf"]
            print("| %s | %s | %s | %s | %s | %s [%s ; %s], %d/%d |" % (q, fr(c["kl"]), fr(c["kl_p99"], 3), fr(100 * c["agree"], 2) + " %", fr(100 * c["routing"]["mean_overlap"], 2) + " %",
                  pc(g["gain"], 1), pc(g["ci95"][0], 1), pc(g["ci95"][1], 1), g["windows_won"], g["n"]))
        c = fin["conf"]
        print("| **Formic %s** | **%s** | %s | %s | %s | |" % (fin["selected"], fr(c["kl"]), fr(c["kl_p99"], 3), fr(100 * c["agree"], 2) + " %", fr(100 * c["routing"]["mean_overlap"], 2) + " %"))
        print("\n### Par domaine, P-I conf (KL)\n\n| Modèle | " + " | ".join(DOMS) + " |\n|---|" + "---:|" * len(DOMS))
        for q in COMP: print("| %s | " % q + " | ".join(fr(fin["competitors"][q]["conf"]["by_domain"][x]["kl"]) for x in DOMS if x in fin["competitors"][q]["conf"]["by_domain"]) + " |")
        print("| Formic | " + " | ".join(fr(fin["conf"]["by_domain"][x]["kl"]) for x in DOMS if x in fin["conf"]["by_domain"]) + " |")
        print("\n### Octets par bloc (paquet complet)\n\n| Bloc | fichiers | octets | poids E8 | bpw réel E8 |\n|---|---:|---:|---:|---:|")
        for k, v in sorted(fin["by_block"].items()):
            print("| %s | %d | %s | %s | %s |" % (k, v["files"], f"{v['bytes']:,}".replace(",", " "), f"{v['params']:,}".replace(",", " "), fr(v["bpw_e8"], 4) if v["bpw_e8"] else "—"))
        print("\nTotal sur disque : %s octets" % f"{fin['package_bytes_on_disk']:,}".replace(",", " "))


if __name__ == "__main__":
    main()
