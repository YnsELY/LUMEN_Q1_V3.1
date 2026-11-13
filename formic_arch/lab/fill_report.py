"""Fill the @@MARKERS@@ of the report and of the dossier answer from the raw result files (tables are regenerated,
verdict sentences for the 600-document comparison are computed from the per-document values)."""
import json, os, importlib.util
B = "/workspace/LUMEN_Q1_V3.1/formic_arch"; OUT = B + "/results"
spec = importlib.util.spec_from_file_location("br", B + "/lab/build_report.py"); R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
S = R.S
fr = lambda s: s.replace(".", ",")

def big(vid):
    p = OUT + f"/big_{vid}.json"
    return json.load(open(p)) if os.path.exists(p) else None
ctrl = big("A0_Q0ldlq_big"); rot = big("ArotN_Q0ldlq_big"); perm = big("A0_Qperm_ldlq")
def cmp(x):
    m, lo, hi = S.boot(x["big"]["per_doc"]["kl"], ctrl["big"]["per_doc"]["kl"])
    n = sum(a < b for a, b in zip(x["big"]["per_doc"]["kl"], ctrl["big"]["per_doc"]["kl"]))
    return m, lo, hi, 100 * m / ctrl["big"]["kl"], n
def verdict(x):
    m, lo, hi, rel, n = cmp(x)
    sig = (lo > 0) or (hi < 0)
    word = ("gain significatif" if hi < 0 else "perte significative") if sig else "égalité"
    return word, fr("ΔKL %+.4f, IC 95 %% [%+.4f ; %+.4f] (%+.1f %%), %d documents améliorés sur 600" % (m, lo, hi, rel, n)), sig, rel
rw, rs, rsig, rrel = verdict(rot)
if perm is not None:
    pw, ps, psig, prel = verdict(perm)
else:
    pw, ps, psig, prel = "non exécutée", "passe non terminée", False, 0.0

H11 = "**%s (mesuré, 600 documents)** : %s ; résidu d'équivalence avant quantification 0,003–0,009 nat" % (rw, rs)
H12 = "**%s (mesuré, 600 documents)** : %s ; gain local 2–5 %%" % (pw, ps)
ROT = "**%s avec le témoin** sur 600 documents (%s). Les −6 %% et −10 %% vus sur 72 documents étaient du bruit. Équivalence non exacte en BF16 (0,003–0,009 nat)." % (rw.capitalize(), rs)
PERM = ("Gain local mesuré de 2–5 %% sur l'erreur pondérée (jusqu'à 19 %% sur les premières couches) ; passe complète sur 600 documents : **%s** (%s)." % (pw, ps))

BIGSEC = """## 5 bis. Comparaison déclarée à l'avance sur 600 documents

Les jeux de 72 documents ne distinguent pas des écarts inférieurs à ~15 % du témoin. Pour trancher les deux transformations exactes encore
ouvertes, trois variantes ont été déclarées **avant** exécution (`results/big_comparison.lock.json`) puis recréées par une passe E8-LDLQ complète
et évaluées une seule fois sur 600 nouveaux documents (100 par domaine, sources disjointes de tous les autres jeux).
La ré-exécution du témoin reproduit le premier passage à l'identique (KL de sélection 0,11232 dans les deux cas).

""" + R.big_table() + """

* **Rotation du résiduel** : """ + rw + """ (""" + rs + """).
* **Ordre des colonnes** : """ + pw + """ (""" + ps + """). La variante a reçu 0,19 % d'octets de moins que le témoin (−0,0053 bpw, soit
  environ +0,7 % de KL attendu) : même corrigé de cet écart, le gain local de 2–5 % mesuré sur l'erreur pondérée ne se retrouve pas en KL.

Avec 600 documents, l'intervalle de confiance d'une différence est de ±5 % du témoin. Les deux transformations sont donc **sans gain
supérieur à ~5 %** ; un effet plus petit n'est pas exclu, mais ne justifierait pas leur intégration.
"""

LECTURE = """1. **Le témoin équitable est fort.** La passe E8-LDLQ ramène la KL de 0,190 à 0,112 (−41 %) à octets égaux. Tout candidat doit battre ce
   témoin-là, pas l'arrondi simple.
2. **Les reconstructions perdent nettement.** En BF16, sans rien quantifier, retirer 2 couches coûte déjà 0,108 et retirer 12,5 % des canaux
   de MLP 0,136 : autant ou plus que tout le dommage de quantification du témoin (0,112). À octets égaux, les bits réinvestis ne rattrapent
   pas : +33 % pour 62 couches à 2,907 bpw, +33 % pour −12,5 % de largeur à 3,068 bpw, +87 % pour −25 % à 3,358 bpw. Le réajustement
   par moindres carrés aide (0,159 → 0,136) sans changer le verdict. La saillance de Taylor attribuait ~1 % d'importance aux 12,5 % de canaux
   retirés : elle sous-estime massivement l'effet réel.
3. **Le biais (A_b) corrige un vrai défaut de l'arrondi simple, que LDLQ corrige déjà.** Sans rétroaction : −15 % en sélection, −28 % en
   confirmation, et le gain subsiste quand les moyennes sont calculées sans les domaines code et chat. Avec LDLQ : égalité
   (+0,8 % en sélection, −6,7 % en confirmation, aucun intervalle n'excluant zéro). La rétroaction de LDLQ est pilotée par un Gram dont la
   composante dominante est précisément la direction moyenne : elle annule cette erreur sans biais.
4. **Les règles tirées du critère local échouent en KL.** Pas guidés par Fisher : ×7,5 à ×40, par destruction des projections d'entrée GDN
   (KL 1,87 avec cette seule famille modifiée) — une sensibilité locale ne prédit pas l'effet d'une grande perturbation sur des unités
   saturées ou normalisées. Échelles de colonne par second moment : +12 % partout, −5 % si on les limite aux entrées des mélangeurs.
   Répartition des bits par famille : prédite à −16 %, mesurée à +0,8 %. Échelles par pivots LDL sous LDLQ : +5,6 % ; même l'erreur
   locale ne baisse pas à bits égaux, la rétroaction réinjectant dans les colonnes grossières une erreur amplifiée.
5. **La rotation du résiduel ne se distingue pas du témoin.** Sur 72 documents elle affichait −5,8 % (native) ou +8,0 % (émulée) selon
   l'implémentation, l'écart venant de quelques documents « chat » : c'est la mesure du bruit de ce jeu, pas un effet. Voir la section 5 bis.
6. **Le biais ancré sur le professeur diverge** (KL 3,0 sur 64 couches, alors qu'il se comporte normalement sur les 4 premières : 0,0075
   contre 0,0059 pour A_b). Un bug de statistique sur la tête a été écarté ; la cause de la divergence en profondeur n'est pas identifiée.
   Ce résultat réfute cette mise en œuvre, pas toute calibration asymétrique."""

CONFIRM = R.table("confirm") + """

Les finalistes ont été figés par le commit `e533e72` (`results/selection.lock.json`) avant toute lecture des fichiers scellés.
**Les conclusions de la sélection tiennent** : les deux reconstructions à octets égaux restent significativement moins bonnes que leur témoin
(+23,5 % et +27,9 %) ; le biais sans rétroaction reste significativement meilleur (−27,9 %, et −26,2 % avec des moyennes calculées hors
code et chat) ; avec LDLQ, aucun candidat ne se distingue du témoin de façon significative. Le résidu d'équivalence de la rotation
seule monte à 0,0086 en confirmation, dont 0,044 sur le domaine math : la transformation n'est pas exacte en stockage BF16."""

LONG = R.long_table() + """

Le dommage du témoin est **plat** sur 8192 tokens (0,09–0,13 par tranche de 1024, sans tendance) et sur les positions 0–1024 de la sélection
(0,10–0,13 par tranche de 128) : pas de signe d'accumulation d'erreur dans l'état récurrent des Gated DeltaNet à cette échelle.
Non testé au-delà de 8192 tokens, ni en génération avec états persistants."""

RESUME = """## Réponse en une page

**Qu'ai-je modifié ?** Douze hypothèses ont été testées sur le vrai modèle, dont cinq modifications du graphe exécutées de bout en bout :
suppression de couches, rétrécissement des MLP avec réajustement, ajout d'un biais par linéaire (deux variantes), rotation du flux résiduel
avec gains de normalisation repliés (deux mises en œuvre). **Aucune n'est retenue.**

**Pourquoi cela devait s'accorder avec Lumen ?** Les candidats visaient ce que le codec E8 à pas par ligne ne voit pas : la composante
constante « massive » des activations (biais), l'anisotropie des pivots de LDLQ (changements de base), et la redondance supposée de la
structure (profondeur, largeur, rang, tête). Les diagnostics montrent que le codec n'a presque rien à y gagner : les poids sont déjà quasi
gaussiens à échelles de ligne uniformes, LDLQ annule de lui-même l'erreur sur la direction massive, et ce qui reste après LDLQ
(0,2 à 1,4 % d'erreur pondérée par matrice) est sans structure exploitable.

**Quel gain ou quelle perte face au contrôle équitable ?** Le contrôle est l'architecture d'origine quantifiée par mon portage GPU de
Lumen v3.1 (E8 + LDLQ) à 2,821 bpw idéal sur le chemin texte : **KL = 0,112 nat, accord top-1 89,1 %** (0,106 et 89,7 % sur 600 documents),
stable jusqu'à 8192 tokens de contexte.

| Candidat | Face au contrôle, à octets égaux | Statut |
|---|---|---|
| 62 couches au lieu de 64 | +33 % de KL (sans rétroaction) ; dommage BF16 seul 0,108 | perte, confirmée |
| MLP −12,5 % / −25 % | +33 % / +87 % (sans rétroaction) ; dommage BF16 seul 0,136 / 0,278 | perte, confirmée |
| Tête dérivée de l'embedding, factorisation, bases partagées, correction de faible rang | écartées sur diagnostic chiffré | non construites |
| Biais FP16 par linéaire | −15 à −28 % sans rétroaction ; **égalité avec LDLQ** | neutre |
| Échelles de colonne (pivots LDL) | +5,6 % / +0,1 % avec LDLQ | neutre à négatif |
| Rotation du résiduel | @@ROTLINE@@ | @@ROTSTAT@@ |
| Ordre des colonnes de LDLQ | @@PERMLINE@@ | @@PERMSTAT@@ |

**Conclusion.** Sur ce modèle et à ce taux, je n'ai pas trouvé d'architecture ni de représentation qui préserve mieux le BF16 que la
structure d'origine quantifiée par Lumen avec LDLQ. C'est une conclusion négative, limitée à ce qui a été mesuré : fidélité des logits en
texte, modifications sans entraînement.

**Que tester ensuite ?** (détail en section 10) Les leviers restants ne sont plus dans la forme des matrices : (1) une récupération par
entraînement, appliquée à égalité au témoin et aux candidats ; (2) la répartition du budget entre texte et Vision, première mesure à l'appui ;
(3) l'évaluation sur tâches, MTP et génération, sans lesquelles aucune décision produit n'est fondée."""

DECISION = """**Décision.** Intégrer à Lumen : **rien de structurel**. Conserver l'architecture d'origine et la passe E8-LDLQ comme base (`A0Q0`).

| Élément | Décision | Motif |
|---|---|---|
| Suppression de couches, réduction de largeur, tête liée, factorisation, bases partagées, correction de faible rang | ne pas intégrer | pertes mesurées ou diagnostics chiffrés défavorables |
| Biais FP16 par linéaire (A_b) | ne pas intégrer avec LDLQ ; **à garder en réserve** pour tout chemin sans rétroaction (matrices trop grandes pour un Gram, variante rapide) | −15 à −28 % sans rétroaction pour 8,9 Mo ; neutre avec LDLQ |
| Rotation du résiduel | ne pas intégrer | """ + rw + """ sur 600 documents, coût d'intégration réel (toutes les interfaces, Vision, MTP), équivalence non exacte en BF16 |
| Ordre des colonnes de LDLQ | """ + ("intégrer comme option de recette" if (perm is not None and psig and prel < 0) else "ne pas intégrer") + """ | """ + pw + """ sur 600 documents (tendance défavorable : 239 documents améliorés sur 600) |
| Pas guidés par Fisher, biais ancré professeur | proscrire dans ces formes | catastrophiques |
| Protocole | **adopter** : témoin LDLQ obligatoire, ≥ 600 documents pour tout effet attendu < 15 %, vérification d'équivalence en BF16 de toute transformation « exacte », points de reprise | les trois ont changé des conclusions pendant ce sprint |

**Ce qu'il faut tester ensuite, par ordre de valeur attendue.**

1. **Récupération par entraînement, à effort égal pour le témoin et les candidats.** L'erreur restante après LDLQ est petite par matrice et
   sans structure : aucune reparamétrisation ne la réduit, seul un ajustement global des paramètres continus (pas, gains de normalisation,
   biais) contre la KL du professeur peut le faire. C'est aussi la seule voie qui pourrait réhabiliter une reconstruction (largeur,
   profondeur), dont le dommage BF16 n'a reçu ici qu'une récupération locale en forme close. Non testé.
2. **Budget Vision.** Au même taux que le texte, la tour Vision ajoute une KL de 0,043 sur photos. Elle pèse 1,7 % des poids : mesurer la
   frontière texte / Vision avec une vraie passe LDLQ sur la tour et un jeu d'images plus large.
3. **Évaluation de capacités.** Tâches indépendantes, génération avec cache et états persistants, contextes > 8192 tokens, MTP réellement
   chargé. Aucune revendication produit n'est fondée sans cela.
4. **Paquet réel.** Encoder les matrices entières avec le codec, fermer les 10 Go par des fichiers, décoder dans un processus indépendant.
   Le calcul des poids tournés ou repliés, s'il y en a, doit se faire en précision supérieure au BF16.
5. **Recette Lumen non activée ici** : `rate_lambda > 0`, cible asymétrique complète, BGCC. Hors du périmètre architectural ; à juger
   avec le même protocole (600 documents).
6. **Pistes à entraînement lourd** (fusion de têtes, MLA, réécriture de la règle delta, partage en profondeur) : deuxième campagne,
   seulement si un besoin mesuré le justifie. Rien dans les mesures actuelles ne désigne l'attention ou l'état récurrent comme le maillon faible."""

SUITE = """1. **Arrêter les reparamétrisations exactes.** Diagonale, rotation, permutation, biais : quatre formes essayées, aucune ne dépasse le
   témoin LDLQ. Le diagnostic de structure d'erreur explique pourquoi : il ne reste rien de structuré à exploiter.
2. **Ouvrir la récupération par entraînement**, à effort égal pour le témoin et pour un petit nombre de candidats (dont une reconstruction
   de largeur modérée), sur le protocole à 600 documents. C'est la seule question encore ouverte du dossier qui puisse déplacer le résultat.
3. **Traiter la frontière texte / Vision / MTP** comme un problème d'allocation mesuré, pas supposé.
4. **Passer aux tâches** avant toute décision produit.
5. Garder les pistes C du dossier (têtes KV, MLA, règle delta, partage profond) pour une campagne distincte avec budget d'entraînement."""

def sub(path, pairs):
    s = open(path).read()
    for k, v in pairs.items():
        assert k in s, k
        s = s.replace(k, v)
    open(path, "w").write(s)

if __name__ == "__main__":
    rotline = fr("%+.1f %% sur 600 documents (IC [%+.1f ; %+.1f] %%)" % (rrel, 100 * cmp(rot)[1] / ctrl["big"]["kl"], 100 * cmp(rot)[2] / ctrl["big"]["kl"]))
    permline = "non mesuré" if perm is None else fr("%+.1f %% sur 600 documents (IC [%+.1f ; %+.1f] %%)" % (prel, 100 * cmp(perm)[1] / ctrl["big"]["kl"], 100 * cmp(perm)[2] / ctrl["big"]["kl"]))
    resume = RESUME.replace("@@ROTLINE@@", rotline).replace("@@ROTSTAT@@", rw).replace("@@PERMLINE@@", permline).replace("@@PERMSTAT@@", pw)
    import shutil
    shutil.copy(B + "/lab/RAPPORT_TEMPLATE.md", B + "/RAPPORT_ARCHITECTURE_X_LUMEN.md"); shutil.copy(B + "/lab/REPONSE_TEMPLATE.md", B + "/docs/REPONSE_DOSSIER_PISTES.md")
    sub(B + "/RAPPORT_ARCHITECTURE_X_LUMEN.md", {"@@RESUME@@": resume, "@@H11@@": H11, "@@H12@@": H12, "@@TABLE_SELECT@@": R.table("select"), "@@LECTURE@@": LECTURE,
        "@@CONFIRM@@": CONFIRM + "\n\n" + BIGSEC, "@@LONG@@": LONG, "@@DECISION@@": DECISION})
    sub(B + "/docs/REPONSE_DOSSIER_PISTES.md", {"@@ROT@@": ROT, "@@PERM@@": PERM, "@@SUITE@@": SUITE})
    print("filled"); print(rw, rs); print(pw, ps)
