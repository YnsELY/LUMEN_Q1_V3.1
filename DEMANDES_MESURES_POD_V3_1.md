# Demandes A → B après la contre-vérification

**Statut : proposées, non acceptées ; transmission manuelle.** Aucun temps GPU supplémentaire n’est autorisé et aucune durée n’est présumée. Le responsable B peut réduire ou refuser ces demandes pour préserver son pilote structurel.

## REQ-31-01 — Vérifier la reprise de l’outillage

Exécuter `tools/check_release.py` sur la livraison, puis identifier L0/S0 historiques, leur commit, les dtype et les conventions d’erreur. Ne pas remplacer L0 par le prototype CPU sans le nommer. Les 107 tests ne sont pas une validation du checkpoint.

**Résultat attendu :** manifeste du runtime et état des composants réellement disponibles. Coût : chronométrage local requis, pas de promesse de minutes H100.

## REQ-31-02 — Pilote asymétrique isolé

Sur une projection ou un bloc contextualisé choisi par B, conserver mêmes entrées et cible BF16, puis comparer cible symétrique et cible asymétrique régularisée. Utiliser un quantifieur correctement exécuté, avec les mêmes possibilités de réglage pour le contrôle.

Capture pleine largeur, normes et contributions complètes ; partitions de documents disjointes ; identité du préfixe quantifié. Le candidat ne doit pas consommer plus d’octets que le contrôle retenu, ou bien rapporter une frontière taille/qualité avec une comparaison explicite. Ne pas utiliser le gain synthétique de ce dossier comme cible à obtenir.

**Mesures :** sortie après export/décodage, octets réels, temps de collecte/résolution/recherche/export, puis logits sur greffe si disponibles. Les états/caches restent fixés afin d’isoler la perturbation. Les gains de bloc ne sont pas des gains de modèle entier.

## REQ-31-03 — Confirmation et interfaces

Seulement si le pilote est correct : confirmer sur un autre document/bloc ou autre famille, examiner les régressions concentrées, et rapprocher la métrique groupée Top-K de l’oracle. Pour des interfaces MTP/Vision non exécutées, inscrire « non testé ». Une courte séquence ne valide pas le contexte maximal.

BGCC et polissage ne sont pas ajoutés à ces demandes par défaut. Leur évaluation exige une décision distincte, des hypothèses et un budget. La reconstruction architecturale reste la mission principale du chantier B ; les gains A et B ne s’additionnent pas par convention.
