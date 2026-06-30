"""
Interface en ligne de commande : `footpredictor France Senegal --neutral`.
"""
from __future__ import annotations

import argparse

from .predictor import Predictor
from .score import afficher


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="footpredictor",
        description="Prédit l'issue et le score d'un match international (données auto-rafraîchies).")
    ap.add_argument("domicile", nargs="?", help="équipe à domicile (nom anglais)")
    ap.add_argument("exterieur", nargs="?", help="équipe à l'extérieur")
    ap.add_argument("--neutral", action="store_true", help="terrain neutre")
    ap.add_argument("--mode", default="equilibre", choices=["equilibre", "precision"])
    ap.add_argument("--update", action="store_true", help="forcer le téléchargement des données")
    ap.add_argument("--top-elo", type=int, metavar="N", help="afficher le top N Elo et quitter")
    ap.add_argument("--search", metavar="MOTIF", help="chercher une équipe et quitter")
    args = ap.parse_args(argv)

    p = Predictor(mode=args.mode, refresh=args.update, verbose=True)
    print(f"Données à jour au {p.derniere_date} · {len(p.equipes)} équipes")

    if args.search:
        print("Équipes trouvées :", ", ".join(p.chercher_equipe(args.search)) or "aucune")
        return
    if args.top_elo:
        for t, r in p.top_elo(args.top_elo):
            print(f"  {t:<18} {r:.0f}")
        return
    if not args.domicile or not args.exterieur:
        ap.error("indiquez deux équipes, ou utilisez --search / --top-elo")

    res = p.predict(args.domicile, args.exterieur, neutral=args.neutral)
    pi = res["proba_issue"]
    print(f"\nIssue (classifieur calibré, mode {args.mode}) :")
    for cls, pr in sorted(pi.items(), key=lambda x: -x[1]):
        print(f"  {cls:<10} {pr:.1%}")
    if "score_exact" in res:
        afficher(res, args.domicile, args.exterieur)


if __name__ == "__main__":
    main()
