#!/bin/bash

CATEGORIES=(
    # "voronoi_simulations"
    "cartesian_simulations"
    "cylindrical_simulations"
    "gmm_simulations"
    # "physics_simulations"
    "spectral_simulations"
    "adaptive_simulations"
    "quantile_simulations"
    "octree_simulations"
    "multizone_simulations"
    "single_simulations"
)

LOG_DIR="logs_postprocess"
mkdir -p "$LOG_DIR"

pids=()
N=3  # max parallèle
for cat in "${CATEGORIES[@]}"; do
    log="$LOG_DIR/${cat}.log"
    echo "🚀 $cat → $log"
    python postprocess.py --category "$cat" > "$log" 2>&1 &
    pids+=($!)
    # Attendre si on a atteint N processus actifs
    while [ "$(jobs -rp | wc -l)" -ge "$N" ]; do
        sleep 2
    done
done

echo ""
echo "⏳ Attente de ${#pids[@]} processus..."

failed=0
for i in "${!pids[@]}"; do
    pid=${pids[$i]}
    cat=${CATEGORIES[$i]}
    if wait "$pid"; then
        echo "✅ $cat"
    else
        echo "❌ $cat (code $?)"
        ((failed++))
    fi
done

echo ""
if [ "$failed" -eq 0 ]; then
    echo "🎉 Toutes les catégories terminées avec succès."
else
    echo "⚠️  $failed catégorie(s) en erreur — voir $LOG_DIR/"
fi