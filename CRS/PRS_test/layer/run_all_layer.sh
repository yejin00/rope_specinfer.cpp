#!/bin/bash

SCRIPT="../../../scripts/compare_absmax.py"
BASE="rope_longbench.bin"
TARGET="rope_longbench.bin"
SCALES="scales_sqrt_cap_k4.bin"
OUTDIR="."

for layer in $(seq 0 31); do
    for head in $(seq 0 7); do
        outfile="${OUTDIR}/comparison_l${layer}_h${head}.png"
        echo "Generating: ${outfile}"
        python3 ${SCRIPT} ../${BASE} ../${TARGET} ../${SCALES} \
            --layer ${layer} --head ${head} --output ${outfile}
    done
done

echo "Done! Generated $(ls -1 ${OUTDIR}/comparison_l*.png 2>/dev/null | wc -l) images."

