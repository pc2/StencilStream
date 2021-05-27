#!/usr/bin/env bash

if [ $# -ne 2 ]
then
    echo "Usage: $0 <lhs> <rhs>" 1>&2
    exit 1
fi

paste $1 $2 | tr " \t" "," | cut -d, -f2,4 | awk -F "," \
    'BEGIN {n_correct = 0; n_lines = 0; diff_sum = 0; diff_max = -1; }

    {
        n_lines++;
        if ($1 == $2) {
            n_correct++;
        }
        diff = $1 - $2;
        if (diff < 0) {
            diff *= -1;
        }
        if (diff_max < diff) {
            diff_max = diff;
        }
        diff_sum += diff;
    }
    END {print "" n_correct/n_lines ", " diff_sum/n_lines ", " diff_max }'
