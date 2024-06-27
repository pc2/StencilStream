#!/usr/bin/env bash
set -e

ml tools release-cli

# Cloning the wiki and adding the new performance metrics.
git clone git@git.uni-paderborn.de:pc2/sycl-stencil.wiki.git wiki
cd wiki
julia --project -e "using Pkg; Pkg.instantiate()"
for metrics_file in ../examples/*/*.json
do
    ./Performance-tracking/add_data.jl $TAG_NAME $metrics_file
done
./Performance-tracking/render-site.jl > Performance-tracking.md
git stage Performance-tracking.md Performance-tracking/data.csv
git commit -m "Adding performance data of $TAG_NAME"
git push
cd ..

# Preparing the assets links
ASSETS_LINK_FILE=$(mktemp)

for package in $(ls build/examples/*/*.tar.gz)
do
    FILE_NAME=$(basename $package)
    PACKAGE_NAME=$(basename -s .tar.gz $package)
    echo "{\"name\":\"${FILE_NAME}\", \"url\":\"${PACKAGE_REGISTRY_URL}/${PACKAGE_NAME}/${TAG_NAME}/${FILE_NAME}\"}" >> $ASSETS_LINK_FILE
done

# Creating the release
release-cli create \
    --name "${RELEASE_NAME}" --tag-name "${TAG_NAME}" --ref "${CI_COMMIT_SHA}" \
    --assets-link="[$(cat $ASSETS_LINK_FILE | tr "\n" ", " | sed 's/,$/\n/')]"

# Cleanup
rm $ASSETS_LINK_FILE
