#!/usr/bin/env bash
set -e

module reset
module load lang Julia

if [ $# -ne 3 ]; then
    echo "Usage: $0 <release name> <tag name> <commit sha>" 1>&2
    exit 1
fi

# Cloning the wiki and adding the new performance metrics.
git clone git@git.uni-paderborn.de:pc2/sycl-stencil.wiki.git wiki
cd wiki
for metrics_file in ../examples/*/*.json
do
    ./Performance-tracking/add_data.jl $TAG_NAME $metrics_file
done
./Performance-tracking/render-site.jl > Performance-tracking.md
git stage Performance-tracking.md Performance-tracking/data.csv
git commit -m "Adding performance data of $TAG_NAME"
git push
cd ..

# Downloading the release cli tool
curl -L https://gitlab.com/gitlab-org/release-cli/-/releases/v0.15.0/downloads/bin/release-cli-linux-amd64 > release-cli
echo "d59169bab5dfe4693af4f181b08bf11ef1c96a4da30eff0f04abb236c54e62e9  release-cli" | sha256sum -c || exit 1
chmod +x release-cli

# Preparing the assets links
ASSETS_LINK_FILE=$(mktemp)

for package in $(ls build/examples/*/*.tar.gz)
do
    FILE_NAME=$(basename $package)
    PACKAGE_NAME=$(basename -s .tar.gz $package)
    echo "{\"name\":\"${FILE_NAME}\", \"url\":\"${PACKAGE_REGISTRY_URL}/${PACKAGE_NAME}/${TAG_NAME}/${FILE_NAME}\"}" >> $ASSETS_LINK_FILE
done

# Creating the release
./release-cli create \
    --name "${RELEASE_NAME}" --tag-name "${TAG_NAME}" --ref "${CI_COMMIT_SHA}" \
    --assets-link="[$(cat $ASSETS_LINK_FILE | tr "\n" ", " | sed 's/,$/\n/')]"

# Cleanup
rm $ASSETS_LINK_FILE
