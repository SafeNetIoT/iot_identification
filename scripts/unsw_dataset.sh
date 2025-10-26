# make sure the folder exists
mkdir -p "$(pwd)/data/unsw_unzipped"

# unzip all archives into that folder
for f in data/unsw/*.tar.gz; do
    echo "Extracting $f..."
    tar -xzf "$f" -C data/unsw_unzipped
done
