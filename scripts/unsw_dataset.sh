#!/bin/bash

# make sure the folder exists
mkdir -p "$(pwd)/data/unsw_unzipped"

# unzip all archives into that folder
for f in data/unsw/*.tar.gz; do
    echo "Extracting $f..."
    tar -xzf "$f" -C data/unsw_unzipped
done

# filter pcaps
for f in data/unsw_unzipped/*.pcap; do
    echo "Filtering $f ..."
    tshark -r "$f" \
        -Y "(wlan.addr == 70:ee:50:03:b8:ac || eth.addr == 70:ee:50:03:b8:ac)" \
        -w "${f%.pcap}_filtered.pcap"
    mv "${f%.pcap}_filtered.pcap" "$f"
done
