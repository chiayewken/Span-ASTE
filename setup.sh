#!/usr/bin/env bash
set -e

# Main Requirements
pip install -r requirements.txt
pip uninstall dataclasses -y # Compatibility issue with Python > 3.6

# Optional: Set up NLTK packages
if [[ -f punkt.zip ]]; then
	mkdir -p /home/admin/nltk_data/tokenizers
	cp punkt.zip /home/admin/nltk_data/tokenizers
fi
if [[ -f wordnet.zip ]]; then
	mkdir -p /home/admin/nltk_data/corpora
	cp wordnet.zip /home/admin/nltk_data/corpora
fi

# Make sample data for quick debugging
unzip -n data.zip -d aste/
cd aste/data/triplet_data
mkdir -p sample
head -n 32 14lap/train.txt >sample/train.txt
head -n 32 14lap/dev.txt >sample/dev.txt
head -n 32 14lap/test.txt >sample/test.txt
cd ../../..
