#!/usr/bin/env bash
set -e

# Main Requirements
pip3 install -r requirements.txt
#python3 -m spacy download en_core_web_sm
pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz
pip3 uninstall dataclasses -y  # Compatibility issue with Python > 3.6

# ASTE Data
FOLDER=Position-Aware-Tagging-for-ASTE
if [[ ! -d aste/data ]]; then
  git clone https://github.com.cnpmjs.org/xuuuluuu/Position-Aware-Tagging-for-ASTE.git
  cd $FOLDER
  git checkout 32572ce75d243c5ea36f1133ebb5c3247062b60c
  cd -
  cp -a $FOLDER/data aste
  # Make sample data for quick debugging
  cd aste/data/triplet_data
  mkdir sample
  head -n 32 14lap/train.txt > sample/train.txt
  head -n 32 14lap/dev.txt > sample/dev.txt
  head -n 32 14lap/test.txt > sample/test.txt
  cd -
  rm -rf $FOLDER
fi
