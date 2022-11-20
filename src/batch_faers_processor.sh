# Designed to be used with bsub to batch process FAERS event files
# after all files have been downloaded.
#
# bash src/batch_faers_processor.sh ENDPOINT

for subpath in ./data/faers/$1/event/*
do
  echo bsub python3 src/faers_processor.py --endpoint $1 --subpath $subpath
done
