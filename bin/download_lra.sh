wget -v https://storage.googleapis.com/long-range-arena/lra_release.gz -P $SCRATCH
mkdir $SCRATCH/raw_datasets
tar -xf $SCRATCH/lra_release.gz -C $SCRATCH/raw_datasets
