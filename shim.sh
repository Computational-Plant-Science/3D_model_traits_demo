# run pipeline
# TODO: maybe just 1 input parameter? (full path to model file instead of specifying dir and name separately)
python3 /opt/code/pipeline.py -p $(dirname $INPUT)/ -m $(basename $INPUT)

# copy nested output files to base working directory
find . -type f -name "*.png" -exec cp {} $WORKDIR \;
find . -type f -name "*.jpg" -exec cp {} $WORKDIR \;
find . -type f -name "*.csv" -exec cp {} $WORKDIR \;
find . -type f -name "*.txt" -exec cp {} $WORKDIR \;
find . -type f -name "*.stl" -exec cp {} $WORKDIR \;
find . -type f -name "*.xlsx" -exec cp {} $WORKDIR \;