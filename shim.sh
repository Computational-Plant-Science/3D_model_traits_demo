# run pipeline
python3 /opt/code/pipeline.py -i $INPUT -o $OUTPUT 

# copy nested output files to working directory
find . -type f -name "*.xyz" -exec cp {} $WORKDIR \;
find . -type f -name "*.ply" -exec cp {} $WORKDIR \;
find . -type f -name "*.png" -exec cp {} $WORKDIR \;
find . -type f -name "*.xlsx" -exec cp {} $WORKDIR \;

