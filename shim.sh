# run pipeline
python3 /opt/3D_model_traits_demo/pipeline.py -p $(dirname $INPUT)/ -m $(basename $INPUT)

# copy nested output files to working directory
find . -type f -name "*.png" -exec cp {} $WORKDIR \;
find . -type f -name "*.jpg" -exec cp {} $WORKDIR \;
find . -type f -name "*.csv" -exec cp {} $WORKDIR \;
find . -type f -name "*.txt" -exec cp {} $WORKDIR \;
find . -type f -name "*.stl" -exec cp {} $WORKDIR \;
find . -type f -name "*.xlsx" -exec cp {} $WORKDIR \;