# run pipeline
python3 /opt/code/pipeline.py -p $INPUT -o $OUTPUT

# copy nested output files to working directory
find . -type f -name "*.xyz" -exec cp {} $WORKDIR \;
find . -type f -name "*.ply" -exec cp {} $WORKDIR \;
find . -type f -name "*.stl" -exec cp {} $WORKDIR \;
<<<<<<< HEAD
find . -type f -name "*.obj" -exec cp {} $WORKDIR \;
=======
find . -type f -name "*.xyz" -exec cp {} $WORKDIR \;
find . -type f -name "*.ply" -exec cp {} $WORKDIR \;
>>>>>>> ea3ba77405bf516f9f3c54c27fcbc0099d420693
find . -type f -name "*.xlsx" -exec cp {} $WORKDIR \;
