mkdir data/network_inputs
mkdir log
mkdir log/check_point
mkdir log/plot
mkdir log/pth
mkdir log/test

echo 'Downloading DeepFix dataset...'
wget https://www.cse.iitk.ac.in/users/karkare/prutor/prutor-deepfix-09-12-2017.zip -P data/
cd data
unzip prutor-deepfix-09-12-2017.zip
mv prutor-deepfix-09-12-2017/* iitk-dataset/
rm -rf prutor-deepfix-09-12-2017 prutor-deepfix-09-12-2017.zip
cd iitk-dataset/
gunzip prutor-deepfix-09-12-2017.db.gz
mv prutor-deepfix-09-12-2017.db dataset.db
cd ../..

echo 'Preprocessing DeepFix dataset...'
export PYTHONPATH=.
python data_processing/preprocess.py



echo 'Data generation...'
bash data_processing/data_generator.sh
python data/data_generator/data_generator_editDist_fixed.py
bash data/data_generator/ids_typo.sh

echo 'Done...'
