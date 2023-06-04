export CONFIG_FILE="exp01"
pip install -r requirements.txt
pip install -r requirements.txt
cd helper
cd ..
cd src/data
printf "Preparing dtataset"
python dataset.py
cd ..
cd predictions
printf "Runing"
python run_predictions.py
