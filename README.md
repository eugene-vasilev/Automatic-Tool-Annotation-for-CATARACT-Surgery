# Automatic Tool Annotation for CATARACT Surgery: Masters Work

All information about tasks, goals, data and etc. see on competition page https://cataracts.grand-challenge.org/

## Local deployment

	$ git clone https://github.com/evgeniivas/Masters-Work.git
	$ cd Masters-Work
	$ pip3 install -r requirements.txt

### Data

Download and place CATARACTS competition data to the `learning/data/` folder, as a result in `data` folder you will have 3 subfolders: `train`, `test`, `train_labels`.

### Launch scripts

All scripts for making classification models and data processing can be run from `/learning` directory (see params to each script with `--help` flag) :

1. Run `/preparation/extractor.py` script to prepare data for learning, set the params;
2. To learn models run `learning.py` script, set the params;
	* To run training on cluster (distributed systems): set `--distributed` parameter;
3. Make predictions to train\test datasets: run `predict.py` script, set the params;
4. Make baseline model (ExtraTreesClassifier trained using HOG descriptors) and predict on test dataset:<br/> 
run `baseline.py` script, set the params;

### Convert to ONNX format:

From `/learning` folder run transfer bash script: <br/>

	$ ./model/transfer.sh

As a result in each model folder appear file: `model.onnx`

