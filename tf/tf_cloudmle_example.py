# Scaling up ML using Cloud ML Engine 
# 
# In this notebook, we take a previously developed TensorFlow model to predict taxifare rides and package it up so that it can be run in Cloud MLE. For now, we'll run this on a small dataset. The model that was developed is rather simplistic, and therefore, the accuracy of the model is not great either.  However, this notebook illustrates *how* to package up a TensorFlow model to run it within Cloud ML. 

# Environment variables for project and bucket 
# 
# Note that:
#
# Your project id is the *unique* string that identifies your project (not the project name). You can find this from the GCP Console dashboard's Home page.  My dashboard reads:  <b>Project ID:</b> cloud-training-demos </li>
# Cloud training often involves saving and restoring model files. If you don't have a bucket already, I suggest that you create one from the GCP console (because it will dynamically check whether the bucket name you want is available). A common pattern is to prefix the bucket name by the project id, so that it is unique. Also, for cost reasons, you might want to use a single region bucket. </li>
# 
# Change the cell below</b> to reflect your Project ID and bucket name.

import os
PROJECT = 'cloud-training-demos' # REPLACE WITH YOUR PROJECT ID
BUCKET = 'cloud-training-demos-ml' # REPLACE WIHT YOUR BUCKET NAME
REGION = 'us-central1' # REPLACE WITH YOUR BUCKET REGION e.g. us-central1

# for bash
os.environ['PROJECT'] = PROJECT
os.environ['BUCKET'] = BUCKET
os.environ['REGION'] = REGION
os.environ['TFVERSION'] = '1.4'  # Tensorflow version

get_ipython().run_line_magic('bash', '')
gcloud config set project $PROJECT
gcloud config set compute/region $REGION


# Allow the Cloud ML Engine service account to read/write to the bucket containing training data.

get_ipython().run_line_magic('bash', '')
PROJECT_ID=$PROJECT
AUTH_TOKEN=$(gcloud auth print-access-token)
SVC_ACCOUNT=$(curl -X GET -H "Content-Type: application/json"     -H "Authorization: Bearer $AUTH_TOKEN"     https://ml.googleapis.com/v1/projects/${PROJECT_ID}:getConfig     | python -c "import json; import sys; response = json.load(sys.stdin);     print response['serviceAccount']")

echo "Authorizing the Cloud ML Service account $SVC_ACCOUNT to access files in $BUCKET"
gsutil -m defacl ch -u $SVC_ACCOUNT:R gs://$BUCKET
gsutil -m acl ch -u $SVC_ACCOUNT:R -r gs://$BUCKET  # error message (if bucket is empty) can be ignored
gsutil -m acl ch -u $SVC_ACCOUNT:W gs://$BUCKET


# Packaging up the code 
# 
# Take your code and put into a standard Python package structure.  <a href="taxifare/trainer/model.py">model.py</a> and <a href="taxifare/trainer/task.py">task.py</a> contain the Tensorflow code from earlier (explore the <a href="taxifare/trainer/">directory structure</a>).

get_ipython().system('find taxifare')

get_ipython().system('cat taxifare/trainer/model.py')


# Find absolute paths to your data 

get_ipython().run_line_magic('bash', '')
echo $PWD
rm -rf $PWD/taxi_trained
head -1 $PWD/taxi-train.csv
head -1 $PWD/taxi-valid.csv


# Running the Python module from the command-line 

get_ipython().run_line_magic('bash', '')
rm -rf taxifare.tar.gz taxi_trained
export PYTHONPATH=${PYTHONPATH}:${PWD}/taxifare
python -m trainer.task    --train_data_paths="${PWD}/taxi-train*"    --eval_data_paths=${PWD}/taxi-valid.csv     --output_dir=${PWD}/taxi_trained    --train_steps=1000 --job-dir=./tmp

get_ipython().run_line_magic('bash', '')
ls $PWD/taxi_trained/export/exporter/

get_ipython().run_line_magic('writefile', './test.json')
{"pickuplon": -73.885262,"pickuplat": 40.773008,"dropofflon": -73.987232,"dropofflat": 40.732403,"passengers": 2}

get_ipython().run_line_magic('bash', '')
model_dir=$(ls ${PWD}/taxi_trained/export/exporter)
gcloud ml-engine local predict     --model-dir=${PWD}/taxi_trained/export/exporter/${model_dir}     --json-instances=./test.json


# Running locally using gcloud 

get_ipython().run_line_magic('bash', '')
rm -rf taxifare.tar.gz taxi_trained
gcloud ml-engine local train    --module-name=trainer.task    --package-path=${PWD}/taxifare/trainer    --    --train_data_paths=${PWD}/taxi-train.csv    --eval_data_paths=${PWD}/taxi-valid.csv     --train_steps=1000    --output_dir=${PWD}/taxi_trained 

from google.datalab.ml import TensorBoard
TensorBoard().start('./taxi_trained')


for pid in TensorBoard.list()['pid']:
  TensorBoard().stop(pid)
  print 'Stopped TensorBoard with pid {}'.format(pid)

get_ipython().system('ls $PWD/taxi_trained')


# Submit training job using gcloud 
# 
# First copy the training data to the cloud.  Then, launch a training job.
 
get_ipython().run_line_magic('bash', '')
echo $BUCKET
gsutil -m rm -rf gs://${BUCKET}/taxifare/smallinput/
gsutil -m cp ${PWD}/*.csv gs://${BUCKET}/taxifare/smallinput/

get_ipython().run_cell_magic('bash', '', 'OUTDIR=gs://${BUCKET}/taxifare/smallinput/taxi_trained\nJOBNAME=lab3a_$(date -u +%y%m%d_%H%M%S)\necho $OUTDIR $REGION $JOBNAME\ngsutil -m rm -rf $OUTDIR\ngcloud ml-engine jobs submit training $JOBNAME \\\n   --region=$REGION \\\n   --module-name=trainer.task \\\n   --package-path=${PWD}/taxifare/trainer \\\n   --job-dir=$OUTDIR \\\n   --staging-bucket=gs://$BUCKET \\\n   --scale-tier=BASIC \\\n   --runtime-version=$TFVERSION \\\n   -- \\\n   --train_data_paths="gs://${BUCKET}/taxifare/smallinput/taxi-train*" \\\n   --eval_data_paths="gs://${BUCKET}/taxifare/smallinput/taxi-valid*"  \\\n   --output_dir=$OUTDIR \\\n   --train_steps=10000')


# Don't be concerned if the notebook appears stalled (with a blue progress bar) or returns with an error about being unable to refresh auth tokens. This is a long-lived Cloud job and work is going on in the cloud. 
# 
# Use the Cloud Console link to monitor the job and do NOT proceed until the job is done. 

# Deploy model 
# 
# Find out the actual name of the subdirectory where the model is stored and use it to deploy the model.

get_ipython().run_line_magic('bash', '')
gsutil ls gs://${BUCKET}/taxifare/smallinput/taxi_trained/export/exporter


get_ipython().run_line_magic('bash', '')
MODEL_NAME="taxifare"
MODEL_VERSION="v1"
MODEL_LOCATION=$(gsutil ls gs://${BUCKET}/taxifare/smallinput/taxi_trained/export/exporter | tail -1)
echo "Run these commands one-by-one (the very first time, you'll create a model and then create a version)"
#gcloud ml-engine versions delete ${MODEL_VERSION} --model ${MODEL_NAME}
#gcloud ml-engine models delete ${MODEL_NAME}
gcloud ml-engine models create ${MODEL_NAME} --regions $REGION
gcloud ml-engine versions create ${MODEL_VERSION} --model ${MODEL_NAME} --origin ${MODEL_LOCATION} --runtime-version $TFVERSION


# Prediction

get_ipython().run_line_magic('bash', '')
gcloud ml-engine predict --model=taxifare --version=v1 --json-instances=./test.json


from googleapiclient import discovery
from oauth2client.client import GoogleCredentials
import json

credentials = GoogleCredentials.get_application_default()
api = discovery.build('ml', 'v1', credentials=credentials,
            discoveryServiceUrl='https://storage.googleapis.com/cloud-ml/discovery/ml_v1_discovery.json')

request_data = {'instances':
  [
      {
        'pickuplon': -73.885262,
        'pickuplat': 40.773008,
        'dropofflon': -73.987232,
        'dropofflat': 40.732403,
        'passengers': 2,
      }
  ]
}

parent = 'projects/%s/models/%s/versions/%s' % (PROJECT, 'taxifare', 'v1')
response = api.projects().predict(body=request_data, name=parent).execute()
print "response={0}".format(response)

# Train on larger dataset

# This took 60 minutes and uses as input 1-million rows.  The model is exactly the same as above. The only changes are to the input (to use the larger dataset) and to the Cloud MLE tier (to use STANDARD_1 instead of BASIC -- STANDARD_1 is approximately 10x more powerful than BASIC).  At the end of the training the loss was 32, but the RMSE (calculated on the validation dataset) was stubbornly at 9.03. So, simply adding more data doesn't help.

get_ipython().run_cell_magic('bash', '', '\nXXXXX  this takes 60 minutes. if you are sure you want to run it, then remove this line.\n\nOUTDIR=gs://${BUCKET}/taxifare/ch3/taxi_trained\nJOBNAME=lab3a_$(date -u +%y%m%d_%H%M%S)\nCRS_BUCKET=cloud-training-demos # use the already exported data\necho $OUTDIR $REGION $JOBNAME\ngsutil -m rm -rf $OUTDIR\ngcloud ml-engine jobs submit training $JOBNAME \\\n   --region=$REGION \\\n   --module-name=trainer.task \\\n   --package-path=${PWD}/taxifare/trainer \\\n   --job-dir=$OUTDIR \\\n   --staging-bucket=gs://$BUCKET \\\n   --scale-tier=STANDARD_1 \\\n   --runtime-version=$TFVERSION \\\n   -- \\\n   --train_data_paths="gs://${CRS_BUCKET}/taxifare/ch3/train.csv" \\\n   --eval_data_paths="gs://${CRS_BUCKET}/taxifare/ch3/valid.csv"  \\\n   --output_dir=$OUTDIR \\\n   --train_steps=100000')
