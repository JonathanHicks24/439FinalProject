# Cloning and changing directory:
git clone https://github.com/sherjilozair/char-rnn-tensorflow.git
cd char-rnn-tensorflow

# Installing Requirements
pip install -r requirements.txt

# Char-rnn library training command (assuming the below command structure)
# char-rnn-tensorflow/
#     data/
#         passwords/
#             input.txt
python train.py --data_dir ./data/passwords --save_dir ./save

# Running the graphing Software (access at http://localhost:6006)
tensorboard --logdir=path/to/log-directory