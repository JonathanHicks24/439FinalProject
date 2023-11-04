# Cloning and changing directory:
git clone https://github.com/sherjilozair/char-rnn-tensorflow.git
cd char-rnn-tensorflow

# Making the Required Directories
cd data
mkdir passwords
cd ..

# Char-rnn library training command (assuming the below command structure)
# char-rnn-tensorflow/
#     data/
#         passwords/
#             input.txt
python train.py --data_dir ./data/passwords --save_dir ./save

# Running the graphing Software (access at http://localhost:6006)
tensorboard --logdir=./logs/