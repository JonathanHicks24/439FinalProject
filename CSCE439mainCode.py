import tensorflow as tf
from char_rnn_tensorflow import model, sample
import numpy as np

# Load trained model
chars, vocab, saved_model = sample.load_model("save")
net = model.CharRNN(len(chars), 128, 3, 0.5, gpu=False)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, saved_model)

# Algorithm 1: Compute the probability of a password
def compute_password_probability(password):
    context = "\n"
    running_prob = 1.0
    for char in password:
        x = np.array([[vocab[c] for c in context]])
        feed = {net.input_data: x, net.initial_state: None}
        probs = sess.run(net.softmax, feed_dict=feed)
        running_prob *= probs[0, -1, vocab[char]]
        context += char
    return running_prob

# Algorithm 2: Sample passwords from learned distribution
def sample_passwords(n):
    passwords = []
    probabilities = []
    for _ in range(n):
        password = ""
        context = "\n"
        running_prob = 1.0
        while True:
            x = np.array([[vocab[c] for c in context]])
            feed = {net.input_data: x, net.initial_state: None}
            probs = sess.run(net.softmax, feed_dict=feed)
            next_char_idx = sample.sample_from_distribution(probs[0, -1])
            next_char = chars[next_char_idx]
            running_prob *= probs[0, -1, next_char_idx]
            if next_char == "\n":
                break
            password += next_char
            context += next_char
        passwords.append(password)
        probabilities.append(running_prob)
    return passwords, probabilities

# Example usage
print("Probability:", compute_password_probability("password123"))
print("Sampled Passwords:", sample_passwords(5))
