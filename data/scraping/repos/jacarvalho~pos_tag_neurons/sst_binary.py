"""
Copyright 2018 University of Freiburg
Joao Carvalho <carvalhj@cs.uni-freiburg.de>

Replicates the results from the sentiment neuron paper from openAI
Script partly adapted from
https://github.com/openai/generating-reviews-discovering-sentiment
"""
import sys
import os
from utils_sst import sst_binary, train_with_reg_cv
from byte_LSTM.model import Model
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Get data.
trX, vaX, teX, trY, vaY, teY = sst_binary(data_dir='data_sst/')

"""
Build trained model and compute the cell state of each review.
"""
# save_dir is the location of the previously trained language model
save_dir = '../byte_LSTM_trained_models/amazon/'

with open(os.path.join(save_dir, 'config.pkl'), 'rb') as f:
    model_saved_args = pickle.load(f)

tf.reset_default_graph()
model = Model(model_saved_args, sampling=True)

with tf.Session() as sess:
    # Restore the saved session.
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state(save_dir)
    if not ckpt:
        print('Unable to find checkpoint file.')
        sys.exit()
    model_path = os.path.join(
        save_dir, ckpt.model_checkpoint_path.split('/')[-1])
    saver.restore(sess, model_path)

    trXt = model.transform(sess, trX)
    vaXt = model.transform(sess, vaX)
    teXt = model.transform(sess, teX)

"""
Train a Logistic Regression Classifier on top of the representation.
"""
lr_model, full_rep_acc, c, nnotzero, coef = train_with_reg_cv(trXt, trY, vaXt,
                                                              vaY, teXt, teY)
pickle.dump(lr_model, open('log_reg_model.sav', 'wb'))

"""
Compute the accuracy of the trained logistic classifier.
Find missclassified reviews in the test dataset.
"""
print('\nLogistic Regression results.')
print('{:.3f} test accuracy'.format(full_rep_acc))
print('{:.3f} regularization coef'.format(c))
print('{:.3f} features used'.format(nnotzero))

coef = np.squeeze(coef)
max_coef = np.argmax(abs(coef))
print('Largest feature: {}'.format(max_coef))

with open('missclassified_test_reviews.txt', 'w') as f:
    f.write('Predicted, Label, Review\n')
    predictions = lr_model.predict(teXt)
    for prediction, y, X in zip(predictions, teY, teX):
        if prediction != y:
            result = "{}, {}, {} \n".format(prediction, y, X)
            f.write(result)

"""
Compute the accuracy of the sentiment neuron. Find missclassified reviews.
(So far, a negative value of the sentiment neuron is associated with a positive
sentiment. This must be inspected visually by reading the output of the
reviews)
The classifier assings 0 to negative, 1 to positive.
"""
print('\nSentiment Neuron results.')
sent_neuron_teXt = teXt[:, max_coef]
index_pos_reviews = sent_neuron_teXt <= 0
index_neg_reviews = sent_neuron_teXt > 0
sent_neuron_teXt[index_pos_reviews] = 1
sent_neuron_teXt[index_neg_reviews] = 0

print('{:.3f} test accuracy'.format(sum(sent_neuron_teXt == teY) / len(teY)))
with open('missclassified_test_reviews_sent_neuron.txt', 'w') as f:
    f.write('Predicted, Label, Review\n')
    for prediction, y, X in zip(sent_neuron_teXt, teY, teX):
        if prediction != y:
            result = "{}, {}, {} \n".format(int(prediction), y, X)
            f.write(result)

"""
Plot relevant figures.
"""
plt.figure(1)
plt.plot([0, len(coef)-1], [0, 0], color='blue')
plt.bar(np.arange(len(coef)), coef, width=5, color='blue')
plt.xlabel('Weight dimension (neuron)')
plt.ylabel('Weight value')
plt.title('Logistic regression weights')
plt.savefig('classifier_weights.png')

# Visualize sentiment unit
plt.figure(3)
sentiment_unit = trXt[:, max_coef]
sns.distplot(sentiment_unit[trY == 0], hist=True, kde=True,
             kde_kws={'shade': True, 'linewidth': 1}, color='red', label='neg')
sns.distplot(sentiment_unit[trY == 1], hist=True, kde=True,
             kde_kws={'shade': True, 'linewidth': 1}, color='blue',
             label='pos')
plt.xlabel('Sentiment neuron cell state')
plt.title('Neuron {}'.format(max_coef))
plt.legend()
plt.savefig('sentiment_unit.png')
