import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from read_data import read_data, read_data_with_labels
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import confusion_matrix, classification_report


if __name__ == '__main__':

    print('Loading data...')
    data, labels = read_data_with_labels('data_mixed_labeled_preprocessed.tsv')
    labels = [0 if x == 'de' else x for x in labels]
    labels = [1 if x == 'bar' else x for x in labels]
    labels = [2 if x == 'nds' else x for x in labels]
    print('Done!\n')

    # split into training data and test data
    data_train = data[:30000]
    labels_train = labels[:30000]
    data_test = data[30000:31000]
    labels_test = labels[30000:31000]

    # vectorize the data
    print('Vectorize data...')
    tf_idf = TfidfVectorizer()
    X_train_tf = tf_idf.fit_transform(data_train)
    X_test_tf = tf_idf.transform(data_test)
    print('Done!\n')

    # dimensionality reduction using PCA
    print('Dimensionality reduction...')
    pca = TruncatedSVD(n_components=128)
    X_train_tf_pca = pca.fit_transform(X_train_tf, None)
    print('Done!\n')

    # Classifier training RandomForest
    print('Classifier training...')
    classifier = RandomForestClassifier(verbose=1)
    classifier.fit(X_train_tf, np.asarray(labels_train))
    print('Done!\n')

    # predict test data
    y_pred = classifier.predict(X_test_tf)
    print(classification_report(np.asarray(labels_test), y_pred))

    # Plot confusion matrix in a beautiful manner
    ax = plt.subplot()
    index = ['Hochdeutsch', 'Bairisch', 'Plattdeutsch']

    cnf_matrix = confusion_matrix(labels_test, y_pred)
    df = pd.DataFrame(cnf_matrix, index=index, columns=index)
    sns.heatmap(df, annot=True, cmap='Blues', ax=ax) # fmt='g'

    # labels, title and ticks
    ax.set_xlabel('Predicted', fontsize=15)
    ax.set_ylabel('True', fontsize=15)
    plt.title('Refined Confusion Matrix - TFIDF Random Forest', fontsize=18)

    plt.savefig('ConMat-TFIDF-RF.png')


