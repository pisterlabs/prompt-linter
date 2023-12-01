import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import warnings

warnings.filterwarnings("ignore")

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# 8.1 Аналіз тексту. Препроцесінг тексту

# 1. Завантаження текстових даних та сформування корпусу текстів
# Припустимо, у вас є датафрейм pandas з колонкою "text" та колонкою "class"
data = pd.read_csv('C:/Users/yonaaani/.spyder-py3/project/emotion.csv')

# 2. Препроцесінг тексту
# Препроцесінг тексту
def preprocess_text(text):
    # Нормалізація - переведення в нижній регістр
    text = text.lower()

    # Токенізація
    words = word_tokenize(text)

    # Видалення стоп-слів
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Стеммінг
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    # Лематизація
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Видалення пунктуації та цифр
    words = [word for word in words if word.isalpha()]

    # Видалення не часто вживаних слів або несуттєвих для подальших операцій
    # (це може бути додатковий крок, залежно від потреб)
    frequent_words = ["p"]  # Замініть це на реальний список часто вживаних слів
    words = [word for word in words if word not in frequent_words]

    return words

# Застосування препроцесінгу до колонки 'text' у DataFrame
data['preprocessed_text'] = data['Comment'].apply(preprocess_text)

# Виведення перших кількох рядків DataFrame для перевірки
print("\n")
print("Препроцесінг тексту: ")
print(data.head())

# 3. Візуалізація найбільш та найменш часто вживаних слів в межах кожного класу
# Створення хмари слів
wordcloud = WordCloud().generate(' '.join(data['Comment']))

# Візуалізація хмари слів
plt.imshow(wordcloud, interpolation="bilinear")
plt.title('Word Cloud')
plt.axis("off")
plt.show()

# Створення хмари слів для іншої колонки
wordcloud = WordCloud().generate(' '.join(data['Emotion']))

# Візуалізація хмари слів
plt.imshow(wordcloud, interpolation="bilinear")
plt.title('Word Cloud')
plt.axis("off")
plt.show()

# 4.Створити зважену матрицю термінів та проаналізувати її. Сформувати мішки слів

# Використання TfidfVectorizer для створення зваженої матриці термінів
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # Обирайте кількість фічей за необхідністю
tfidf_matrix = tfidf_vectorizer.fit_transform(data['preprocessed_text'].astype('str'))

# Створення DataFrame для зваженої матриці термінів
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Додавання до оригінального DataFrame
data = pd.concat([data, tfidf_df], axis=1)

# Збереження зваженої матриці термінів у CSV файл (за необхідності)
tfidf_df.to_csv('tfidf_matrix.csv', index=False)

# Виведення перших кількох рядків DataFrame для перевірки
print("\n")
print("Зважена матриця термінів: ")
print(data.head())

# Формування мішка слів (bag of words)
bow_vectorizer = TfidfVectorizer(max_features=1000, use_idf=False, binary=True)
bow_matrix = bow_vectorizer.fit_transform(data['preprocessed_text'].astype('str'))

# Створення DataFrame для мішка слів
bow_df = pd.DataFrame(bow_matrix.toarray(), columns=bow_vectorizer.get_feature_names_out())

# Додавання до оригінального DataFrame
data = pd.concat([data, bow_df], axis=1)

# Збереження мішка слів у CSV файл (за необхідності)
print("\n")
print("Мішок слів (bag of words): ")
bow_df.to_csv('bag_of_words.csv', index=False)

# Виведення перших кількох рядків DataFrame для перевірки
print(data.head())

# 5. Побудувати кластеризацію на основі TF-IDF для встановлення подібності текстів

# Побудова кластерів за допомогою K-Means
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(tfidf_matrix)

# Додавання міток кластерів до DataFrame
data['cluster_label'] = kmeans.labels_

# Візуалізація кластерів у двох вимірах (використовуючи PCA)
pca = PCA(n_components=2)
tfidf_matrix_2d = pca.fit_transform(tfidf_matrix.toarray())
data['PCA1'] = tfidf_matrix_2d[:, 0]
data['PCA2'] = tfidf_matrix_2d[:, 1]

# Розфарбовування кластерів на графіку
plt.figure(figsize=(10, 6))
scatter = plt.scatter(data['PCA1'], data['PCA2'], c=data['cluster_label'], cmap='viridis', alpha=0.5)
plt.title('Кластеризація за TF-IDF з використанням K-Means')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend(*scatter.legend_elements(), title='Кластери')
plt.show()

# Виведення перших кількох рядків DataFrame з мітками кластерів
print("\n")
print("Кластеризація за TF-IDF з використанням K-Means: ")
print(data[['Comment', 'cluster_label']].head())

# 6. Повторити попередні етапи для суміші n-грам та порівняти отримані результати

# Використання TfidfVectorizer для створення зваженої матриці термінів з сумішшю n-грам
tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))  # Враховувати біграми
tfidf_matrix = tfidf_vectorizer.fit_transform(data['preprocessed_text'].astype('str'))

# Побудова кластерів за допомогою K-Means
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(tfidf_matrix)

# Додавання міток кластерів до DataFrame
data['cluster_label'] = kmeans.labels_

# Візуалізація кластерів у двох вимірах (використовуючи PCA)
pca = PCA(n_components=2)
tfidf_matrix_2d = pca.fit_transform(tfidf_matrix.toarray())
data['PCA1'] = tfidf_matrix_2d[:, 0]
data['PCA2'] = tfidf_matrix_2d[:, 1]

# Розфарбовування кластерів на графіку
plt.figure(figsize=(10, 6))
scatter = plt.scatter(data['PCA1'], data['PCA2'], c=data['cluster_label'], cmap='viridis', alpha=0.5)
plt.title('Кластеризація за TF-IDF (з сумішшю n-грам) з використанням K-Means')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend(*scatter.legend_elements(), title='Кластери')
plt.show()

# Виведення перших кількох рядків DataFrame з мітками кластерів
print("\n")
print("Кластеризація за TF-IDF (з сумішшю n-грам) з використанням K-Means: ")
print(data[['Comment', 'cluster_label']].head())

print("\n")
print("Порівняння: ")
print("З використанням суміші n-грам у кластеризації TF-IDF спостерігається поліпшення виявлення схожості за контекстом текстів. Комбінації слів та біграм дозволяють краще розрізняти семантично схожі тексти.")

print("\n")
print("==============================================================================================")
print("\n")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

# 8.2 Аналіз тексту. Класифікація тексту

# 1. На основі даних з попередньої роботи та 2-х відомих вам алгоритмів побудувати класифікатор текстів.

# Розділення даних на тренувальний та тестовий набори
# Конвертація колонки "preprocessed_text" у список рядків
texts = data['preprocessed_text'].astype(str).tolist()

# Розділення даних на тренувальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(texts, data['Emotion'], test_size=0.2, random_state=42)

# Векторизація текстів за допомогою TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Класифікація за допомогою SVM
svm_classifier = SVC(kernel='linear', C=1.0)
svm_classifier.fit(X_train_tfidf, y_train)
svm_predictions = svm_classifier.predict(X_test_tfidf)

# Оцінка результатів класифікації за допомогою SVM
print("\nТочність (SVM): ", accuracy_score(y_test, svm_predictions))
print("Звіт про класифікацію (SVM): \n", classification_report(y_test, svm_predictions))

# Класифікація за допомогою Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_tfidf, y_train)
rf_predictions = rf_classifier.predict(X_test_tfidf)

# Оцінка результатів класифікації за допомогою Random Forest
print("\nТочність (Random Forest): ", accuracy_score(y_test, rf_predictions))
print("Звіт про класифікацію(SVM): (Random Forest): \n", classification_report(y_test, rf_predictions))

# 2. Провести підбір оптимальних параметрів (параметрів самого алгоритму та самого тексту, як вхідних даних)

# Параметри для SVM
svm_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly']
}

# Створення об'єкту GridSearchCV для SVM
svm_grid_search = GridSearchCV(SVC(), svm_param_grid, cv=5, scoring='accuracy')
svm_grid_search.fit(X_train_tfidf, y_train)

# Виведення кращих параметрів для SVM
print("\nНайкращі параметри для SVM: ", svm_grid_search.best_params_)

# Параметри для Random Forest
rf_param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Створення об'єкту GridSearchCV для Random Forest
rf_grid_search = GridSearchCV(RandomForestClassifier(), rf_param_grid, cv=5, scoring='accuracy')
rf_grid_search.fit(X_train_tfidf, y_train)

# Виведення кращих параметрів для Random Forest
print("\nНайкращі параметри для Random Forest: ", rf_grid_search.best_params_)

# 3. Оцінити отримані моделі з застосуванням метрик (precision, recall, f1, support) та порівняти отримані результати.

# Оцінка результатів класифікації за допомогою SVM з кращими параметрами
best_svm_classifier = svm_grid_search.best_estimator_
svm_predictions = best_svm_classifier.predict(X_test_tfidf)
print("\nТочність (SVM з кращими параметрами): ", accuracy_score(y_test, svm_predictions))
print("Звіт про класифікацію (SVM з кращими параметрами): \n", classification_report(y_test, svm_predictions))

# Оцінка результатів класифікації за допомогою Random Forest з кращими параметрами
best_rf_classifier = rf_grid_search.best_estimator_
rf_predictions = best_rf_classifier.predict(X_test_tfidf)
print("\nТочність (Random Forest з кращими параметрами): ", accuracy_score(y_test, rf_predictions))
print("Звіт про класифікацію (Random Forest з кращими параметрами): \n", classification_report(y_test, rf_predictions))

print("\n")
print("==============================================================================================")
print("\n")

# 8.3 Аналіз тексту. Sentiment analysis та topic modeling

import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk import pos_tag
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# 1. Завантажити необхідні дані

# Завантаження текстових даних та сформування корпусу текстів
data = pd.read_csv('C:/Users/yonaaani/.spyder-py3/project/emotion.csv')

# 2. Підготувати дані на основі підходів з роботи 10

# Препроцесінг тексту (використаємо функцію preprocess_text з попереднього коду)
data['preprocessed_text'] = data['Comment'].apply(preprocess_text)

# 3. Оцінити тексти за показниками polarity та subjectivity в межах кожної категорії

# Функція для аналізу sentiment та subjectivity
def analyze_sentiment(text):
    blob = TextBlob(' '.join(text))  # Об'єднання токенізованих слів в текст
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    return polarity, subjectivity

# Застосуємо функцію до кожної категорії
data['polarity'], data['subjectivity'] = zip(*data['preprocessed_text'].apply(analyze_sentiment))

# Виведемо перші кілька рядків DataFrame для перевірки
print("\n")
print("Sentiment та Subjectivity аналіз: ")
print(data[['Comment', 'polarity', 'subjectivity', 'Emotion']].head())

# 4. Візуалізувати отримані результати за категоріями (класами) та показниками

# Set the style of seaborn
sns.set(style="whitegrid")

# Visualize sentiment polarity by category
plt.figure(figsize=(12, 6))
sns.boxplot(x='Emotion', y='polarity', data=data)
plt.title('Sentiment Polarity by Category')
plt.xlabel('Emotion Category')
plt.ylabel('Sentiment Polarity')
plt.show()

# Visualize subjectivity by category
plt.figure(figsize=(12, 6))
sns.boxplot(x='Emotion', y='subjectivity', data=data)
plt.title('Subjectivity by Category')
plt.xlabel('Emotion Category')
plt.ylabel('Subjectivity')
plt.show()

# 5. Якщо дані дають таку можливість - провести оцінку якості роботи алгоритму

# Створення нового стовпця 'polarity'
data['polarity'] = data['Comment'].apply(analyze_sentiment)

# Конвертація polarity у бінарний sentiment (позитивний чи негативний)
data['sentiment'] = data['polarity'].apply(lambda x: 'positive' if x[0] > 0 else 'negative')

# Оцінка точності sentiment analysis
accuracy = accuracy_score(data['Emotion'], data['sentiment'])
print("Точність аналізу відчуттів: ", accuracy)

# Звіт про класифікацію
print("Звіт про класифікацію для аналізу відчуттів: \n", classification_report(data['Emotion'], data['sentiment']))

# 6. Визначити іменники в досліджуваному тексті для конкретизації його тематики

# Функція для визначення іменників у тексті
def extract_nouns(text):
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stopwords.words('english')]
    pos_tags = pos_tag(words)
    nouns = [word for word, pos in pos_tags if pos.startswith('N')]
    return nouns

# Застосування функції до всього стовпця 'Comment' у вашому датасеті
data['nouns'] = data['Comment'].apply(extract_nouns)

# Загальний список іменників у всьому датасеті
all_nouns = [noun for nouns_list in data['nouns'] for noun in nouns_list]

# Виведення найчастіше зустрічаються іменників у всьому датасеті
print("Найчастіше зустрічаються іменники у всьому датасеті: \n", FreqDist(all_nouns).most_common(10))

# 7. Провести моделювання тематики тексту та підібрати оптимальні параметри

def preprocess_text(text):
    # Нормалізація - переведення в нижній регістр
    text = text.lower()

    # Токенізація
    words = word_tokenize(text)

    # Видалення стоп-слів
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Стеммінг
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    # Лематизація
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Видалення пунктуації та цифр
    words = [word for word in words if word.isalpha()]

    # Видалення не часто вживаних слів або несуттєвих для подальших операцій
    # (це може бути додатковий крок, залежно від потреб)
    frequent_words = ["p"]  # Замініть це на реальний список часто вживаних слів
    words = [word for word in words if word not in frequent_words]

    # Об'єднання токенів в рядок
    processed_text = ' '.join(words)

    return processed_text

# Створіть векторизатор для перетворення текстових даних у матрицю документів-термінів
vectorizer = CountVectorizer(stop_words='english')
doc_term_matrix = vectorizer.fit_transform(data['preprocessed_text'])

# Визначте параметри для LDA та виконайте пошук по сітці
params = {
    'n_components': [5, 10, 15],
    'learning_method': ['online', 'batch'],
    'random_state': [42]
}

lda = LatentDirichletAllocation()
grid_search_lda = GridSearchCV(lda, params, cv=3)
grid_search_lda.fit(doc_term_matrix)

# Отримайте найкращі параметри з пошуку по сітці
best_params_lda = grid_search_lda.best_params_
print("Найкращі параметри для LDA:", best_params_lda)

# Навчіть модель LDA з найкращими параметрами
best_lda_model = grid_search_lda.best_estimator_
best_lda_model.fit(doc_term_matrix)

# Отримайте теми та їх ваги для кожного документа
topics_for_documents = best_lda_model.transform(doc_term_matrix)

# Виведіть результати
print("Теми та їх ваги для першого документа:")
for i, topic_weights in enumerate(topics_for_documents[0]):
    print(f"Тема {i + 1}: {topic_weights:.4f}")
    
# 8. Візуалізувати отримані результати щодо тематики тексту

# Візуалізуйте теми для першого документа
plt.figure(figsize=(10, 6))
plt.bar(range(len(topics_for_documents[0])), topics_for_documents[0])
plt.xlabel('Тема')
plt.ylabel('Вага')
plt.title('Теми та їх ваги для першого документа')
plt.show()


