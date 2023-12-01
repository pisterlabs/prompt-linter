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
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# 1. Завантажити необхідні дані

# Завантаження текстових даних та сформування корпусу текстів
data = pd.read_csv('C:/Users/yonaaani/.spyder-py3/project/emotion.csv')

# 2. Підготувати дані на основі підходів з роботи 10

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