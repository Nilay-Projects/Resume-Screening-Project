"""
Topic : Resume Screening Project

"""

# Step 1 : Importing Required Library's

# Importing Basic Library's
import pandas as pd
import re
import string
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Importing NLP Library's
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer

# Importing Scikit Learn Library's
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Importing Keras Library's
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.losses import SparseCategoricalCrossentropy

# Importing warning's library to prevent warnings
import warnings

warnings.filterwarnings('ignore')

# Required For Downloading stopwords Package
nltk.download('stopwords')

# Required For Downloading Tokenization Package
nltk.download('punkt')
nltk.download('punkt_tab')
# Required For Downloading Lemmatization Package
nltk.download('wordnet')


# Making class
class Resume_screening:

    # Function for initializing the variables
    def __init__(self, path="data/ResumeDataset.csv"):

        # Loading the dataset
        self.data = pd.read_csv(path)

        # Creating necessary variables
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred_test = None
        self.y_pred_train = None
        self.Model = None

    # Step 2: Viewing the dataset

    """
    Overview : This is the most basic step to view the dataset
               Methods to view the dataset are as follows

               Step 1 : Printing the head,tail(First five rows and last five rows)
               Step 2 : Printing the Information of the dataset by info method
               Step 3 : Printing the total number of categories from category column
               Step 4 : Counting the null values 
               Step 5 : Counting the duplicates values
               step 6 : Counting the total number of categories
    """

    # Function for viewing the dataset
    def Viewing_the_dataset(self):

        # Printing the head
        print('=' * 80)
        print("\n Printing the head : \n")
        print(self.data.head())

        # Printing the tail
        print('=' * 80)
        print("\n Printing the tail : \n")
        print(self.data.tail())

        # Printing the information of whole dataset by info method
        print('=' * 80)
        print("\n Printing the info of the dataset :  \n")
        self.data.info()

        # Counting the null values
        print('=' * 80)
        print("\n Counting the null values : \n")
        print(self.data.isna().sum())

        # Counting the duplicates values
        print('=' * 80)
        print("\n Counting the duplicates values :\n")
        print(self.data.duplicated().sum())

        print("\n Here we get 796 duplicates value which is 82% (percent) of the dataset\nso we can't "
              "drop this 82% percent of the dataset,if we drop them then we have \n only 166 rows left ")

        # Displaying the distinct categories of resume:
        print('=' * 80)
        print("\n Displaying the Distinct categories of resume:\n")
        print(self.data['Category'].unique())

        # Displaying the number of resumes according to their Category
        print('=' * 80)
        print("\n Displaying the number of resume's according to their Category \n")
        print(self.data['Category'].value_counts())

    """
    Step 3 : Modifying the dataset

    Overview : In this step we merge some similar categories into one single category 
               That is we merge some categories and made new categories named as Quality Assurance(QA) 
               and Human Resources(HR).

    """

    # Function for Categorizing_the_category
    def Categorizing_the_category(self):
        print('=' * 80)
        print("\n Merging some similar categories of the resume \n")

        # Making Category for Quality_Assurance
        self.data['Category'] = self.data['Category'].replace(['Testing', 'Automation Testing'],
                                                              'Quality_Assurance')

        # Making Category for HR
        self.data['Category'] = self.data['Category'].replace(['Health and fitness', 'Arts'], 'HR')

        # Displaying the category's after merging similar categories of resume
        print('=' * 80)
        print("\n Displaying the category's after merging similar categories of resume \n")
        print(self.data['Category'].unique())

        # Displaying the total categories after merging similar categories of resume
        print('=' * 80)
        print("\n Displaying the total categories after merging similar categories of resume:\n")
        print(self.data['Category'].nunique())

        # Displaying the number of resume's according to their Category after merging similar categories
        print('=' * 80)
        print("\n Displaying the number of resume's according to their Category after merging similar categories\n")
        print(self.data['Category'].value_counts())

        # Plotting the count plot for Category

        plt.figure(figsize=(15, 10))
        sns.countplot(y="Category", data=self.data)
        plt.title("Graph for displaying the numbers of resume in each category")
        plt.xlabel("Counts for number of resume's")
        plt.grid()
        plt.show()

        # Plotting pie charts for Category

        Category_counts = self.data['Category'].value_counts()
        Category_labels = self.data['Category'].unique()

        plt.figure(figsize=(15, 12))
        plt.pie(Category_counts, labels=Category_labels, autopct='%1.1f%%', shadow=True)
        plt.title("Pie chart for displaying the percent of resume in each category")
        plt.show()

    """
    # Step 4: Data cleaning

    Overview : Step 4 is divided into 2 parts ,

    First part : To clean the dataset by applying python core functions that is to remove punctuations,to decode the
                 unidentified letters and words and to remove the extra white spaces,
                 And by using regular expressions to remove URLs.

    Second part : After we get the clean dataset from first part we have to apply NLP (Natural Language Processing)

                  Steps of Nlp are as follows :

                  Step 1 : Lower the text
                  Step 2 : Make tokens
                  Step 3 : Remove stop words
                  Step 4 : Lemmatize the text
                  Step 5 : Apply Tfidf to encode the text

    """

    # Step 4.1 : Cleaning the dataset by Python core functions and Regular expressions

    def Data_Cleaning(self):

        # Assigning a new column for cleaned resume text
        self.data['cleaned_resume'] = ''

        # Step 1 : Function for removing URLs
        def Removing_urls(raw_text_of_Resume):
            raw_text_of_Resume = re.sub(r'http\S+\s*|www\S+\s*', '', raw_text_of_Resume)
            raw_text_of_Resume = re.sub(r'[a-zA-Z0-9]\S+com\s*', '', raw_text_of_Resume)

            return raw_text_of_Resume

        self.data['cleaned_resume'] = self.data['Resume'].apply(Removing_urls)

        # Step 2 : Function Decoding the unidentified characters and words
        def Decoding_the_text(raw_text_of_Resume):
            raw_text_of_Resume = raw_text_of_Resume.encode('ascii', 'ignore')
            raw_text_of_Resume = raw_text_of_Resume.decode()
            return raw_text_of_Resume

        self.data['cleaned_resume'] = self.data['cleaned_resume'].apply(Decoding_the_text)

        # Step 3 : Function for removing punctuations and extra white spaces
        def Removing_punctuations_and_white_spaces(raw_text_of_Resume):
            # Removing punctuations
            for i in string.punctuation:
                raw_text_of_Resume = raw_text_of_Resume.replace(i, '')

            # Removing extra white spaces
            raw_text_of_Resume = " ".join(raw_text_of_Resume.split())
            return raw_text_of_Resume

        self.data['cleaned_resume'] = self.data['cleaned_resume'].apply(Removing_punctuations_and_white_spaces)

        # Printing the head after clean data by core python functions
        print('=' * 80)
        print("\n Printing the head after cleaning the dataset by python core functions \n")
        print(self.data.head())

    # Step 4.2 : Applying NLP (Part 2 of data cleaning)

    def Applying_the_NLP(self):

        # Step 1 : Function for Convert text to lower case
        def Lowering_the_text(raw_text_of_Resume):
            raw_text_of_Resume = raw_text_of_Resume.lower()
            return raw_text_of_Resume

        self.data['cleaned_resume'] = self.data['cleaned_resume'].apply(Lowering_the_text)

        # Step 2 : Function for Making Word Tokens
        def Making_words_tokens(raw_text_of_Resume):
            raw_text_of_Resume = word_tokenize(raw_text_of_Resume)
            return raw_text_of_Resume

        self.data['cleaned_resume'] = self.data['cleaned_resume'].apply(Making_words_tokens)

        # Step 3 : Function for Removing Stop words
        def Removing_stop_words(raw_text_of_Resume):
            set_of_stopwords = stopwords.words("english")
            lst = []
            for i in raw_text_of_Resume:
                if i not in set_of_stopwords:
                    lst.append(i)
            raw_text_of_Resume = " ".join(lst)

            return raw_text_of_Resume

        self.data['cleaned_resume'] = self.data['cleaned_resume'].apply(Removing_stop_words)

        # Step 4 : Function for making lemmas of the text
        def lemmatize_the_text(raw_text_of_Resume):
            lemmatizer = WordNetLemmatizer()

            words = raw_text_of_Resume.split()
            words = [lemmatizer.lemmatize(word, pos='v') for word in words]
            raw_text_of_Resume = " ".join(words)

            return raw_text_of_Resume

        self.data['cleaned_resume'] = self.data['cleaned_resume'].apply(lemmatize_the_text)

        #  Printing one particular row after data cleaning
        print('=' * 80)
        print("\n Printing one particular row after applying NLP and data cleaning \n")
        print(self.data['cleaned_resume'][0])

    """ 
    Step 5 : Verifying the clean dataset

    Overview : After we clean the raw text of resume we have to verify the dataset is cleaned or not 
               for that this step is required.

               In this step I added two more columns, First column is for word count in raw text of resume and 
               second column is for word count for cleaned text of resume.

               After that I plot the pie chart that shows the weightage of word counts for both the column,Here I get
               more weightage of word count in raw text in comparison to word count in cleaned text of the resume 
               this shows that the data gets cleaned after data cleaning process.

    """

    # Function for Verifying the clean data text
    def Verifying_the_clean_data_text(self):

        # Step 1: Function for fetching Numbers of words in the resume
        def Fetching_the_len_words(words):
            words = len(word_tokenize(words))
            return words

        # Step 2: Assigning the len of words for raw text of resume in new column
        self.data['Num_of_words_in_raw_text_resume'] = self.data['Resume'].apply(Fetching_the_len_words)

        # Step 3 : Assigning the len of words for clean resume text in new column
        self.data['Num_of_words_in_clean_resume'] = self.data['cleaned_resume'].apply(Fetching_the_len_words)

        # Step 4 : Printing the sum of words count
        print('=' * 80)
        print("\n Sum of words count in raw text of resume :\n")
        print(self.data['Num_of_words_in_raw_text_resume'].sum())

        print('=' * 80)
        print("\n Sum of words count in cleaned text of resume :\n")
        print(self.data['Num_of_words_in_clean_resume'].sum())

        # Step 5 : Plotting the pie chart
        labels = ['Weightage of word count in raw text of resume', "Weightage of word count in cleaned text of resume"]

        words_count = [self.data['Num_of_words_in_raw_text_resume'].sum(),
                       self.data['Num_of_words_in_clean_resume'].sum()]

        # Creating pie plot
        plt.figure(figsize=(10, 7))
        plt.pie(words_count, labels=labels, autopct='%1.1f%%', shadow=True)
        plt.title("Pie chart for displaying the weightage of word count in cleaned text "
                  "of resume and raw text of resume,"
                  "\n\n This pie chart shows that words get decreased in clean text of resume")
        plt.show()

        print('=' * 80)
        print("\n Printing the head after adding the new columns for word counts \n")
        print(self.data.head())

        # step 6 : Dropping the added two columns

        self.data = self.data.drop(['Num_of_words_in_raw_text_resume', 'Num_of_words_in_clean_resume'], axis=1)

    """    
    Step 6: Encoding the columns

    Overview : This step is very important because this step encode the columns to feed the model with machine code 
               Models can only work with numerical values. For this reason, it is necessary to 
               transform the categorical values and text of resume into numerical form. 
               This process is called Encoding.

    """

    def Encoding_the_columns(self):

        # Step 1 : Applying Tf-idf(Term Frequency-Inverse Document Frequency) vectorization to encode the resume text
        Tfidf = TfidfVectorizer()
        Tfidf_data = Tfidf.fit_transform(self.data['cleaned_resume']).toarray()

        # Step 2 : Applying Label encoder to category column
        lb = LabelEncoder()
        self.data['Category'] = lb.fit_transform(self.data['Category'])

        # Step 3 : Assigning the values to X and y variables
        self.X = Tfidf_data
        self.y = self.data['Category']

    """
    Step 7 : Splitting the dataset

    Overview : This is also one of the most important step before creating the model,
               because this step will split the data into two parts,
               that is testing data and training data,Training data is used to train the model,
               while testing data is used to evaluate or to test the model
    """

    # Function for Splitting the dataset
    def Splitting_the_dataset(self):

        # Step 1 : Splitting the dataset into train and test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y)

        # Summarizing the dataset
        print('=' * 80)
        print("\n Printing the shapes :\n")
        print("The shape of X train : ", self.X_train.shape)
        print("The shape of Y train : ", self.y_train.shape)
        print("The shape of X test : ", self.X_test.shape)
        print("The shape of y test : ", self.y_test.shape)

        # Normalizing the dataset
        self.y_test = np.array(self.y_test)
        self.y_train = np.array(self.y_train)

    """
    Step 8  : Making Model

    Overview : In this step we apply the Machine Learning model to the dataset
               The Steps for Machine Learning model are as follows :

               Step 1 : Making the model
               Step 2 : Fitting the model
               Step 3 : Making variable for Predicting the values for testing data (y_pred_test)
               Step 4 : Making variable for Predicting the values for training data (y_pred_train)
               Step 5 : Printing the accuracy of training and testing data
               Step 6 : Printing the Precision Score,Recall Score,F1 Score

    """

    # Function for applying Machine Learning
    def Applying_Machine_Learning_Model(self):
        print('=' * 140)
        print("\n Applying Machine Learning \n")
        print('=' * 140)

        # Step 1 : Making Machine Learning Model
        self.Model = MultinomialNB()

        # Step 2 : Fitting the model
        self.Model.fit(self.X_train, self.y_train)

        # Step 3 : Making variable for Predicting the values for testing data
        self.y_pred_test = self.Model.predict(self.X_test)

        # Step 4 : Making variable for Predicting the values for training data
        self.y_pred_train = self.Model.predict(self.X_train)

    # Function for printing ML Performance
    def Print_ML_Performance(self):

        # Step 5 : Printing the accuracy of training and testing data
        print("\n Testing Accuracy by Multinomial NB ML Model \n")
        print(accuracy_score(self.y_test, self.y_pred_test) * 100)
        print("\n")
        print('=' * 80)
        print("\n Training Accuracy by Multinomial NB ML Model \n")

        print(accuracy_score(self.y_train, self.y_pred_train) * 100, "\n")

        Precision_score = precision_score(self.y_test, self.y_pred_test, average='weighted') * 100
        Recall_Score = recall_score(self.y_test, self.y_pred_test, average='weighted') * 100
        F1_score = f1_score(self.y_test, self.y_pred_test, average='weighted') * 100

        # Step 6 : Printing the Precision score,Recall Score and F1 score
        print('=' * 80)
        print("\nPrecision_score\t\tRecall_Score\t\tF1_score")
        print(f"{Precision_score}\t{Recall_Score}\t{F1_score}\n")
        print('=' * 140)

    """ 
    Step 9 : Applying Deep Learning Model To the dataset

    Overview: In this step we have apply the deep learning model to the dataset
              The steps of deep learning are as follows :

              Step 1 : Making Model
              Step 2 : Adding Layers to the model
              Step 3 : Fitting the model
              Step 4 : Evaluating the model
              Step 5 : Printing the summary of the model
    """

    # Function for Making the model and Adding the layers
    def Making_model_and_adding_layers(self):

        print("\n Applying Deep Learning\n")
        print('=' * 140, '\n')

        # Step 1 : Making the model
        self.Model = Sequential()

        # Step 2 : Adding the layers

        # Adding Input layer
        self.Model.add(Flatten())

        # Adding Hidden layer
        self.Model.add(Dense(32, activation="relu"))
        self.Model.add(Dense(32, activation="relu"))

        # Adding Output layer
        self.Model.add(Dense(22, activation="softmax"))

        # Step 3 : Compling the model
        self.Model.compile(loss=SparseCategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])

    # Function for Fitting the model (Step 4)
    def Fitting_the_model(self):
        history = self.Model.fit(self.X_train, self.y_train, epochs=15, validation_data=[self.X_test, self.y_test],
                                 batch_size=128)

        # Printing the loss
        print("\n")
        print('=' * 140)
        print("\n Printing the training data loss : \n")
        print(history.history['loss'])

        # Printing the accuracy
        print("\n Printing the training data accuracy : \n")
        print(history.history['accuracy'])
        print('=' * 80)

        # Plotting the graph of model accuracy and model validation accuracy
        plt.figure(figsize=(10, 8))
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'], linestyle='dashed')
        plt.title('Graph for Displaying Deep Learning Model accuracy')
        plt.xlabel('Epochs --->>')
        plt.ylabel('Accuracy --->>')
        plt.legend(['Training data', 'Validation data'], loc='lower right')
        plt.grid()
        plt.show()

        # Plotting the graph of model loss and model validation loss
        plt.figure(figsize=(10, 8))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'], linestyle='dashed')
        plt.title('Graph for Displaying Deep Learning Model Loss')
        plt.xlabel('Epochs --->>')
        plt.ylabel('Loss ---->>')
        plt.legend(['Training data', 'Validation data'], loc='lower right')
        plt.grid()
        plt.show()

    # Function for evaluating the model (Step 5)
    def Evaluating_the_model(self):
        print("\n Printing the evaluation of the model \n")
        val_loss, val_acc = self.Model.evaluate(self.X_test, self.y_test, verbose=2)
        print('=' * 80)

        print("\n Printing the validation data loss of the model \n")
        print(val_loss)
        print("\n Printing the validation data accuracy of the model \n")
        print(val_acc * 100)

        # Printing the summary (Step 6)
        print('=' * 80)
        print("\n Printing the summary of the model \n")
        self.Model.summary()


# Driver code

if __name__ == "__main__":
    # Making the object of the class
    resume_obj = Resume_screening()

    # Calling all the functions with the help of object

    # Calling the Viewing the dataset function
    resume_obj.Viewing_the_dataset()

    # Calling the Categorizing the category function
    resume_obj.Categorizing_the_category()

    # Calling the Data Cleaning function
    resume_obj.Data_Cleaning()

    # Calling the Applying the NLP function
    resume_obj.Applying_the_NLP()

    # Calling the Verifying the clean data text function
    resume_obj.Verifying_the_clean_data_text()

    # Calling the Encoding the columns function
    resume_obj.Encoding_the_columns()

    # Calling the Splitting the dataset function
    resume_obj.Splitting_the_dataset()

    # Calling the Applying Machine Learning Model function
    resume_obj.Applying_Machine_Learning_Model()

    # Calling the Print ML Performance function
    resume_obj.Print_ML_Performance()

    # Calling the Making model and adding layers function
    resume_obj.Making_model_and_adding_layers()

    # Calling the Fitting the model function
    resume_obj.Fitting_the_model()

    # Calling the Evaluating the model function
    resume_obj.Evaluating_the_model()
