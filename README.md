# IMDB-sentiment-analysis-with-simple-RNN


This Streamlit application allows you to analyze the sentiment of an IMDB movie review, classifying it as either positive or negative. It leverages a pre-trained Recurrent Neural Network (RNN) model to make predictions based on the text you provide.

**Key Features:**

- **Sentiment Classification:** Identifies whether a movie review expresses a positive or negative sentiment.
- **User-Friendly Interface:** Provides a simple text area for entering your review and clear output for the classification.
- **Pre-trained Model:** Utilizes a pre-trained RNN model, eliminating the need for model training on your end.

**How it Works:**

1. **Enter a Movie Review:** Type your review into the designated text area.
2. **Preprocessing:** The application cleans up your review by removing special characters, converting it to lowercase, splitting it into words, and performing lemmatization (converting words to their base form). Additionally, it removes common stop words (e.g., "the", "a") that don't contribute significantly to sentiment analysis.
3. **One-Hot Encoding:** Each word in the review is converted into a numerical representation using one-hot encoding.
4. **Padding:** The review is padded with extra zeros to ensure it has the same length as other reviews expected by the model.
5. **Prediction:** The pre-trained RNN model analyzes the processed review and predicts its sentiment as positive or negative.
6. **Output:** The application displays the predicted sentiment ("Positive" or "Negative") along with a score indicating the model's confidence (a value closer to 1 signifies a stronger positive sentiment, while a value closer to 0 indicates a stronger negative sentiment).

**Technical Details:**

- **Framework:** Streamlit
- **Machine Learning Model:** Recurrent Neural Network (RNN)
- **Preprocessing Techniques:** Text Cleaning, Lowercasing, Tokenization, Stop Word Removal, Lemmatization, One-Hot Encoding, Padding

**Disclaimer:**

The pre-trained model used in this application may not be perfect and could potentially misclassify some reviews. As with any machine learning model, the accuracy depends on the quality of the training data. 
