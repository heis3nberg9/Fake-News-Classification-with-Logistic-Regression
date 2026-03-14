import pickle
import sys

def load_and_predict():
    try:
        # 1. Load the saved model and vectorizer
        model = pickle.load(open('fake_news_model.pkl', 'rb'))
        vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
        
        print("--- Fake News Detector Loaded ---")
        print("Type 'exit' to quit.")
        
        while True:
            # 2. Get user input
            user_input = input("\nEnter a news headline to check: ")
            
            if user_input.lower() == 'exit':
                break
            
            if not user_input.strip():
                continue

            # 3. Process the input (Must be a list/array for the vectorizer)
            data = [user_input]
            vectorized_input = vectorizer.transform(data)
            
            # 4. Make prediction
            prediction = model.predict(vectorized_input)
            
            # 5. Show results
            result = "🚨 FAKE NEWS" if prediction[0] == 1 else "✅ REAL NEWS"
            print(f"Result: {result}")

    except FileNotFoundError:
        print("Error: Could not find 'fake_news_model.pkl' or 'tfidf_vectorizer.pkl'.")
        print("Make sure you have run the saving code in your Jupyter Notebook first!")

if __name__ == "__main__":
    load_and_predict()