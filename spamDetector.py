import streamlit as st
import pickle

# Load the trained model and vectorizer from files
model = pickle.load(open('spam.pkl', 'rb'))  # Replace 'spam.pkl' with the correct model filename
cv = pickle.load(open('vec.pkl', 'rb'))  # Replace 'vec.pkl' with the correct vectorizer filename

# Set up the Streamlit app
st.title("Email Spam Classification Application")
st.write("This is a Machine Learning application to classify emails as spam or not spam.")

# Create a text area for the user to input an email
user_input = st.text_area("Enter an email to classify", height=200)

# Classification button
if st.button("Classify"):
    if user_input:
        data = [user_input]
        vec = cv.transform(data).toarray()  # Transform the input using the vectorizer
        result = model.predict(vec)  # Predict whether the email is spam or not

        # Display result based on prediction
        if result[0] == 0:
            st.success("This is Not A Spam Email")
        else:
            st.error("This is A Spam Email")
    else:
        st.write("Please enter an email to classify.")

main()