# streamlit_app.py
import streamlit as st
import requests

# Streamlit UI setup
st.title("AI Query Assistant")
st.write("Enter your query below to find relevant information.")

# User input for the query
query_text = st.text_input("Your Query:", "I am facing problem during logging in")

# When the button is pressed
if st.button("Search"):
    with st.spinner("Searching for the closest chunk..."):
        # Send the query to the FastAPI backend
        try:
            response = requests.post("http://127.0.0.1:8000/search", json={"query_text": query_text})
            response_data = response.json()

            if "summary" in response_data:
                # Display the summary in a styled container for better UI/UX
                st.markdown("""
                    <div style='border: 2px solid #FF4B4B; border-radius: 10px; padding: 15px; background-color: #FFF0F0;'>
                        <h3 style='color: #FF4B4B;'>Summary:</h3>
                        <p style='font-size: 16px; line-height: 1.6;'>{}</p>
                    </div>
                """.format(response_data["summary"]), unsafe_allow_html=True)
            else:
                st.error(response_data.get("error", "An unknown error occurred."))

        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred: {e}")
