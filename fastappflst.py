import streamlit as st

# Function to render the Flask webpage in Streamlit
def render_flask_webpage(url):
    st.components.v1.iframe(url, width=1000, height=800)

# Main Streamlit code
if __name__ == '__main__':
    st.title('Render Flask Webpage in Streamlit')
    
    # URL of the Flask webpage
    flask_webpage_url = "http://localhost:5000/"  # Replace with your Flask app URL
    
    # Render the Flask webpage
    render_flask_webpage(flask_webpage_url)
