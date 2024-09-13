import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import json
import os
import requests
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()

# --- User Management Functions ---
USER_FILE = "users.json"

def load_users():
    if os.path.exists(USER_FILE):
        with open(USER_FILE, "r") as file:
            return json.load(file)
    else:
        return {}

def save_users(users):
    with open(USER_FILE, "w") as file:
        json.dump(users, file)

def register(username, password):
    users = load_users()
    if username in users:
        return False
    users[username] = password
    save_users(users)
    return True

def authenticate(username, password):
    users = load_users()
    return users.get(username) == password

def login():
    st.session_state["username"] = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        if authenticate(st.session_state["username"], password):
            st.session_state["authenticated"] = True
            st.sidebar.success(f"Logged in as {st.session_state['username']}!")
        else:
            st.sidebar.error("Invalid username or password")

def signup():
    new_username = st.sidebar.text_input("New Username")
    new_password = st.sidebar.text_input("New Password", type="password")

    if st.sidebar.button("Sign Up"):
        if register(new_username, new_password):
            st.sidebar.success("Registration successful! Please log in.")
        else:
            st.sidebar.error("Username already taken. Please choose a different one.")

def logout():
    st.session_state["authenticated"] = False
    st.session_state.pop("username", None)
    st.sidebar.success("Logged out successfully!")

# Initialize session state for authentication and chatbot history
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Initialize Groq client
def initialize_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY environment variable not set.")
        return None
    try:
        return Groq(api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing Groq client: {e}")
        return None

# Function to get chatbot response
def get_chatbot_response(message):
    client = initialize_groq_client()
    if client is None:
        return "Error initializing Groq client."

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": message}
            ],
            model="llama3-8b-8192",
        )
        return chat_completion.choices[0].message.content
    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error occurred: {http_err}"
    except requests.exceptions.RequestException as req_err:
        return f"Request error occurred: {req_err}"
    except Exception as e:
        return f'Error contacting the chatbot service: {e}'

# --- Streamlit App ---
st.title("Tshepo's Health Livestock Monitor")

# Initialize filtered_df
filtered_df = pd.DataFrame()

if st.session_state["authenticated"]:
    # Add a logout button
    if st.sidebar.button("Logout"):
        logout()

    # Load data
    data = {
        'Animal ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Breed': ['Cow', 'Cow', 'Cow', 'Pig', 'Pig', 'Pig', 'Sheep', 'Sheep', 'Sheep', 'Goat'],
        'Weight (kg)': [50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
        'Temperature (°C)': [38.5, 39.2, 40.1, 40.5, 41.0, 41.5, 42.0, 42.5, 43.0, 43.5],
        'Heart Rate (bpm)': [60, 70, 80, 90, 100, 110, 120, 130, 140, 150],
        'Respiratory Rate (bpm)': [20, 25, 30, 35, 40, 45, 50, 55, 60, 65],
        'Health Status': ['Healthy', 'Healthy', 'Sick', 'Healthy', 'Sick', 'Healthy', 'Sick', 'Healthy', 'Sick', 'Healthy']
    }

    df = pd.DataFrame(data)

    # Function to load images for animals
    def load_animal_image(breed):
        images = {
            'Cow': 'images/cow.jpg',
            'Pig': 'images/pig.jpg',
            'Sheep': 'images/sheep.jpg',
            'Goat': 'images/goat.jpg'
        }
        return images.get(breed, 'https://example.com/default_image.jpg')

    # Function to suggest medication based on health metrics
    def suggest_medication(row):
        suggestions = []
        
        # Real-world medications for different animals
        medications = {
            'Cow': {
                'fever': ["Ibuprofen", "Flunixin Meglumine", "Ketoprofen"],
                'heart': ["Propranolol", "Atenolol"],
                'respiratory': ["Oxytetracycline", "Enrofloxacin"]
            },
            'Pig': {
                'fever': ["Paracetamol", "Doxycycline", "Sulfaquinoxaline"],
                'heart': ["Digoxin", "Epinephrine"],
                'respiratory': ["Tylosin", "Florfenicol"]
            },
            'Sheep': {
                'fever': ["Ketoprofen", "Flunixin Meglumine"],
                'heart': ["Propranolol", "Atropine"],
                'respiratory': ["Oxytetracycline", "Chlortetracycline"]
            },
            'Goat': {
                'fever': ["Ibuprofen", "Flunixin Meglumine"],
                'heart': ["Digoxin", "Isoproterenol"],
                'respiratory': ["Tilmicosin", "Doxycycline"]
            }
        }

        if row["Temperature (°C)"] > 40.0:
            suggestions.extend(medications.get(row["Breed"], {}).get('fever', []))
        if row["Heart Rate (bpm)"] > 100:
            suggestions.extend(medications.get(row["Breed"], {}).get('heart', []))
        if row["Respiratory Rate (bpm)"] > 50:
            suggestions.extend(medications.get(row["Breed"], {}).get('respiratory', []))

        return ", ".join(suggestions) if suggestions else "No medication needed"

    # Sidebar filters
    st.sidebar.header("Filters")
    breed_filter = st.sidebar.selectbox("Breed", df["Breed"].unique())
    weight_filter = st.sidebar.slider("Weight (kg)", 0, 200, (0, 200))
    temp_filter = st.sidebar.slider("Temperature (°C)", 30.0, 45.0, (30.0, 45.0))
    hr_filter = st.sidebar.slider("Heart Rate (bpm)", 40, 160, (40, 160))
    rr_filter = st.sidebar.slider("Respiratory Rate (bpm)", 15, 70, (15, 70))

    filtered_df = df[
        (df["Breed"] == breed_filter) &
        (df["Weight (kg)"].between(weight_filter[0], weight_filter[1])) &
        (df["Temperature (°C)"].between(temp_filter[0], temp_filter[1])) &
        (df["Heart Rate (bpm)"].between(hr_filter[0], hr_filter[1])) &
        (df["Respiratory Rate (bpm)"].between(rr_filter[0], rr_filter[1]))
    ]

    # Display the image of the selected breed at the top
    st.image(load_animal_image(breed_filter), caption=f"Selected Breed: {breed_filter}", use_column_width=True)

    # Calculate feed amounts and revenues
    feed_amount_per_kg = {
        'Cow': {'Standard': 0.1, 'High Protein': 0.12, 'Organic': 0.14},
        'Pig': {'Standard': 0.08, 'High Protein': 0.1, 'Organic': 0.12},
        'Sheep': {'Standard': 0.07, 'High Protein': 0.09, 'Organic': 0.1},
        'Goat': {'Standard': 0.09, 'High Protein': 0.11, 'Organic': 0.13}
    }

    feed_type = st.sidebar.selectbox("Feed Type", ["Standard", "High Protein", "Organic"])

    if breed_filter in feed_amount_per_kg:
        amount_per_kg = feed_amount_per_kg[breed_filter][feed_type]
        st.sidebar.write(f"Feed Amount (kg per kg of animal weight): {amount_per_kg}")

    # Revenue Estimator in ZAR
    st.sidebar.header("Revenue Estimator")
    price_per_kg = st.sidebar.number_input("Market Price per kg (ZAR)", min_value=0.0, value=100.0, format="%.2f")
    
    if not filtered_df.empty:
        filtered_df['Estimated Revenue (ZAR)'] = filtered_df['Weight (kg)'] * price_per_kg
        st.write("Estimated Revenue for Filtered Animals:")
        st.write(filtered_df[['Animal ID', 'Breed', 'Weight (kg)', 'Estimated Revenue (ZAR)']])

    # Paddocks Management
    st.sidebar.header("Paddocks Management")
    paddocks = st.sidebar.multiselect("Assign to Paddocks", ["Paddock 1", "Paddock 2", "Paddock 3"])
    
    # Assign animals to selected paddocks
    if not filtered_df.empty:
        filtered_df['Assigned Paddock'] = ", ".join(paddocks)
        st.write("Animals Assigned to Paddocks:")
        st.write(filtered_df[['Animal ID', 'Assigned Paddock']])

    # Servings Calculation
    st.sidebar.header("Servings")
    servings_per_kg = {
        'Cow': {'Standard': 0.1, 'High Protein': 0.12, 'Organic': 0.14},
        'Pig': {'Standard': 0.08, 'High Protein': 0.1, 'Organic': 0.12},
        'Sheep': {'Standard': 0.07, 'High Protein': 0.09, 'Organic': 0.1},
        'Goat': {'Standard': 0.09, 'High Protein': 0.11, 'Organic': 0.13}
    }

    if breed_filter in servings_per_kg:
        servings = servings_per_kg[breed_filter][feed_type] * filtered_df['Weight (kg)']
        filtered_df['Daily Servings (kg)'] = servings
        st.write("Daily Feed Servings for Filtered Animals:")
        st.write(filtered_df[['Animal ID', 'Daily Servings (kg)']])

    # Display filtered data with medication suggestions
    filtered_df['Medication Suggestions'] = filtered_df.apply(suggest_medication, axis=1)
    st.write("Filtered Data with Medication Suggestions:")
    st.write(filtered_df)

    # Health Metrics Visualization
    if not filtered_df.empty:
        fig = px.scatter(filtered_df, x="Weight (kg)", y="Temperature (°C)",
                         color="Health Status", size="Heart Rate (bpm)",
                         hover_name="Animal ID", size_max=60, title="Animal Health Metrics")
        st.plotly_chart(fig)

    # Train a model
    X = df[['Weight (kg)', 'Temperature (°C)', 'Heart Rate (bpm)', 'Respiratory Rate (bpm)']]
    y = df['Health Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

   

    # --- Chatbot Integration ---
    st.sidebar.header("Chat with the Bot")
    user_input = st.sidebar.text_input("Ask a question or describe symptoms")

    if st.sidebar.button("Send"):
        if user_input:
            chatbot_response = get_chatbot_response(user_input)
            st.session_state.chat_history.append(f"You: {user_input}")
            st.session_state.chat_history.append(f"Bot: {chatbot_response}")
        else:
            st.sidebar.error("Please enter a message.")

    # Display chat history
    if st.session_state.chat_history:
        st.write("Chat History:")
        for chat in st.session_state.chat_history:
            st.write(chat)

else:
    # Show login and signup options
    auth_option = st.sidebar.radio("Choose an option", ["Login", "Sign Up"])
    if auth_option == "Login":
        login()
    else:
        signup()
