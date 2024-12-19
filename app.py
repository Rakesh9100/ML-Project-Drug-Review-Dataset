import requests
import math
import warnings
from bs4 import BeautifulSoup
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


warnings.filterwarnings("ignore")


def get_data_from_excel():
    dtypes = {
        "Unnamed: 0": "int32",
        "drugName": "category",
        "condition": "category",
        "review": "category",
        "rating": "float32",
        "date": "string",
        "usefulCount": "int16",
    }
    chunk_size = 10000  # Define the size of each chunk
    train_chunks = []
    # Load the training dataset in chunks
    for chunk in pd.read_csv(
        r"datasets/drugsComTrain_raw.tsv",
        sep="\t",
        quoting=2,
        dtype=dtypes,
        chunksize=chunk_size,
    ):
        train_chunks.append(chunk)
    # Concatenate all the chunks to create the training dataframe
    train_df = pd.concat(train_chunks)
    test_chunks = []
    # Load the test dataset in chunks
    for chunk in pd.read_csv(
        r"datasets/drugsComTest_raw.tsv",
        sep="\t",
        quoting=2,
        dtype=dtypes,
        chunksize=chunk_size,
    ):
        test_chunks.append(chunk)
    # Concatenate all the chunks to create the test dataframe
    test_df = pd.concat(test_chunks)

    ## Converting date column to datetime format
    train_df["date"], test_df["date"] = pd.to_datetime(
        train_df["date"], format="%B %d, %Y"
    ), pd.to_datetime(test_df["date"], format="%B %d, %Y")

    ## Extracting day, month, and year into separate columns
    for df in [train_df, test_df]:
        df["day"] = df["date"].dt.day.astype("int8")
        df["month"] = df["date"].dt.month.astype("int8")
        df["year"] = df["date"].dt.year.astype("int16")

    return train_df, test_df, df


def home(df):
    # Add CSS style to center the title
    center_css = """
        <style>
        .title {
            text-align: center;
        }
        </style>
        """
    st.markdown(center_css, unsafe_allow_html=True)
    # Add header image
    header_image = "https://wisdomml.in/wp-content/uploads/2023/03/drug.png"  # Replace with the path to your image
    st.image(header_image, use_container_width=True)

    def web_scraping(qs):
        try:
            URL = "https://www.drugs.com/" + qs + ".html"
            page = requests.get(URL)
            soup = BeautifulSoup(page.content, "html.parser")
            title = soup.find("h2", id="uses").text
            description = soup.find("h2", id="uses").find_next("p").text
            warnings = soup.find("h2", id="warnings").find_next("strong").text
            before_taking_title = soup.find("h2", id="before-taking").text
            before_taking_items = (
                soup.find("h2", id="before-taking").find_next("ul").find_all("li")
            )
            before_taking_list = [item.text.strip() for item in before_taking_items]

            # Creating a dictionary with the extracted information
            result = {
                "title": title,
                "description": description,
                "warnings": warnings,
                "before_taking_title": before_taking_title,
                "before_taking_list": before_taking_list,
            }

            return result

        except requests.exceptions.HTTPError:
            print("Page not found. Please check the input.")
        except Exception as e:
            print("An error occurred:", str(e))

    st.sidebar.header("Please Filter Here:")
    drug = st.sidebar.text_input("Enter the drug name")

    # Only display the dashboard if a drug name is entered
    if drug:
        review = st.sidebar.text_input("Enter your review")
        drug = drug.title()
        df_selection = df.query("drugName == @drug")
        df_selection.head()
        # ---- MAINPAGE ----
        st.title(":bar_chart: Drug Dashboard")
        st.markdown("##")
        # TOP KPI's
        total_count = int(df_selection["usefulCount"].sum())
        average_rating = round(df_selection["rating"].mean(), 1)
        if math.isnan(average_rating):
            star_rating = 0
        else:
            star_rating = ":star:" * int(round(average_rating, 0))
        left_column, middle_column, right_column = st.columns(3)

        with left_column:
            st.subheader("Useful Count:")
            st.subheader(f" {total_count:,}")

        with middle_column:
            st.subheader("Average Rating:")
            st.subheader(f"{average_rating} ")

        with right_column:
            st.subheader("Approx Ratings")
            st.subheader(f"{star_rating}")
        # returnnig info about drugs from drugs.com
        scraped_data = web_scraping(drug.lower())
        if scraped_data:
            # Displaying the extracted information using Streamlit
            st.subheader("Title")
            st.write(scraped_data["title"])
            st.subheader("Description")
            st.write(scraped_data["description"])

            st.subheader("Warnings")
            st.write(scraped_data["warnings"])

            st.subheader(scraped_data["before_taking_title"])
            for item in scraped_data["before_taking_list"]:
                st.write("- " + item)
        else:
            st.error("Page not found. Please check the input.")

        # Additional dashboard components and visualizations can be added here
    else:
        st.write("Please enter a drug name.")
    st.markdown("""---""")

    # ---- HIDE STREAMLIT STYLE ----
    hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """
    st.markdown(hide_st_style, unsafe_allow_html=True)
    # Rest of the code for the home page


def admin(train_df, test_df, df):
    # Function to display data visualization
    def preprocess_data(train_df, test_df):
        X_train = train_df.drop("rating", axis=1)
        X_test = test_df.drop("rating", axis=1)
        Y_train = train_df["rating"]
        Y_test = test_df["rating"]
        # Set the column names to string type
        X_train.columns = X_train.columns.astype(str)
        X_test.columns = X_test.columns.astype(str)
        return train_df, test_df, X_train, Y_train

    def visualize_data(train_df, test_df, X_train, Y_train):
        # Scatter Plot: Drug Name vs Ratings (Testing Data)
        st.subheader("Scatter Plot: Drug Name vs Ratings (Testing Data)")
        fig, ax = plt.subplots(figsize=(8, 6))  # Adjust the figsize as needed
        ax.scatter(LabelEncoder().fit_transform(test_df["drugName"]), test_df["rating"])
        ax.set_xlabel("Drug Name")
        ax.set_ylabel("Ratings")
        st.pyplot(fig)

        # Scatter Matrix for training dataset
        feature_columns = ["drugName", "condition", "rating", "usefulCount"]
        st.subheader("Scatter Matrix For Training Dataset")
        plt.figure(figsize=(8, 6))  # Adjust the figsize as needed
        sns.pairplot(train_df[feature_columns])
        st.pyplot()

        histogram_drugname = plt.hist(train_df["drugName"], bins=50)
        plt.xlabel("Bins")
        plt.ylabel("Frequency")
        plt.title("Drug Name Histogram (Training Dataset)")
        fig = plt.figure()
        plt.hist(histogram_drugname[0], bins=histogram_drugname[1])
        col3, col4 = st.columns(2)
        with col3:
            st.pyplot(fig)

        # Box Plot of Rating
        st.subheader("Box Plot of Rating")
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="rating", data=train_df)
        with col4:
            st.pyplot()

    def display(train_df, test_df):
        # Display data description
        st.subheader("Data Description")
        st.write(df.describe())
        # Display sample data
        st.subheader("Sample Data")
        st.write(df)
        train_df, test_df, X_train, Y_train = preprocess_data(train_df, test_df)
        visualize_data(train_df, test_df, X_train, Y_train)

    display(train_df, test_df)


# Main function
def main():
    st.set_page_config(page_title="Drug Review Analysis", layout="wide")
    st.title("Drug Review Analysis")

    train_df, test_df, df = get_data_from_excel()
    # Create a navigation bar for tab selection
    tab = st.radio(
        "Select Tab",
        options=["Home", "Admin"],
        index=0,  # Set the default index to select Home page
        format_func=lambda x: x.upper(),  # Display tab names in uppercase
    )
    # Display corresponding page based on selected tab
    if tab == "Home":
        home(df)
    elif tab == "Admin":
        admin(train_df, test_df, df)


if __name__ == "__main__":
    # st.set_page_config(page_title="Drugs Review", page_icon=":bar_chart:", layout="wide")
    main()
