import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import folium
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
#from pandas_profiling import ProfileReport
#from streamlit_pandas_profiling import st_profile_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

#Importing the Keras library and packages
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.layers import Dropout
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
from keras import losses

# sns.set_style('darkgrid')

# Insert icon of web app
icon = Image.open("icon.png")
#  Page Layout
st.set_page_config(page_title="EOR Screening by ML", page_icon=icon)

# CSS codes to improve the design of the web app
st.markdown(
    """
<style>
h1 {text-align: center;
}
body {background-color: #DCE3D5;
      width: 1400px;
      margin: 15px auto;
}
</style>""",
    unsafe_allow_html=True,
)

# Insert image
logo = Image.open("background.png")
st.image(logo, width=100, use_column_width=True)

# # Adding of a mp4 video
# st.markdown(
#     """
# **Demo users video**
# """
# )
# video_app = open("app_explain.mp4", "rb")
# st.video(video_app)
# st.markdown("<span style=“background-color:#121922”>", unsafe_allow_html=True)

# Write title and additional information
st.title("EOR METHODS SCREENING BY MACHINE LEARNING")

st.write("---")

# Insert image into left side section
img = Image.open("icon.png")
st.sidebar.image(img)
# Sidebar - collects user input features into dataframe
with st.sidebar.header(":file_folder: 1. Upload the csv data"):
    upload_file = st.sidebar.file_uploader("Upload your csv file", type=["csv"])
    st.sidebar.markdown(
        """
    [Download csv file](https://github.com/lmvu103/Enhanced-Oil-Recovery-Method-Recommendation-/blob/main/DATA%20WORLWIDE%20EOR%20PROJECTSP.csv)
    """
    )

# Sidebar - ML Algorithms
with st.sidebar.subheader(":arrow_down_small: 2. Select ML Algorithm"):
    algorithm = st.sidebar.selectbox(
        "Select algorithm", ("K Nearest Neighbors(knn)", "Decision tree", "ANN")
    )

# Setting parameters
st.sidebar.subheader(":arrow_down_small: 3. Set User Input Parameters")
with st.sidebar.subheader("3.1 Data Split"):
    split_size = st.sidebar.selectbox(
        "Data split ratio (% for training set)", np.arange(10, 91, 10)
    )

with st.sidebar.subheader("3.2 Learning parameters"):
    if algorithm == "K Nearest Neighbors(knn)":
        parameter_k_neighbors = st.sidebar.number_input(
            "Number of K neighbors", 1, 10, 2
        )

    else:
        parameter_decision_tree = st.sidebar.number_input(
            "Number of max depth", 1, 10, 3
        )

with st.sidebar.subheader("3.3 Reservoir Parameters"):
    Porosity = st.sidebar.slider("Porosity (%)", 2, 30)
    Permeability = st.sidebar.slider("Permeability (md)", 8, 5000)
    Depth = st.sidebar.slider("Depth (ft)", 1000, 10000, 1200)
    Gravity = st.sidebar.slider("API Gravity", 5, 40, 8)
    Viscosity = st.sidebar.slider("Oil Viscosity (cp)", 10, 500000, 490058)
    Temperature = st.sidebar.slider("Reservoir Temperature (F)", 50, 300)
    Oil_saturation = st.sidebar.slider("Oil Saturation (%)", 10, 80, 35)

# Calling data processing modules
sc = MinMaxScaler()
le = LabelEncoder()
ohe = OneHotEncoder()


# Model Building
def model(dataframe):
    # Calling the independent and dependent variables
    x = dataframe.iloc[:, 2:9]
    y = dataframe.iloc[:, 1:2]

    # Data details
    st.markdown("**1.2. Data Split**")
    st.write("Training set")
    st.info(x.shape)
    st.info(y.shape)

    # Variable information
    st.markdown("**1.3. Variable details**")
    st.write("Independent Variables")
    st.info(list(x.columns))
    st.write("Dependent Variable")
    st.info(list(y.columns))

    # data processing step
    x = sc.fit_transform(x)
    dfle = dataframe
    dfle.EOR_Method = le.fit_transform(dfle.EOR_Method)
    y = ohe.fit_transform(y).toarray()
    # Data splitting
    x_train, x_test, y_train, y_test = train_test_split( x, y, train_size=split_size, random_state=0)
    # Calling the information that will be used for model prediction
    cnames = [
        "Porosity",
        "Permeability",
        "Depth",
        "Gravity",
        "Viscosity",
        "Temperature",
        "Oil_Saturation",
    ]
    data = [
        [Porosity, Permeability, Depth, Gravity, Viscosity, Temperature, Oil_saturation]
    ]
    my_x = pd.DataFrame(data=data, columns=cnames)
    my_x = sc.transform(my_x)

    # Calling the ML algorithms for their training, plottings, and predictions
    if algorithm == "K Nearest Neighbors(knn)":
        knn = KNeighborsClassifier(n_neighbors=parameter_k_neighbors)
        knn.fit(x_train, y_train)
        training_score = knn.score(x_train, y_train)
        test_score = knn.score(x_test, y_test)

        # Plot of Accuracy vs K values using the training and testing data
        fig, ax = plt.subplots(figsize=(15, 8))
        neighbors = np.arange(1, 10)
        train_accuracy = np.empty(len(neighbors))
        test_accuracy = np.empty(len(neighbors))
        for i, k in enumerate(neighbors):
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(x_train, y_train)
            train_accuracy[i] = knn.score(x_train, y_train)
            test_accuracy[i] = knn.score(x_test, y_test)
        ax.plot(neighbors, test_accuracy, label="Test")
        ax.plot(neighbors, train_accuracy, label="Train")
        plt.legend(fontsize=12)
        ax.set_xlabel("K neighbors", size=14)
        ax.set_ylabel("Accuracy", size=14)
        ax.set_title(
            "Accuracy Vs K neighbors",
            fontname="Times New Roman",
            size=16,
            fontweight="bold",
        )
        plt.show()

        prediction = knn.predict(my_x)
    elif algorithm == "Decision tree":
        tree = DecisionTreeClassifier(max_depth=parameter_decision_tree)
        tree.fit(x_train, y_train)
        training_score = tree.score(x_train, y_train)
        test_score = tree.score(x_test, y_test)

        # Plot of Accuracy vs max depth values using the training and testing data
        fig, ax = plt.subplots(figsize=(15, 8))
        max_depth = np.arange(1, 9)
        train_accuracy = np.empty(len(max_depth))
        test_accuracy = np.empty(len(max_depth))
        for i, r in enumerate(max_depth):
            tree = DecisionTreeClassifier(max_depth=r)
            tree.fit(x_train, y_train)
            train_accuracy[i] = tree.score(x_train, y_train)
            test_accuracy[i] = tree.score(x_test, y_test)
        ax.plot(max_depth, test_accuracy, label="Test")
        ax.plot(max_depth, train_accuracy, label="Train")
        plt.legend(fontsize=12)
        ax.set_xlabel("Max depth", size=14)
        ax.set_ylabel("Accuracy", size=14)
        ax.set_title(
            "Accuracy Vs Max depth",
            fontname="Times New Roman",
            size=16,
            fontweight="bold",
        )
        plt.show()
        prediction = tree.predict(my_x)
    else:
        # Initializing the ANN
        model = Sequential()

        model.add(Dense(8, activation='relu'))  ## Rectified Linear Unit(RELU) chosen as activation function
        model.add(Dense(8, activation='relu'))
        model.add(Dense(5, activation='relu'))
        #model.add(Dense(1))
        # Compiling the ANN

        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=[
            'accuracy'])  # Optimizer is stochastic gradient descent (Adam model). Loss is the Logarithmix loss function in this case (It could be the Ordinary Least squares function)

        # Fitting the ANN to the training set
        history = model.fit(x_train, y_train, batch_size=1, epochs=100)

        # Plotting the accuracy and loss
        print(history.history.keys())
        Max_accuracy = np.max(history.history['acc'])
        Min_loss = np.min(history.history['loss'])
        training_mean_accuracy = np.mean(history.history['acc'])
        training_mean_loss = np.mean(history.history['loss'])

        # summarize history for accuracy
        fig1 = plt.plot(history.history['acc'])
        # plt.plot(history.history['val_acc'])
        plt.title('Training accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.title('Training loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # Predicting the Test set results
        y_test = model.predict(x_test)

    # Model performance information
    st.subheader("2. Model Performance")
    st.markdown("**2.1 Training set**")
    st.write("Accuracy of training set")
    st.info(training_score)
    st.write("---")
    st.markdown("**2.2 Test set**")
    st.write("Accuracy of Test set")
    st.info(test_score)
    st.write("---")
    st.markdown("**2.3 Graphical Performance**")
    st.pyplot(plt)

    # Model prediction information
    st.write("---")
    st.subheader("3. Model Prediction")
    if np.argmax(prediction) == 0:
        st.info("CO2 Injection")
    elif np.argmax(prediction) == 1:
        st.info("Combustion")
    elif np.argmax(prediction) == 2:
        st.info("HC Injection")
    elif np.argmax(prediction) == 3:
        st.info("Polymer")
    elif np.argmax(prediction) == 4:
        st.info("Steam Injection")


tab1, tab2, tab3 = st.tabs(["***Exploratory Data Analysis***", "***Machine Learning***", "***About***"])
with tab1:
    # Exploratory Data Analysis (EDA)
    if st.button("Press to See the Exploratory Data Analysis (EDA)"):
        st.header("**Exploratory Data Analysis (EDA)**")
        st.write("---")
        if upload_file is not None:
            @st.cache_data
            def load_csv():
                data = pd.read_csv(upload_file)
                return data

        df = load_csv()
        # pr = ProfileReport(df)
        st.markdown("**Input Dataframe**")
        st.write(df)
        st.write("---")
        # st.markdown("**EDA Report**")
        # st_profile_report(pr)

    st.write("---")
    st.header("**Geospatial Data**")

    # Load the coordinates of the countries where the EOR projects of this dataset are
    coordinates = {
        "Norway": ([64.5783, 17.8882], 5),
        "Canada": ([56.130366, -106.346771], 38),
        "Usa": ([37.09024, -95.712891], 140),
        "Brazil": ([-23.533773, -46.625290], 8),
        "Egypt": ([26.820553, 30.802498], 1),
        "Germany": ([51.5167, 9.9167], 10),
    }
    # Load the world map
    m = folium.Map(zoom_start=14)
    # Load the markers and popups
    for country, point in coordinates.items():
        folium.Marker(
            point[0], popup="<b>{}: </b> {} EOR Projects".format(country, point[1])
        ).add_to(m)
    folium_static(m)

with tab2:
    # Model Deployment
    if st.button("Model Deployment"):
        if upload_file is not None:
            st.write("---")
            st.subheader("1. Dataset")
            df = pd.read_csv(upload_file)
            df.rename(columns={"Viscocity": "Viscosity"}, inplace=True)
            st.markdown("**1.1 Showing dataset**")
            st.write(df)
            model(df)
        else:
            st.write("Please upload Dataset or Download from the link!")

with tab3:
    st.write("This app is built by vulm. Feel free to contact me via email: lmvu103@gmail.com")

    st.markdown(
        """
    This App consists of implementing an **EOR Screening** for any well by using Machine 
    Learning algorithms.
    This project consists of implementing an EOR Screening by using 
        Machine Learning algorithms. It must be mentioned that the 
        dataset used for training and evaluating these algorithms have a
        size of roughly 200 successful EOR Projects (Rows or 
        observations) from some countries, as well as 7 reservoir 
        parameters, which are the feature or dependent variables. 
        Furthermore, the target variable of this model is a categorical 
        variable, which contains 5 EOR Methods (classes) such as the 
        steam injection method, CO2 injection method, HC injection 
        method, polymer injection method, and combustion in situ 
        method."""
    )


    # Adding of a mp4 video
    st.markdown(
        """
        **The Phases of Oil Recovery**
        """
    )
    video = open("Oil Phases Recovery.mp4", "rb")
    st.video(video)
    st.markdown("<span style=“background-color:#121922”>", unsafe_allow_html=True)
    st.markdown(
        "Energy & Environmental Research Center. (2014, April). The Phases of Oil Recovery"
    )

