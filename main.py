import streamlit as st
# EDA Pkgs
import pandas as pd 
import numpy as np 
# Data Viz Pkg
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")
import seaborn as sns 
from lazypredict.Supervised import LazyRegressor,LazyClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.datasets import load_diabetes, load_boston
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='The Machine Learning Algorithm Comparison App',
    layout='wide')
st.markdown("<h1 font='IBM Flex Sans'; style='text-align: center; color: black; font-size:29px;'> Automated Machine Learning </h1>", unsafe_allow_html=True)
st.markdown("<p font='IBM Flex Sans'; style='text-align: center; color: blue; font-size:29px;'> Auto ML applications built on Streamlit</p>", unsafe_allow_html=True)
st.markdown("""
<hr style="width:100%;height:3px;background-color:gray;border-width:10">
""", unsafe_allow_html=True)
with open('style/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)  
def page_2():
	"""Semi Automated ML App with Streamlit """

	activities = ["EDA","Plots"]	
	choice = st.sidebar.selectbox("Select Activities",activities)

	if choice == 'EDA':
		st.subheader("Exploratory Data Analysis")

		data = st.file_uploader("Upload a Dataset", type=["csv", "txt"])
		if data is not None:
			df = pd.read_csv(data)
			st.dataframe(df.head())
			

			if st.checkbox("Show Shape"):
				st.write(df.shape)

			if st.checkbox("Show Columns"):
				all_columns = df.columns.to_list()
				st.write(all_columns)

			if st.checkbox("Summary"):
				st.write(df.describe())

			

			if st.checkbox("Show Value Counts"):
				st.write(df.iloc[:,-1].value_counts())

			if st.checkbox("Correlation Plot(Matplotlib)"):
				plt.matshow(df.corr())
				st.pyplot()

			if st.checkbox("Correlation Plot(Seaborn)"):
				st.write(sns.heatmap(df.corr(),annot=True))
				st.pyplot()


			if st.checkbox("Pie Plot"):
				all_columns = df.columns.to_list()
				column_to_plot = st.selectbox("Select 1 Column",all_columns)
				pie_plot = df[column_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
				st.write(pie_plot)
				st.pyplot()



	elif choice == 'Plots':
		st.subheader("Data Visualization")
		data = st.file_uploader("Upload a Dataset", type=["csv", "txt", "xlsx"])
		if data is not None:
			df = pd.read_csv(data)
			st.dataframe(df.head())


			if st.checkbox("Show Value Counts"):
				st.write(df.iloc[:,-1].value_counts().plot(kind='bar'))
				st.pyplot()
		
			# Customizable Plot

			all_columns_names = df.columns.tolist()
			type_of_plot = st.selectbox("Select Type of Plot",["area","bar","line","hist","box","kde"])
			selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)

			if st.button("Generate Plot"):
				st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))

				# Plot By Streamlit
				if type_of_plot == 'area':
					cust_data = df[selected_columns_names]
					st.area_chart(cust_data)

				elif type_of_plot == 'bar':
					cust_data = df[selected_columns_names]
					st.bar_chart(cust_data)

				elif type_of_plot == 'line':
					cust_data = df[selected_columns_names]
					st.line_chart(cust_data)

				# Custom Plot 
				elif type_of_plot:
					cust_plot= df[selected_columns_names].plot(kind=type_of_plot)
					st.write(cust_plot)
					st.pyplot()    

def page_1(df):
 
    df = df.loc[:100] 
    X = df.iloc[:,:-1]
    Y = df.iloc[:,-1] 

    st.subheader('**1.2. Dataset dimension**')
    st.subheader('X')
    st.info(X.shape)
    st.subheader('Y')
    st.info(Y.shape)

    st.subheader('**1.3. Variable details**:')
    st.subheader('X variable (first 20 are shown)')
    st.info(list(X.columns[:20]))
    st.subheader('Y variable')
    st.info(Y.name)

    # Build lazy model
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size = split_size,random_state = seed_number)
    reg = LazyRegressor(verbose=0,ignore_warnings=False, custom_metric=None)
    models_train,predictions_train = reg.fit(X_train, X_train, Y_train, Y_train)
    models_test,predictions_test = reg.fit(X_train, X_test, Y_train, Y_test)

    st.subheader('2. Table of Model Performance')

    st.subheader('Training set')
    st.write(predictions_train)
    st.markdown(filedownload(predictions_train,'training.csv'), unsafe_allow_html=True)

    st.subheader('Test set')
    st.write(predictions_test)
    st.markdown(filedownload(predictions_test,'test.csv'), unsafe_allow_html=True)

    st.subheader('3. Plot of Model Performance (Test set)')


    with st.subheader('**R-squared**'):
        # Tall
        predictions_test["R-Squared"] = [0 if i < 0 else i for i in predictions_test["R-Squared"] ]
        plt.figure(figsize=(3, 9))
        sns.set_theme(style="whitegrid")
        ax1 = sns.barplot(y=predictions_test.index, x="R-Squared", data=predictions_test)
        ax1.set(xlim=(0, 1))
    st.markdown(imagedownload(plt,'plot-r2-tall.pdf'), unsafe_allow_html=True)
        # Wide
    plt.figure(figsize=(9, 3))
    sns.set_theme(style="whitegrid")
    ax1 = sns.barplot(x=predictions_test.index, y="R-Squared", data=predictions_test)
    ax1.set(ylim=(0, 1))
    plt.xticks(rotation=90)
    st.pyplot(plt)
    st.markdown(imagedownload(plt,'plot-r2-wide.pdf'), unsafe_allow_html=True)

    with st.subheader('**RMSE (capped at 50)**'):
        # Tall
        predictions_test["RMSE"] = [50 if i >50  else i for i in predictions_test["RMSE"] ]
        plt.figure(figsize=(3, 9))
        sns.set_theme(style="whitegrid")
        ax2 = sns.barplot(y=predictions_test.index, x="RMSE", data=predictions_test)
    st.markdown(imagedownload(plt,'plot-Accuracy-tall.pdf'), unsafe_allow_html=True)
        # Wide
    plt.figure(figsize=(9, 3))
    sns.set_theme(style="whitegrid")
    ax2 = sns.barplot(x=predictions_test.index, y="RMSE", data=predictions_test)
    plt.xticks(rotation=90)
    st.pyplot(plt)
    st.markdown(imagedownload(plt,'plot-RMSE-wide.pdf'), unsafe_allow_html=True)

    with st.subheader('**Calculation time**'):
        # Tall
        predictions_test["Time Taken"] = [0 if i < 0 else i for i in predictions_test["Time Taken"] ]
        plt.figure(figsize=(3, 9))
        sns.set_theme(style="whitegrid")
        ax3 = sns.barplot(y=predictions_test.index, x="Time Taken", data=predictions_test)
    st.markdown(imagedownload(plt,'plot-calculation-time-tall.pdf'), unsafe_allow_html=True)
        # Wide
    plt.figure(figsize=(9, 3))
    sns.set_theme(style="whitegrid")
    ax3 = sns.barplot(x=predictions_test.index, y="Time Taken", data=predictions_test)
    plt.xticks(rotation=90)
    st.pyplot(plt)
    st.markdown(imagedownload(plt,'plot-calculation-time-wide.pdf'), unsafe_allow_html=True)
def filedownload(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href

def imagedownload(plt, filename):
    s = io.BytesIO()
    plt.savefig(s, format='pdf', bbox_inches='tight')
    plt.close()
    b64 = base64.b64encode(s.getvalue()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:image/png;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href

#---------------------------------#
st.title("Multi-Page Streamlit App")
menu = ["clean_dataset 1", "data_preprocessing 2", "data_visualization 3","ML Model Selector 4","Automl Regression 5","Automl classification 6"]
choice = st.sidebar.selectbox("Select a page", menu)

#---------------------------------#
# Sidebar - Collects user input features into dataframe
if choice == "Automl Regression 5":
    with st.sidebar.header('1. Upload your CSV data'):
        uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    # Sidebar - Specify parameter settings
    with st.sidebar.header('2. Set Parameters'):
        split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
        seed_number = st.sidebar.slider('Set the random seed number', 1, 100, 42, 1)
        st.subheader('1. Dataset')
        st.subheader(" upload your file ")

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
elif choice =="Automl classification 6":
    with st.sidebar.header('1. Upload your CSV data'):
        uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    # Sidebar - Specify parameter settings
    with st.sidebar.header('2. Set Parameters'):
        split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
        seed_number = st.sidebar.slider('Set the random seed number', 1, 100, 42, 1)

    #---------------------------------#
    # Main panel

    # Displays the dataset
    st.subheader('1. Dataset')
    st.subheader(" upload your file ")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

def page_3():
    
    st.title("Data Cleaning App")

    uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "txt", "xlsx"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file,index_col=0,na_values=['?','??','????','???'])
        st.write("Shape of the dataset:", data.shape)
        st.write("First 5 rows of the dataset:", data.head())

        # Missing values
        st.header("Handling Missing Values")
        if st.checkbox("Show Missing Values"):
            st.write(data.isnull().sum())

        # Fill missing values with mean
        if st.checkbox("Fill Missing Values with Mean"):
            mean1 = data.mean()
            data.fillna(mean1, inplace=True)
            st.write("Missing values filled with mean")
            st.write(data.isnull().sum())

        # Fill missing values with median
        if st.checkbox("Fill Missing Values with Median"):
            median1 = data.median()
            data.fillna(median1, inplace=True)
            st.write("Missing values filled with median")
            st.write(data.isnull().sum())

        # Fill missing values with mode
        if st.checkbox("Fill Missing Values with Mode"):
            mode1 = data.mode().iloc[0]
            data.fillna(mode1, inplace=True)
            st.write("Missing values filled with mode")
            st.write(data.isnull().sum())

        # Download cleaned data
        if st.checkbox("Download Cleaned Dataset"):
            score_model =data.to_csv(index=False).encode('utf-8')
            if st.download_button(label='download',data=score_model, mime='text/csv' ,file_name= 'cleaned_data.csv'):
                st.success('done')

def page_4():
    from sklearn.preprocessing import LabelEncoder

    # Streamlit app
    st.title("Data Encoding App")

    # Upload data
    data_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if data_file is not None:
        data = pd.read_csv(data_file)

        # Show raw data
        st.subheader("Raw Data")
        st.write(data)

        # Encode selected columns
        columns = data.columns
        selected_columns = st.multiselect("Select columns to encode", columns)
        if selected_columns:
            for column in selected_columns:
                if data[column].dtype == 'object':
                    encoder = LabelEncoder()
                    data[column] = encoder.fit_transform(data[column])
                    if len(encoder.classes_) == 2:
                        data[column] = data[column].astype('float')
                else:
                    st.write(f"{column} is not a string column")

            # Show encoded data
            st.subheader("Encoded Data")
            st.write(data)
            score_model =data.to_csv(index=False).encode('utf-8')
            if st.download_button(label='download',data=score_model, mime='text/csv' ,file_name= 'mlt_data.csv'):
                st.success('done')

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from statsmodels.tsa.arima.model import ARIMA
def train_and_evaluate_model(dataset, target_column, dataset_type):
    X = dataset.drop(columns=[target_column])
    y = dataset[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if dataset_type == 'classifier':
        models = [LogisticRegression(), RandomForestClassifier(), SVC()]
        best_model = None
        best_accuracy = 0
        for model in models:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            if accuracy > best_accuracy:
                best_model = model
                best_accuracy = accuracy
        return best_model, best_accuracy

    elif dataset_type == 'regressor':
        models = [LinearRegression(), RandomForestRegressor(), SVR()]
        best_model = None
        best_rmse = float('inf')
        for model in models:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            if rmse < best_rmse:
                best_model = model
                best_rmse = rmse
        return best_model, 1 - (best_rmse / y_test.mean())

    elif dataset_type == 'timeseries':
        model = ARIMA(y_train, order=(1, 1, 1)).fit()
        y_pred = model.forecast(len(y_test))
        return model, r2_score(y_test, y_pred)

    # Define the Streamlit app
def page_5():
    st.title("ML Model Selector")
    st.write("Upload a CSV file to train and evaluate the best ML model for your dataset.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        dataset = pd.read_csv(uploaded_file)
        target_column = st.selectbox("Select the target column", options=dataset.columns)
        dataset_type = None
        if dataset[target_column].dtype == 'object':
            dataset_type = 'classifier'
        elif dataset[target_column].dtype in ['int64', 'float64']:
            if dataset[target_column].nunique() <= 10:
                dataset_type = 'classifier'
            else:
                dataset_type = 'regressor'
        elif dataset[target_column].dtype == 'datetime64[ns]':
            dataset_type = 'timeseries'
        if dataset_type is None:
            st.subheader("Invalid target column type. Must be one of: object, int64, float64, datetime64[ns]")
        else:
            st.subheader(f"Detected dataset type: {dataset_type}")
            best_model, accuracy = train_and_evaluate_model(dataset, target_column, dataset_type)
            st.subheader(f"Best model: {type(best_model).__name__}")
            st.subheader(f"Accuracy: {accuracy}")


def page_6(df):
 
    df = df.loc[:100] 
    X = df.iloc[:,:-1]
    Y = df.iloc[:,-1] 

    st.subheader('**1.2. Dataset dimension**')
    st.subheader('X')
    st.info(X.shape)
    st.subheader('Y')
    st.info(Y.shape)

    st.subheader('**1.3. Variable details**:')
    st.subheader('X variable (first 20 are shown)')
    st.info(list(X.columns[:20]))
    st.subheader('Y variable')
    st.info(Y.name)

    # Build lazy model
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size = split_size,random_state = seed_number)
    reg = LazyClassifier(verbose=0,ignore_warnings=False, custom_metric=None)
    models_train,predictions_train = reg.fit(X_train, X_train, Y_train, Y_train)
    models_test,predictions_test = reg.fit(X_train, X_test, Y_train, Y_test)

    st.subheader('2. Table of Model Performance')

    st.subheader('Training set')
    st.write(predictions_train)
    st.markdown(filedownload(predictions_train,'training.csv'), unsafe_allow_html=True)

    st.subheader('Test set')
    st.write(predictions_test)
    st.markdown(filedownload(predictions_test,'test.csv'), unsafe_allow_html=True)

    st.subheader('3. Plot of Model Performance (Test set)')


    # with st.subheader('**R-squared**'):
    #     # Tall
    #     predictions_test["R-Squared"] = [0 if i < 0 else i for i in predictions_test["R-Squared"] ]
    #     plt.figure(figsize=(3, 9))
    #     sns.set_theme(style="whitegrid")
    #     ax1 = sns.barplot(y=predictions_test.index, x="R-Squared", data=predictions_test)
    #     ax1.set(xlim=(0, 1))
    # st.markdown(imagedownload(plt,'plot-r2-tall.pdf'), unsafe_allow_html=True)
    #     # Wide
    # plt.figure(figsize=(9, 3))
    # sns.set_theme(style="whitegrid")
    # ax1 = sns.barplot(x=predictions_test.index, y="R-Squared", data=predictions_test)
    # ax1.set(ylim=(0, 1))
    # plt.xticks(rotation=90)
    # st.pyplot(plt)
    # st.markdown(imagedownload(plt,'plot-r2-wide.pdf'), unsafe_allow_html=True)

    with st.subheader('**Accuracy (capped at 50)**'):
        # Tall
        predictions_test["Accuracy"] = [50 if i > 50 else i for i in predictions_test["Accuracy"] ]
        st.write(predictions_test.nlargest(5,['Time Taken']))
        plt.figure(figsize=(3, 9))
        sns.set_theme(style="whitegrid")
        ax2 = sns.barplot(y=predictions_test.index, x="Accuracy", data=predictions_test)
    st.markdown(imagedownload(plt,'plot-rmse-tall.pdf'), unsafe_allow_html=True)
        # Wide
    plt.figure(figsize=(9, 3))
    sns.set_theme(style="whitegrid")
    ax2 = sns.barplot(x=predictions_test.index, y="Accuracy", data=predictions_test)
    plt.xticks(rotation=90)
    st.pyplot(plt)
    st.markdown(imagedownload(plt,'plot-Accuracy-wide.pdf'), unsafe_allow_html=True)

    with st.subheader('**Calculation time**'):
        # Tall
        predictions_test["Time Taken"] = [0 if i < 0 else i for i in predictions_test["Time Taken"] ]
        st.write(predictions_test.nlargest(5,['Time Taken']))
        plt.figure(figsize=(3, 9))
        sns.set_theme(style="whitegrid")
        ax3 = sns.barplot(y=predictions_test.index, x="Time Taken", data=predictions_test)
    st.markdown(imagedownload(plt,'plot-calculation-time-tall.pdf'), unsafe_allow_html=True)
        # Wide
    plt.figure(figsize=(9, 3))
    sns.set_theme(style="whitegrid")
    ax3 = sns.barplot(x=predictions_test.index, y="Time Taken", data=predictions_test)
    plt.xticks(rotation=90)
    st.pyplot(plt)
    st.markdown(imagedownload(plt,'plot-calculation-time-wide.pdf'), unsafe_allow_html=True)
def filedownload(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href

def imagedownload(plt, filename):
    s = io.BytesIO()
    plt.savefig(s, format='pdf', bbox_inches='tight')
    plt.close()
    b64 = base64.b64encode(s.getvalue()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:image/png;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href

def main():

    if choice == "data_visualization 3":
        page_2()
    elif choice == "data_preprocessing 2":
        page_4()
    elif choice == "clean_dataset 1":
        page_3()
    elif choice =="Automl Regression 5":
        try:
            page_1(df)
        except:
            st.write("upload the csv file....")
    elif choice =="Automl classification 6":
        try:
            page_6(df)
        except:
            st.write("please upload the csv file")
    else:
        page_5()
if __name__ == "__main__":
    main()
