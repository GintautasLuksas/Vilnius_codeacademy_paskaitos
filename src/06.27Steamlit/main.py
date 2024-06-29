#1
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Pagrindinė antraštė")


st.subheader("Paantraštė")

st.write("Hello, World!")


#


# Įkelti CSV failą
uploaded_file = st.file_uploader("Įkelkite CSV failą", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Įkeltas duomenų failas:")
    st.write(df)

#3

# Įkelti CSV failą
uploaded_file = st.file_uploader("Įkelkite CSV failą", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Apibendrinta statistika kiekvienam skaitiniam stulpeliui:")
    st.write(df.describe())

#4

# Įkelti CSV failą
uploaded_file = st.file_uploader("Įkelkite CSV failą", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

    # Pasirinkti stulpelius
    x_col = st.selectbox("Pasirinkite X ašies stulpelį", numeric_columns)
    y_col = st.selectbox("Pasirinkite Y ašies stulpelį", numeric_columns)

    if x_col and y_col:
        st.write(f"Linijinis grafikas: {x_col} vs {y_col}")
        fig, ax = plt.subplots()
        ax.plot(df[x_col], df[y_col])
        st.pyplot(fig)
#5


# Įkelti CSV failą
uploaded_file = st.file_uploader("Įkelkite CSV failą", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

    # Pasirinkti stulpelį
    selected_column = st.selectbox("Pasirinkite stulpelį", numeric_columns)

    if selected_column:
        st.write(f"Histograma stulpeliui: {selected_column}")
        fig, ax = plt.subplots()
        ax.hist(df[selected_column], bins=20)
        st.pyplot(fig)
#6

# Įkelti CSV failą
uploaded_file = st.file_uploader("Įkelkite CSV failą", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    corr = df.corr()

    st.write("Koreliacijos šilumos žemėlapis:")
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
#7

# Įkelti CSV failą
uploaded_file = st.file_uploader("Įkelkite CSV failą", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()

    # Pasirinkti stulpelius
    numeric_col = st.selectbox("Pasirinkite skaitinį stulpelį", numeric_columns)
    categorical_col = st.selectbox("Pasirinkite kategorinį stulpelį", categorical_columns)

    if numeric_col and categorical_col:
        st.write(f"Dėžės diagrama: {numeric_col} pagal {categorical_col} kategorijas")
        fig, ax = plt.subplots()
        sns.boxplot(x=df[categorical_col], y=df[numeric_col], ax=ax)
        st.pyplot(fig)

