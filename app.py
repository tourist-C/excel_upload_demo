import streamlit as st
import joblib
import pandas as pd

st.title("Sample App for Random Forest Model for Diabetes Regression")
model = joblib.load("diabetes_rf_model.joblib")

st.markdown("#### Step 1) Download sample input")
# template file for download
df_sample = pd.read_csv("sample.csv", index_col=0)
csv = df_sample.to_csv(index=False)
st.download_button(
    label="Download Sample Input",
    data=csv,
    file_name="sample.csv",
    mime="text/csv"
)

st.markdown("#### Step 2) Add your data")


st.markdown("#### Step 3) Upload")

# Upload file
file = st.file_uploader("Upload CSV file", type=["csv"])

# Check if file is uploaded
if file is not None:
    # Read the file
    df = pd.read_csv(file, index_col=0)
    
    # Display the DataFrame
    st.dataframe(df)

    st.success("Job done! Prediction appended to the last column of the table")

    yhat = model.predict(df)

    series = pd.Series(yhat, name="Prediction")

    df_out = pd.concat((df, series), axis=1)

    st.markdown("#### Step 4) View Results")
    st.write(df_out)

    # Add a download button
    csv = df_out.to_csv(index=False)
    st.download_button(
        label="Download",
        data=csv,
        file_name="result.csv",
        mime="text/csv"
    )


