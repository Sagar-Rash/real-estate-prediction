# Real-Estate-Prediction
App to predict house sale price given different inputs including year built, number of bedrooms, etc.

## Features
Streamlit application with input form.

Prediction of house sale price.

Accessible via Streamlit Community Cloud: https://real-estate-prediction-sr.streamlit.app/

# Dataset
The dataset is based on a real estate dataset. The features used in this model are

* Price (target)
* Year Sold
* Property Tax (Per month)
* Insurance
* Beds (Number of bedrooms)
* Baths (Number of bathrooms)
* Sqft (Area of house)
* Year Built
* Lot Size (sqft)
* Basement (Boolean, True if has a basement)
* Popular (Calculated by app, if house has 2 bedrooms and 2 bathrooms)
* Recession (Calculated by app, if Year Sold was between 2008 and 2013)
* Property Age (Calculated by app, Year Sold - Year Built)
* Property type Bunglow (Boolean, if false then property type is Condo)


# Technologies Used
Streamlit: For the application.

Scikit-learn: For model training and evaluation.

Pandas and NumPy: For data preprocessing and manipulation.

Matplotlib and Seaborn: For exploratory data analysis and visualization.

# Model
The predictive model was trained using a Mall Customer dataset which has multiple copies available on Kaggle.

# Installation (to local computer):
If you want to run the application locally, follow these steps:

1. Clone the repository:
git clone https://github.com/your-username/real-estate-prediction.git
cd real-estate-prediction
2. Create and activate a virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows, use `env\\Scripts\\activate`
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Run the Streamlit application:
```bash
streamlit run RE_app_streamlit.py
```
# Note that some filepaths may need to be changed based on the local system (Windows vs. Linux) (backslash vs. forward slash)
