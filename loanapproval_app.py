{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "23dba9fa-87a8-45a5-b018-5e8beab806eb",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-02-05 11:51:26.203 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\Gargi\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2025-02-05 11:51:26.203 Session state does not function when running a script without `streamlit run`\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import pickle\n",
    "import numpy as np\n",
    "\n",
    "# Load the trained model and scaler\n",
    "with open('stacking_model.pkl', 'rb') as model_file:\n",
    "    model = pickle.load(model_file)\n",
    "\n",
    "with open('scaler.pkl', 'rb') as scaler_file:\n",
    "    scaler = pickle.load(scaler_file)\n",
    "\n",
    "# Streamlit UI\n",
    "st.title(\"Loan Approval Prediction\")\n",
    "\n",
    "# User Inputs\n",
    "Income = st.number_input(\"Income\")\n",
    "CCAvg = st.number_input(\"CCAvg\")\n",
    "Mortgage = st.number_input(\"Mortgage\")\n",
    "Education = st.selectbox(\"Education\", [1, 2, 3])\n",
    "Family = st.number_input(\"Family\", min_value=1, max_value=4, step=1)\n",
    "Securities_Account = st.selectbox(\"Securities Account\", [0, 1])\n",
    "CD_Account = st.selectbox(\"CD Account\", [0, 1])\n",
    "Online = st.selectbox(\"Online\", [0, 1])\n",
    "CreditCard = st.selectbox(\"Credit Card\", [0, 1])\n",
    "\n",
    "# Predict button\n",
    "if st.button(\"Predict\"):\n",
    "    try:\n",
    "        # Prepare input data\n",
    "        data = np.array([[Income, CCAvg, Mortgage, Education, Family, Securities_Account, CD_Account, Online, CreditCard]])\n",
    "        scaled_data = scaler.transform(data)\n",
    "        prediction = model.predict(scaled_data)[0]\n",
    "        result = \"Approved\" if prediction == 1 else \"Not Approved\"\n",
    "        st.success(f\"Loan Approval Prediction: {result}\")\n",
    "    except Exception as e:\n",
    "        st.error(f\"Error: {e}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "147525bb-6b6b-4016-b0ba-d1eeba71616d",
   "metadata": {},
   "outputs": [],
   "source": [
    "!streamlit run loanapproval_app.py\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "29a125e9-b3bd-4f3a-a238-a2e8f2eb13c6",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
