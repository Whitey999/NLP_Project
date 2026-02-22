{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "70284e61-bc95-4b5b-9904-5e5e81fdb81e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Batch Prediction Interface\n",
    "def batch_predict(file_path):\n",
    "    \"\"\"\n",
    "    Predict spam/ham for multiple messages from a CSV file\n",
    "    \"\"\"\n",
    "    # Load messages\n",
    "    batch_df = pd.read_csv(file_path)\n",
    "    \n",
    "    if 'message' not in batch_df.columns:\n",
    "        print(\"Error: CSV file must contain a 'message' column\")\n",
    "        return\n",
    "    \n",
    "    # Make predictions\n",
    "    predictions = []\n",
    "    confidences = []\n",
    "    \n",
    "    for msg in batch_df['message']:\n",
    "        cleaned = clean_text(msg)\n",
    "        msg_tfidf = tfidf_vectorizer.transform([cleaned])\n",
    "        features = extract_features(msg)\n",
    "        features_array = np.array([[features[col] for col in feature_columns]])\n",
    "        final_features = hstack([msg_tfidf, features_array])\n",
    "        \n",
    "        pred = lr_model.predict(final_features)[0]\n",
    "        prob = lr_model.predict_proba(final_features)[0]\n",
    "        \n",
    "        predictions.append('spam' if pred == 1 else 'ham')\n",
    "        confidences.append(prob[pred])\n",
    "    \n",
    "    # Add results to dataframe\n",
    "    batch_df['prediction'] = predictions\n",
    "    batch_df['confidence'] = confidences\n",
    "    \n",
    "    # Save results\n",
    "    output_file = 'predictions_' + file_path\n",
    "    batch_df.to_csv(output_file, index=False)\n",
    "    print(f\"Predictions saved to {output_file}\")\n",
    "    \n",
    "    # Summary\n",
    "    print(f\"\\nSummary:\")\n",
    "    print(f\"Total messages: {len(batch_df)}\")\n",
    "    print(f\"Spam detected: {sum(predictions == 'spam')}\")\n",
    "    print(f\"Ham detected: {sum(predictions == 'ham')}\")\n",
    "    \n",
    "    return batch_df\n",
    "\n",
    "# Example usage\n",
    "# results = batch_predict('new_messages.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "72e8beba-71cf-48cb-9d98-068c0e77e62e",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
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
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
