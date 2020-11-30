# AuthorClassification
Information gain (ID3) for feature selection (words) from text and Machine Learning to classifiy the author from selected features  
by Merlin Carson  

# Requirements
* Python3
* Sklearn

I first parsed all the text using REGEX to capture only alpha chars from all words > length 2, thus removing all punctuation, numeric values and words containing only 1 char. I converted this list of words to a set to remove all duplicates, this is the dictionary of words. I then parse each text paragraph, creating a list of words using the same constraints as stated above for creating the dictionary. I iterate through the dictionary for each paragraph, checking for each word in the text, setting the feature to true if it's in the paragraph and setting it false if it is not. The text's name and paragraph number along with the class (author) is concatenated with the feature vector for the paragraph.

After the feature vectors are created for all parapraphs in all the texts, the gain is calculated for each word split in the dictionary. The feature set is then pruned to only keep the words that have the highest gains. This information is then printed to standard output and saved to file for use by a machine learner.

I then used the sklearn API to split the data into k-folds and process them through a machine learner. I initially used RandomForest and XGBoost since these learners work well for classification using binary features. XGBoost performed slightly better than the Random Forest as expected, ~90.42% average for 10 folds with one fold at ~96.55%. To compare my results with the professor, who used his own implementation of Naive Bayes, I also added sklearn's Bernoulli Naive Bayes learner as an option to the training script. While the average results were slightly lower than XGBoost, ~87.69%, the highest fold reached ~99.08%, noticably higher than XGBoost's best fold. 

Overall XGBoost is a more sophistacated algorithm and on average performed better than Random Forest and Naive Bayes. However, Naive Bayes performed admirable well for such a simple algorithm and is significantly faster to train. 

- features.py: creates a pruned feature set of words from a directory of .txt files based on the information gain achieved by splitting on each word.
- info-gain.py: helper functions for calculating entropy and gain, and pruning the feature set to the highest n-gains.
- train.py: Machine Learner training script for the feature set outputted by features.py. This script performs k-fold cross-validation on the optional learner ('NB' = Naive Bayes, 'RF' = Random Forest, 'XGB' = XGBoost) 
