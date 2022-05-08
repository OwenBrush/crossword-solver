# Crossword Solving Model
Machine learning model trained to solve crossword questions

### Goal:
This is a proof of concept, focused on getting a baseline indication of the potential that this aproach has for solving crossword clues.

### Datasets and tools:
- https://xd.saul.pw/data/#download (xd-clues.zip)
- gensim word2vec pre-trained models:
  - glove-wiki-gigaword-100
  - glove-twitter-25
  - word2vec-google-news-300
- Scikit/Sklearn libraries

# How it Works:

![Flowchart of model operations](/images/model_diagram.png?raw=true "Number of unique clues and answers in cleaned dataset" )

### Step 1) Eight features are built from the given clue:
    -  Wiki PCA           X# of Scalars   (Word vectors from the wiki trained model are reduced to a set of scalars)
    -  Twitter PCA        X# of Scalars   (Word vectors from the twitter trained model are reduced to a set of scalars)
    -  Google PCA         X# of Scalars   (Word vectors from the google trained model are reducedto a set of scalars)
    -  Pressence of Noun  Boolean         (Determined by clue containing multiple capital letters)
    -  Fill the Blank     Boolean         (Determined by clue containing an underscore)
    -  Number of Words    Scalar          (Counts number of words in clue, threshold for minimum character count can be given during training)
    -  Length of Answer   Scalar          (The total number of characters the answer is expected to be)
    -  Known characters   String          (Characters of the answer that are already known. This is used filter the vocabulary being used)
    
### Step 2) Cosine Similarities are predicted:
  -  Using all of the features, except the known characters, the model then makes three predictions to determine what the likely cosine similarity between the given clue and expected answer are, for each of the pretrained models being used.

### Step 3) Answer is predicted
  - Each model parses it's vocabulary to look at only the words that will fit the answer and selects a list of 10 words which have the closest cosine similarity to the clue as the model has predicted it should have.
  - Each word is given a confidence score based on how close it's cosine similarity is to what has been predicted.
  - The selection of each model is then pooled together, and the confidence score of any identical words are added together.  The the word with the highest score is selected as the predicted answer.


# Performance:

The results of this first iteration of the model are very promising, showing that it consistently performs signifigantly better than random selection while also having a lot of room for future improvement. 

!["Overview of model Accuracy](/images/accuracy_overview.png?raw=true "Overview of model Accuracy" )


- From the total dataset, a random sampling of 60,000 was made with 15,000 being used for testing and 45,000 being used for training.
- The vocabulary limit indicates the limitation of the vocabulary of the models used, meaning that it would be impossible for the models used in this trial to predict higher than this threshold.  Models trained on google, wiki, and twitter were selected for their extensive vocabularies but even still, crossword answers are often creatively made up words, or compounded words that would not be used in normal speach.
- It is worth noting that the random selection for no known characters returned only a single correct answer so it's accuracy may actually be lower than indicated.

There are two possible uses for this model with which these results can be assessed:

#### 1) Automated solving of crossword puzzles:
  For this task, the model is not yet ready.  With no known characters the model can only predict a correct answer 1 in 2000 times, making the task of automatically solving a puzzle with it statistically impossible. However, the improvement that it shows above simple random selection does show that it could be useful for this task given enough refinement and optimization and/or by being used in conjuction with another aproach.
  
#### 2) A tool to assist solving of crossword puzzles:
  For this task, the model would be useful even in it's current un-optimized state.   With the top five choices containing a correct answer roughly 1 in 4 times with a handful of known characters, it would be a very useful tool in helping a human to solve a puzzle by presenting a list of possibilities, and these results can likely be greatly improved with further optimization.

  
  
# Areas for improvement:

### Language Models:
The Models used were chosen for the size of their vocabularies and a desire to create a robust model that would be able to predict crossword cluse from a different source than the data being used for training.  A downside to this, however, is that there is an immense amount of noise contained within these models with words that would never likely be seen in a crossword.  This has an effect on both the model's accuracy and also the time it takes to make a prediction, as cosine similarities need to be considered for every possible word before a selection can be made.

This aspect of the model could be refined by using a different ensemble of language models.  I think these broad language models do have a place in this aproach, but it might also be worth including language models that have been trained specifically with crosswords in mind so that it's vocabulary would be a better fit for the task at hand.

### Cosine Prediction:
Currently, all cosine predictions are being done with default Random Forest Regressor models from the sci-kit library.  The accuracy of the cosine prediction could likely be greatly imrpvoed through tweaking of hyper-parameters, experimentation of different PCA sizes, and exploring the use of different prediction models as aswell as tailoring each of the elements individually for each language model.   


# Exploratory Data Analysis on the data used:


![Percentage of clues with different special characters](/images/punctuation_percents.png?raw=true "Percentage of clues that contain each for of punctuation (before cleaning)")

- The presence of an underscore indicates a "fill in the blank" type of clue which is different in nature from other clues
  - these clues could also be diverted to a model trained for this purpose
- Quotation marks or capital letters frequently indicate the involvement of a noun
- ! and $ symbols can convey important meaning and are worth keeping
- Most other punctuation is just noise

![3141343 unique pairings, 2650153 unique clues, 315686 unique answers](/images/number_of_clues_and_answers.png?raw=true "Number of unique clues and answers in cleaned dataset" )

- Of the 6 million + samples in the full dataset, this graph shows that roughly half of them are duplicate answer-clue pairings, and that there is a relatively small number of possible answers spread across those clues.

![1 occurance: 50.1%, 2-10 occurances: 33%, 11-100 occurances: 13.1%, 100+ occurances: 3.8%](/images/answer_frequency.png?raw=true "Frequency of repeated answers." )

- Of the ~315,000 unique answers aproximately half of them, occur only a single time in the data set, and 3.8% of them occur more than 100 times. Indicating that the majority of answers appear only one or a few times, while a small number of them will occur much more frequently in the dataset. 
- This also indicates that the model will need to be robust, as even working within the confines of the data set it will need to be predicting words that it has not seen before.

![glove-wiki-gigaword: 85.59%, glove-twitter-25: 85.17%, word2vec-google-news: 77.26%](/images/vocabulary_percentages.png?raw=true "Percentage of answers contained in vocabularies of pre-trained word2vec models." )

- All pretrained models tested have 75-85% of the answers in their vocabulary
  - There are a fair number of answers for crosswords that are compound words, or creatively made up words that are unlikely to be caught by any model that doesn't include that particular crossword clue in it's training.
  - An additional model could be made for generating compound words to increase the robustness of the predictions, but otherwise these types of anwers will not be able to be predicted 



