# Crossword Solving Model
Machine learning model trained to solve crossword questions

### Goals:
- Practice with natural language processing 
- Create a model with an interesting and understandable application
- Provide clear data visualization to showcase the methodology and results of the model

### What it will do:
- Given a clue and a number of letters (with some optionally filled in) the model will return the most likely answers.

### Ideas for Future Improvement:
- Solve an entire puzzle, when given a list of clues and intersecting character positions
- Use image recognition to solve a puzzle from an image or photo of a crossword

### Datasets and tools:
- https://xd.saul.pw/data/#download (xd-clues.zip)
- gensim word2vec pre-trained models:
  - glove-wiki-gigaword-100
  - glove-twitter-25
  - word2vec-google-news-300
- Scikit/Sklearn libraries

# Results:

### *Coming Soon...*

# Model Map:

![Flowchart of model operations](/images/model_diagram.png?raw=true "Number of unique clues and answers in cleaned dataset" )

### Step 1) Eight features are built from the given clue:
    -  Wiki Cluster       Categorical (K-means clusters are created for clues to be fitted to, using the word vectors of the pretrained Wiki model)
    -  Twitter Cluster    Categorical (K-means clusters are created for clues to be fitted to, using the word vectors of the pretrained Twitter model)
    -  Google Cluster     Categorical (K-means clusters are created for clues to be fitted to, using the word vectors of the pretrained Google model)
    -  Pressence of Noun  Boolean     (Determined by clue containing multiple capital letters
    -  Fill the Blank     Boolean     (Determined by clue containing an underscore)
    -  Number of Words    Scalar      (Counts number of words in clue, threshold for minimum character count can be given during training)
    -  Length of Answer   Scalar      (The total number of characters the answer is expected to fill)
    -  Known characters   String      (This is not used for predicting, but to filter the vocabulary being used)
    
### Step 2) Cosign Similarities are predicted:
  -  Using all of the features, except the known characters, the model then makes three predictions to determine what the likely cosign similarity between the given clue and expected answer are, for each of the pretrained models being used.

### Step 3) Answer is predicted
  - The model then parses then selects a list of all words contained in the vocabularies of the pretrained models that fit the expect length and known characters of the answer.
  - Cosign similarities are generated for each word in the parsed vocabulary list.
  - The words with the cosign similarity that is closest to the predicted cosing similarity are then chosen as possible answer, and the one with the highest confidence between all three models is chosen as the final prediction.

# Exploratory Data Analysis:


![Percentage of clues with different special characters](/images/punctuation_percents.png?raw=true "Percentage of clues that contain each for of punctuation (before cleaning)")

- The presence of underscore indicates a "fill in the blank" type of clue which is different in nature from other clues
  - these clues could be diverted to a  model trained for this purpose
- Quotation marks or capital letters frequently indicate the involvement of a noun
- - Most other punctuation is just noise

![3103325 unique pairings, 2579749 unique clues, 315116 unique answers](/images/number_of_clues_and_answers.png?raw=true "Number of unique clues and answers in cleaned dataset" )

- Of the 6 million + samples in the full dataset, this graph shows that roughly half of them are duplicate answer-clue pairings and that there are aproximately, and only 1/10 of that number when looking at unique answers only.

![1 occurance: 33.3%, 2-10 occurances: 12.3%, 11-99 occurances: 12.3%, 100+ occurances: 1.9%](/images/answer_frequency.png?raw=true "Frequency of repeated answers." )

- Of the ~300,000 unique answers 52.5% of of them occure only once in the data set, and 1.9% of them occure more than 100 times.
- This indicates that the model will need to be robust, as even working within the confines of the data set it will need to be predicting words that it has not seen before.

![conceptnet: 79.27%, glove-wiki-gigaword: 79.18%, glove-twitter-25: 78.39%, word2vec-google-news: 70.94%](/images/vocabulary_percentages.png?raw=true "Percentage of answers contained in vocabularies of pre-trained word2vec models." )

- All pretrained models tested have 70-80% of the answers in their vocabulary
  - There are a fair number of answers that are compound words, or creatively made up words
  - An additional model could be made for generating compound words to increase the robustness of the predictions, but otherwise these types of anwers will not be able to be predicted 
- Using an ensamble aproach of all the pretrained models might be worth while in order to capture the greatest number of answers. 


