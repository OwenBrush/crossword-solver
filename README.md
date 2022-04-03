# crossword-solver
Machine learning model trained to solve crossword questions

### Goals:
- Practice with natural language processing 
- Create a model with an interesting and understandable application
- Provide clear data visualization to showcase the results of the model

### What it will do:
- Given a clue and a number of letters (with some optionally filled in) the model will return the most likely answers.

### Stretch Goals:
- Solve an entire puzzle, when given a list of clues and intersecting character positions
- Use image recognition to solve a puzzle from an image or photo of a crossword

### Datasets and tools:
- https://xd.saul.pw/data/#download (xd-clues.zip)
- gensim word2vec pre-trained models:
  - conceptnet-numberbatch-17-06-300
  - glove-wiki-gigaword-300
  - glove-twitter-200
  - word2vec-google-news-30

### Insights from data exploration and cleaning:


![Alt Text](/images/punctuation_percents.png?raw=true "Percentage of clues that contain each for of punctuation (before cleaning)")

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


