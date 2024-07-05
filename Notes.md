1. Maybe use a scanner such as nomos first to check quickly if the file contains a license? depends on how reliable they are in terms of recall
2. Context window is an issue no matter the model at this size. - Although the models can be fine-tuned to have a much larger context window, I have never tried this before.


Now
1. From .txt to json or csv file 
2. Use paragraphs or some sort of different text for embedding
3. Create a dataset for and metrics:
   1. Dataset (1->75)
      1. Use Linux as baseline + Pytorch. Will add more incrementally later on
      2. Find out which lines are critical license identification and record them
      3. Find out which licenses these lines correspond to 
   2. Metrics
      1. Selecting correct line(s) for license identification out of top k
      2. Matching those lines to the correct license(s)