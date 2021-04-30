# Dataset

Here are the datasets which can be used by this emotional chat-bot project. Currently, we find 2 useful dataset:
* DailyDialog
* EmpatheticDialog

The origin datasets are available at these links, respectively:
* http://yanran.li/dailydialog.html
* https://github.com/facebookresearch/EmpatheticDialogues

Here, we made some pre-process of these dataset so that they can be aligned to our project.

## Data format

Since our project focuses on the impact of introducing emotion into the chat-bot's dialogue generation, we construct data in this format:

```
{'src': source text, 'trg': target text, 'trg_emotion': the emotion that should be shown in the target text}
```

The data are saved in json files. Each json file has multiple lines where each line is a data instance which has the above format. __Source Text__ are text in the history of this dialogue (i.e. previou utterances in this dialogue). __Target Text__ are text in the up-coming utterance that the model need to generate. Currently, we assume that the emotion is known (or say, we can have directly control of the emotion), therefore, we also provide the ground-truth emotion of the target text. Clearly, this emotion can also be predicted based on previous utterance, but we believe it is out of the boundary of our project (at this time :) ).

Note that the __trg_emotion__ region is the text of the emotion (i.e. 'surprised'), not the relevant index (i.e. 0).

## Text Data

In order to seperate different utterances from each other and also emphasize which user provides the specific utterance, we add 3 special tokens: 
* \<user0\>
* \<user1\>
* \_\_eou\_\_

where \<user0\> and \<user1\> are tokens that represent different users/agents in the dialogue (we assume that there are only 2 people in each dialogue) and utterances in the __src__ field are separeted by the special token \_\_eou\_\_. An example of the data is shown below:

```python
{"src": "<user0> My word , you do look ill ! __eou__ <user1> I'm quite out of thoughts recently . __eou__ <user0> You seem to have something on mind . Promise me , go to see the doctor right now .", "trg": "<user1> I'm worrying about my exam .", "trg_emotion": "neutral"}
```

## Emotion Labels
There are multiple emotion labels in these datasets.
### DailyDialog
*dailydialog* dataset has 7 different emotions:
* 0 - neutral
* 1 - anger
* 2 - disgust
* 3 - fear
* 4 - happiness
* 5 - sadness
* 6 - surprise

### EmpatheticDialog
*empatheticdialog* dataset has even more and fine-grained emotions:
| | | | |
| :--- | :--- | :--- | :--- |
| 0 - Surprised | 1 - Excited | 2 - Angry | 3 - Proud |
| 4 - Sad   | 5 - Annoyed | 6 - Grateful | 7 - Lonely |
| 8 - Afraid | 9 - Terrified | 10 - Guilty | 11 - Impressed |
| 12 - Disgusted | 13 - Hopeful | 14 - Confident | 15 - Furious |
| 16 - Anxious | 17 - Anticipating | 18 - Joyful | 19 - Nostalgic |
| 20 - Disappointed | 21 - Prepared | 22 - Jealous | 23 - Content |
| 24 - Devastated | 25 - Embarrassed | 26 - Caring | 27 - Sentimental |
| 28 - Trusting | 29 - Ashamed | 30 - Apprehensive | 31 - Faithful |