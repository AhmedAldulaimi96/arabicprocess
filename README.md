## arabicprocess package

The `arabicprocess` class is designed to perform various text preprocessing steps for Arabic text data. It provides methods to clean and preprocess Arabic text, including data cleaning, stop word removal, and stemming. It also offers options to control the level of detail, generate charts, and save results as CSV files.

**Key Features:**
- Data Cleaning: Remove irrelevant characters, punctuation, and URLs, and handle common spelling errors in Arabic text.
- Stop Word Removal: Eliminate common Arabic stopwords to improve the quality of text analysis.
- Stemming: Apply Arabic word stemming to reduce inflected words to their base form, helping with text normalization.

`arabicprocess` is built on top of popular Python libraries such as NLTK, NumPy, Pandas, Seaborn, and Matplotlib. It is easy to use, efficient, and suitable for various NLP tasks, including text classification, sentiment analysis, and topic modeling.

Please refer to the documentation for more details and examples on how to use arabicprocess in your projects. Your feedback and contributions are highly appreciated as we aim to continuously improve the library and make it more useful for the Arabic NLP community.


### Class Initialization:

```python
process(process_list=['clean','st_remove','stemm'], details=True, chart=True, csv=True, column_stay=True)
```

**Parameters:**
- `process_list` (default: `['clean','st_remove','stemm']`): A list that defines the text preprocessing steps to be performed. The available options are `'clean'` for data cleaning, `'st_remove'` for stop word removal, and `'stemm'` for stemming.
- `details` (default: `True`): A boolean flag indicating whether to print detailed information about each preprocessing step. If set to `True`, it will print information like the number of tokens and characters at each preprocessing stage.
- `chart` (default: `True`): A boolean flag indicating whether to generate and display charts showing the number of tokens and characters at each preprocessing step.
- `csv` (default: `True`): A boolean flag indicating whether to save the resulting dataset as CSV files after each preprocessing step.
- `column_stay` (default: `True`): A boolean flag indicating whether to keep the text column in the dataset. If `True`, it will move the column with the most tokens to the first position; otherwise, it will select the column with the most tokens as the text column.

### Usage:

Here's how you can use the `arabicprocess` package:

```python
#install arabicprocess package
pip install arabicprocess

#Import process class from arabicprocess library
from arabicprocess import process

# Load your dataset (MAKE SURE TO ADD encoding='utf-8', if you dont add it may cause problems)
import pandas as pd
dataset = pd.read_csv('your_dataset.csv', encoding='utf-8')

# Initialize the arabicprocess object
arabic_processor = process(process_list=['clean', 'st_remove', 'stemm'], details=True, chart=True, csv=True, column_stay=True)

# Perform preprocessing on the dataset
arabic_processor.main(dataset)
```

The class will perform the preprocessing steps based on the `process_list` specified during initialization. It will display detailed information about the preprocessing steps and generate charts if `details` and `chart` are set to `True`. If `csv` is set to `True`, the preprocessed datasets will be saved as CSV files after each preprocessing step.


### Example

First, you need to import your dataset as follows (MAKE SURE TO ADD encoding='utf-8', if you dont add it may cause problems),

```python
import pandas as pd

df=pd.read_csv('Original.csv', encoding='utf-8')
df.head(3)
```

Output:

```
                                                text     label
0  اجتمع رئيس الوزراء بعدد من أعضاء مجلس الوزراء لبحث القضايا المهمة في البلاد.  politics
1       فاز الفريق المحلي في نهائي بطولة الكرة الطائرة الوطنية بنتيجة 3-1.     sport
2  افتتحت المعرض الفني للفنان المبدع، يعرض فيه أحدث أعماله التشكيلية.       art
```

Then we can use `arabicprocess` package as follows,

```python
#install arabicprocess package
pip install arabicprocess

#Import process class from arabicprocess library
from arabicprocess import process

AP = process()
AP.main(dataset=df)
```

Output:

```
-> Origninal data Stats calculating...

Number of Tokens (words) in the orijinal dataset (before any pre-processing) : 8851197
Number of Characters in the original dataset (before any pre-processing) : 46860802 

-----------------------------


--> Data Cleaning step begain...

Number of Tokens (words) after (Data Cleaning) step : 7832346
Number of Characters after (Data Cleaning) step : 45701603 

-----------------------------


---> Stop Word Removal step begain...

Number of Tokens (words) after (Data Cleaning) step : 5624127
Number of Characters after (Data Cleaning) step : 36965437 

-----------------------------


----> Text Steamming step begain...

Number of Tokens (words) after (Text Steamming) step : 5624124
Number of Characters after (Text Steamming) step : 25694698 

-----------------------------

Time required to complete pre-processing steps: 453.09  second  ( 7.55  minute).

-----------------------------
```
![arabicprocess chart if chart==True](https://github.com/AhmedAldulaimi96/arabicprocess/blob/main/images/chart.png)

- If you set the `csv` parameter to `True`, each of these datasets will be saved as CSV files after their respective preprocessing stages. You will get CSV files for the data after data cleaning, stop word removal, and stemming with the filenames `1after_Cleaning.csv`, `2after_Stop_word_remove.csv`, and `3after_stemming.csv`, respectively. These CSV files will contain the preprocessed data with the respective columns for tokens and characters counts.

#### After Data Cleaning:
```
text                                              label  tokens_count  char_count  tokens_count2
0   اجتمع رئيس الوزراء بعدد من أعضاء مجلس الوزراء لبحث القضايا المهمة في البلاد  politics          162        1089            161
1   فاز الفريق المحلي في نهائي بطولة الكرة الطائرة الوطنية بنتيجة   sport          193        1076            191
2   افتتحت المعرض الفني للفنان المبدع يعرض فيه أحدث أعماله التشكيلية  art          450        2551            442
...  ...
```

#### After Stop Word Removal:
```
text                                              label  tokens_count  char_count  tokens_count2  char_count2
0   اجتمع رئيس الوزراء بعدد أعضاء مجلس الوزراء لبحث القضايا المهمة البلاد  politics          162        1089            161         1054
1   فاز الفريق المحلي نهائي بطولة الكرة الطائرة الوطنية بنتيجة  sport          193        1076            191         1030
2   افتتحت المعرض الفني للفنان المبدع يعرض أحدث أعماله التشكيلية  art          450        2551            442         2496
...  ...
```

#### After Data Stemming:
```
text                                              label  tokens_count  char_count  tokens_count2  char_count2  tokens_count3  char_count3
0   اجتمع رئيس وزراء عدد أعضاء مجلس وزراء بحث قضايا مهم بلاد  politics          162        1089            161         1054            156          992
1   فاز فريق محلي نهائي بطولة كرة طائرة وطني نتيجة  sport          193        1076            191         1030            189          987
2   افتتح معرض فني فنان مبدع عرض أحدث أعمال تشكيل  art          450        2551            442         2496            438         2420
...  ...
```

The `tokens_count`, `char_count`, `tokens_count2`, `char_count2`, `tokens_count3`, and `char_count3` columns represent the number of tokens and characters at different stages of preprocessing, 1 for `data cleaning`, 2 for `stop word removal`, and 3 for `stemming`.


### Documentation

Find more information in the [documentation](https://link-to-documentation).

### License

This project is licensed under the MIT License. Find more information about MIT License [HERE](https://opensource.org/licenses/MIT).

## **Questions or Feedback? Get in Touch!**

If you have any questions, suggestions, or feedback related to this package, I'd love to hear from you! Please don't hesitate to reach out via email at [ahmedhashimirq@gmail.com](mailto:ahmedhashimirq@gmail.com). Your input is valuable, and I'm here to help in any way I can.
