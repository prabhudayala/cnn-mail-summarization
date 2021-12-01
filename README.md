# cnn-mail-summarization
## Read Text summarization.docx<br>
### Download data from https://drive.google.com/file/d/1VGthRzHtBSIO182zMCMiqY-YV-D0mLLG/view and place train and test folders in a folder named dataset in root directory.<br>
### Place the text to be summarized in sample.txt or you can place the file in root directory as well. Below steps takes sample.txt as an example.
# steps to execute
1. pip install -r requirements.txt

2. open python in command line pointing to current directory and run below code. 
```python
import run
run.format_and_process_data()
run.train_abstractive()
print("Abstractive summary is:\n", run.predict_abstactive_summary('sample.txt'))
run.train_extractive()
print("Extractive summary is:\n", run.predict_extractive_summary('sample.txt'))
```
3. or as an alternative you can open "how to use.ipynb" file as reference to execute the code.
