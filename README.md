# cnn-mail-summarization
## Read Text summarization.docx<br>
### Download data from https://drive.google.com/file/d/1VGthRzHtBSIO182zMCMiqY-YV-D0mLLG/view and place train and test folders in a folder named dataset in root directory.<br>
### Place the text to be summarized in sample.txt or you can place the file in root directory as well. Below steps takes sample.txt as example.
# steps to execute
1. pip install -r requirements.txt
2. import run
3. run.format_and_process_data()
4. run.train_abstractive()
5. print("Abstractive summary is:\n", run.predict_abstactive_summary('sample.txt'))
6. run.train_extractive()
7. print("Extractive summary is:\n", run.predict_extractive_summary('sample.txt'))
