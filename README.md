# NLP-Text-Summarization

The goal for this project is to build and assess a machine learning model that reads in news articles and automatically converts them into a summarized form. The model achieves this by identifying the salient features in an article, filtering out the less important information, and then producing a shortened version of the original text.

To accomplish this, Natural Language Processing (NLP) was leveraged by applying Hugging Face’s Text-To-Text Transfer Transformer, also known as T5. T5 is a pre-trained transfer learning model that is applicable to many NLP tasks, including text summarization. Transfer learning is a machine learning technique in which a pre-trained model is repurposed to handle a related task. That is, the pre-trained model is re-trained, or fined-tuned, to more adequately perform the task at hand. 

To train, test, and validate the model, a dataset with 300,000 articles from CNN, with target summaries, was identified. The dataset can be found [here](https://huggingface.co/datasets/cnn_dailymail). 


This project can be broken into four main steps
- Preprocessing the data
- Visualizing the data
- Building and fine-tuning the model
- Testing the model


This project is a work in progress. Check back later!
