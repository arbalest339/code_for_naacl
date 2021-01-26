# code_for_naacl

# Description
This is my experimental codes, the name of the method hasn't been figured out yet.

It consists of two parts the Dependency Parsing embedding model and the Open Information Extraction model, the latter part uses the out put of the formmer part.

# Environment
python>=3.6.7

CUDA>=9.0

torch==1.3.0

tensorflow==1.14.0

You can install all the required python packages by running
```
pip  install -r requirements.txt
```

# Preparations
+ Create a dir named checkpoints under both EMB_model and OIE_model dirs.

+ Place the English dataset files under en_data dir and Chinese dataset files under zh_data dir.There are examples  in OIE_model/zh_data and OIE_model/en_data. Every single data should have following fields:
	+ token: the origin tokens of the sentence
	+ pos: the Part-Of-Speech tag of every token
	+ ner: the Named-Entity-Recognition tag of every token
	+ gold: the tagert OIE tag used for traing for every token
	+ sub: the index of the subject
	+ obj: the index of the object
	+ pred: the index of the predicate
+ "pos" and "ner" field are tagged by [LTP](http://www.ltp-cloud.com/) (for Chinese data) and [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/) (for English data). And it must be converted to integers according to the *_map.json in the same dir.
+ The dependecy parsing tree adjacency matrix for every sentence should also be packaged as .npy files. The sequence is consistent with the data file. **NOTE** that the first node on DP tree is the root node while the second node of DP tree is the first token.
+ You can also download the [NeuralOpenIE](https://1drv.ms/u/s!ApPZx_TWwibImHl49ZBwxOU0ktHv) data set and use extract_mydata.py to convert it to a ready-to-use data. To do this, you need to place neural_oie.sent and neural_oie.triple under the extract_dataset/ dir and run 
```
python extract_mydata.py
```
+ Make sure the training, development and testing sets are already prepared and properly named.
+ After the previous steps, the current file directory should look like this:

```
EMB_model/
├── checkpoints
├── config.py
├── data_reader.py
├── en_data/
│   ├── dp_emb_dev_matrixs_en.npy
│   ├── dp_emb_dev_en.json
│   ├── dp_emb_train_matrixs_en.npy
│   ├── dp_emb_train_en.json
│   ├── dp_map.json
│   ├── ner_map.json
│   └── pos_map.json
├── models/
│   ├── aggcn.py
│   ├── cnn_lstm_model.py
│   ├── oie_model.py
│   ├── TransD.py
│   └── trans_model.py
├── out/
├── train_trans_model.py
└── zh_data/
    ├── dp_emb_dev_matrixs_zh.npy
    ├── dp_emb_dev_zh.json
    ├── dp_emb_train_matrixs_zh.npy
    ├── dp_emb_train_zh.json
    ├── dp_map.json
    ├── ner_map.json
    └── pos_map.json
```
```
OIE_model/
├── checkpoints
├── config.py
├── data_reader.py
├── en_data/
│   ├── dp_map.json
│   ├── ner_map.json
│   ├── oie_dev_en.json
│   ├── oie_dev_matrixs_en.npy
│   ├── oie_test_en.json
│   ├── oie_test_matrixs_en.npy
│   ├── oie_train_en.json
│   ├── oie_train_matrixs_en.npy
│   └── pos_map.json
├── error_analyse.py
├── models/
│   ├── aggcn.py
│   ├── cnn_lstm_model.py
│   ├── oie_model.py
│   ├── TransD.py
│   └── trans_model.py
├── out/
├── test.py
├── train_main_model.py
└── zh_data/
    ├── dp_map.json
    ├── ner_map.json
    ├── oie_dev_matrixs_zh.npy
    ├── oie_dev_zh.json
    ├── oie_test_matrixs_zh.npy
    ├── oie_test_zh.json
    ├── oie_train_matrixs_zh.npy
    ├── oie_train_zh.json
    └── pos_map.json
```

# Run Experiment
+ Edit config.py in both parts, vars need to be set are pointed out by `【Should be set according to the actual situation】` comment.
+ Change dir to EMB_model/ and run train_trans_model.py, the training process may take several days when dataset is large.
```
cd EMB_model
python3 train_trans_model.py
```
+ A .npy file named dp_emb_{language}.npy will be stored in the EMB_model/checkpoints
+ Move this .npy file to OIE_model/checkpoints
```mv EMB_model/checkpoints/dp_emb_{language}.npy OIE_model/checkpoints```
+ Change dir to OIE_model/ and run train_main_model.py, wait for training. The result can be obtained by running test.py

```
cd OIE_model
python3 train_main_model.py
python3 test.py
```
