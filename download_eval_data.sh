# aqua

cd data/aqua
wget https://github.com/google-deepmind/AQuA/blob/master/dev.json -O dev.json
wget https://github.com/google-deepmind/AQuA/blob/master/test.json -O test.json

# bgqa
cd ..
wget https://storage.googleapis.com/gresearch/BoardgameQA/BoardgameQA.zip -O BoardgameQA.zip
unzip BoardgameQA.zip
mv BoardgameQA/* boardgameqa
rm -r BoardgameQA
rm BoardgameQA.zip

# ConditionalQA
cd conditionalqa
wget https://raw.githubusercontent.com/haitian-sun/ConditionalQA/master/v1_0/train.json -O train.json
wget https://raw.githubusercontent.com/haitian-sun/ConditionalQA/master/v1_0/dev.json -O dev.json
wget https://raw.githubusercontent.com/haitian-sun/ConditionalQA/master/v1_0/test_no_answer.json -O test_no_answer.json 
wget https://raw.githubusercontent.com/haitian-sun/ConditionalQA/master/v1_0/documents.json -O documents.json

# you need to contact the authors of ConditionalQA to get the test set answers or ask them to evaluate your responses

# HotpotQA
cd ../hotpotqa

wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json -O hotpot_train_v1.1.json
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json -O hotpot_dev_distractor_v1.json

cd ..