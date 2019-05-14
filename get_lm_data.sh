# Modified From https://github.com/salesforce/awd-lstm-lm/ getdata.sh
echo "=== Acquiring Language Modeling Datasets ==="
echo "---"
mkdir -p data
cd data

echo "- Downloading Penn Treebank (PTB)"
wget --continue http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
tar -xzf simple-examples.tgz

mkdir -p ptb
cd ptb
mv ../simple-examples/data/ptb.train.txt train.txt
mv ../simple-examples/data/ptb.test.txt test.txt
mv ../simple-examples/data/ptb.valid.txt valid.txt
cd ..

echo "- Downloading WikiText-2 (WT2)"
wget --continue https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
unzip -q wikitext-2-v1.zip
cd wikitext-2
mv wiki.train.tokens train.txt
mv wiki.valid.tokens valid.txt
mv wiki.test.tokens test.txt
cd ..

echo "- Cleaning up extra files"
rm -r ./simple-examples/
rm simple-examples.tgz
rm wikitext-2-v1.zip

echo "... Done"
