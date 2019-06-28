# Data from https://github.com/dunan/NeuralPointProcess
echo "=== Acquiring Temporal Point Process Data ==="
echo "---"
mkdir -p data
cd data

echo "- Downloading Book Order Financial Data"
wget --continue https://raw.githubusercontent.com/dunan/NeuralPointProcess/master/data/real/book_order/event-1-test.txt
wget --continue https://raw.githubusercontent.com/dunan/NeuralPointProcess/master/data/real/book_order/event-1-train.txt
wget --continue https://raw.githubusercontent.com/dunan/NeuralPointProcess/master/data/real/book_order/time-1-test.txt
wget --continue https://raw.githubusercontent.com/dunan/NeuralPointProcess/master/data/real/book_order/time-1-train.txt

mkdir -p tpp-book-order
cd tpp-book-order
mv ../event-1-test.txt  event-1-test.txt
mv ../event-1-train.txt  event-1-train.txt
mv ../time-1-test.txt  time-1-test.txt
mv ../time-1-train.txt  time-1-train.txt
cd ..

#echo "- Cleaning up extra files"
echo "... Done"
