function normalize_text {
    awk '{print tolower($0);}' < $1 | LC_ALL=C sed -e 's/\./ \. /g' -e 's/<br \/>/ /g' -e 's/"/ " /g' \
    -e 's/,/ , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' -e 's/\?/ \? /g' \
    -e 's/\;/ \; /g' -e 's/\:/ \: /g' > $1-norm
}

wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xf aclImdb_v1.tar.gz
## normalize the data
cd aclImdb
for j in train/pos train/neg test/pos test/neg train/unsup; do
    rm -f temp
    rm -f $j/norm.txt
    for i in `ls $j`; do cat $j/$i >> temp; awk 'BEGIN{print;}' >> temp; done
    normalize_text temp
    mv temp-norm $j/norm.txt
done
cat train/pos/norm.txt train/neg/norm.txt train/unsup/norm.txt test/pos/norm.txt test/neg/norm.txt > alldata.txt
## shuffle the training set
shuf alldata.txt > alldata-shuf.txt
cd ..

cd build
cmake .. && make

cd ..
# this script trains on all the data (train/test/unsup), you could also remove the test documents from the learning of word/document representation
time build/doc2vecc -train ./aclImdb/alldata-shuf.txt -word wordvectors.txt -output docvectors.txt -cbow 1 -size 100 -window 10 -negative 5 -hs 0 -sample 0 -binary 0 -iter 20 -min-count 10 -test ./aclImdb/alldata.txt -sentence-sample 0.1 -save-vocab alldata.vocab

head -n 25000 docvectors.txt | awk 'BEGIN{a=0;}{if (a<12500) printf "1 "; else printf "-1 "; for (b=1; b<=NF; b++) printf b ":" $(b) " "; print ""; a++;}' > train.txt
tail -n 25000 docvectors.txt | awk 'BEGIN{a=0;}{if (a<12500) printf "1 "; else printf "-1 "; for (b=1; b<=NF; b++) printf b ":" $(b) " "; print ""; a++;}' > test.txt
