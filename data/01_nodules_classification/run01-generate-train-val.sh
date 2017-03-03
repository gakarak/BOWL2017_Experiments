#!/bin/bash

finp="idx.txt"
finpShuf="${finp}-shuf.txt"

foutTrain="${finp}-train.txt"
foutVal="${finp}-val.txt"

#############################
cat $finp | grep -v 'label' | shuf > $finpShuf
numFn=`cat $finpShuf | wc -l`
pVal=30

((numVal=numFn*pVal/100))
((numTrain=numFn-numVal))

echo "train/val/tot = ${numTrain}/${numVal}/${numFn}"

#############################
thdr=`cat ${finp} | head -n 1`

echo "${thdr}" > $foutTrain
cat $finpShuf | head -n $numTrain >> $foutTrain

echo "${thdr}" > $foutVal
cat $finpShuf | tail -n $numVal   >> $foutVal
