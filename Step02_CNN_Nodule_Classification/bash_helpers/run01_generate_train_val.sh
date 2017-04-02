#!/bin/bash

ftmp='tmp_shuf.txt'
numVal='50'

lstLbl="0 1 2"

fidxTrn="idx-trn.txt"
fidxVal="idx-val.txt"

ostype=`uname -s`

if [ "${ostype}" == "Darwin" ]; then
    rshuf="gshuf"
else
    rshuf="shuf"
fi

:> $fidxTrn
:> $fidxVal

for ii in `echo $lstLbl`
do
    echo "[${ii}]"
    find . -wholename "*/${ii}_*.nii.gz" | sed 's/\.\///' | ${rshuf} > $ftmp
    head -n $numVal $ftmp >> $fidxVal
    numLbl=`cat ${ftmp} | wc -l`
    ((numTrn=numLbl-numVal))
    tail -n $numTrn $ftmp >> $fidxTrn
done
