#!/bin/sh
cd /mnt/hgfs/vmfiles/genres/
for i in `ls`;
  do 
      cd $i;
      for j in `ls`;
        do
        	newj=`echo $j|sed -r 's/(.*)(\..*)/\1/g'`
        	sox $j $newj."wav"
        	rm $j
        done
      cd ..
  done
