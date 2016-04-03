#!/bin/sh
cd /mnt/hgfs/vmfiles/genres/
for i in `ls`;
  do 
      cd $i;
      mkdir -p /mnt/hgfs/vmfiles/genreschord/$i
      for j in `ls`;
        do
        	file=`pwd`/$j;
          echo $file
        	python /home/tyc/Downloads/machine\ learning/chordextract.py $file
          
        done
      cd ..
  done
