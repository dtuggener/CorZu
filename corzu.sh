#!/bin/sh

corzu_dir="/home/user/tuggener/hex_storage/CorZu_v2.0"  #adjust this to your CorZu directory
parzu_cmd="parzu -q"   #the command used for starting parzu

echo "Parsing ..."
cat $1 |$parzu_cmd > $1".parzu" #parse
echo "\nDone.\nExtracting markables..."
python $corzu_dir/extract_mables_from_parzu.py $1".parzu" > $1".mables" #markable extraction
echo "Done.\nResolving coreference..."
python $corzu_dir/corzu.py $1".mables" $1".parzu" $1".coref"  #coreference resolution
echo "\nDone.\nWriting HTML..."
python $corzu_dir/conll_to_html.py $1".coref" > $1".html"
echo "\nDone."
#rm $1".mables" #uncomment to delete markable file
#rm $1".parzu" #uncomment to delete parse file
