PATH="s3://aspangher/edit-pathways/spark_processing_scripts-output/nyt/"
ENDPOINT="http://s3.dev.obdc.bcs.bloomberg.com"

aws s3 ls s3://aspangher/edit-pathways/spark_processing_scripts-output/nyt/ --endpoint http://s3.dev.obdc.bcs.bloomberg.com | awk -F ' ' '{print $3,$4}' | awk -F ' ' '$1 < 101 {print $2}' | xargs -IP echo '{"Key": "P"}' > delete.txt

#Because bulk delete limit is 1000 per api call.
split -l 1000 delete.txt

#Append json start and end parts to complete a bulk delete request in every file.
for file in x*; do
  echo '{"Objects": [' >> delete"$file" && paste -d, -s "$file" >> delete"$file" &&
  echo '],"Quiet": true }' >> delete"$file"
done

#Send delete requests as json content in delete* files.
for file in deletex*; do
  aws s3api delete-objects --bucket s3://aspangher --delete file://"$file" --endpoint http://s3.dev.obdc.bcs.bloomberg.com
done