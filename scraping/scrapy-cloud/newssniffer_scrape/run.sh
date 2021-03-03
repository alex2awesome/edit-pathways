#scrapy crawl newssniff-article-scraper -a num_splits=5 -a split_num=2  -o "working_dir/%(batch_id)d-articles-%(batch_time)s.json"

#scrapy crawl newssniff-article-page -a num_splits=5 -a split_num=2

if True
then
  for i in 1 2 3 4 5 6 7 8
  do
  echo gcloud beta compute \
      --project=usc-research instances create-with-container s-5-$i \
      --zone=us-central1-a \
      --machine-type=e2-standard-4 \
      --subnet=default \
      --network-tier=PREMIUM \
      --metadata=google-logging-enabled=true \
      --maintenance-policy=MIGRATE \
      --service-account=520950082549-compute@developer.gserviceaccount.com \
      --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append \
      --tags=http-server,https-server \
      --image=cos-stable-85-13310-1041-161 \
      --image-project=cos-cloud \
      --boot-disk-size=50GB \
      --boot-disk-type=pd-standard \
      --boot-disk-device-name=s-5-$i \
      --container-image=us.gcr.io/usc-research/selenium-scrapy-newssniff:latest \
      --container-restart-policy=always \
      --container-privileged \
      --container-stdin \
      --labels=container-vm=cos-stable-85-13310-1041-161
  done
fi

if False
then
  for i in 0 1 2 3 4 5 6 7
  do
     echo gcloud compute ssh \
      --zone "us-central1-a" "s-2-$[i+1]" \
      --project "usc-research" \
      --command \'"docker run -it \
  -v /home/:/app/working_dir \
  us.gcr.io/usc-research/selenium-scrapy-newssniff \
  scrapy crawl newssniff-article-page \
    -a num_splits=8 \
    -a split_num=$i \
    -o working_dir/article-output-$[i+1]-8.json"\'
  done
fi

if False
then
  for i in 0 1 2 3 4 5 6 7
  do
     echo "docker run -it \
  -v /home/:/app/working_dir \
  us.gcr.io/usc-research/selenium-scrapy-newssniff \
  scrapy crawl newssniffer-version-page \
    -a num_splits=8 \
    -a split_num=$i \
    -o working_dir/article-output-$[i+1]-8.json"
  done
fi



if False
then
  for i in 1 2 3 4 5 6 7 8
  do
    gcloud compute ssh \
      --zone "us-central1-a" "s-4-$i" \
      --project "usc-research" \
      --command 'docker ps -f "ancestor=us.gcr.io/usc-research/selenium-scrapy-newssniff:latest" --format "{{.ID}}" | xargs -I {} docker kill {}'
  done
fi

if False
then
  for i in 1 2 3 4 5 6 7 8
  do
#    gcloud compute ssh s-4-$i --command "docker ps" --zone "us-central1-a" --project "usc-research"
    gcloud beta compute scp s-4-$i:/home/article-output-$i-8.json . --zone "us-central1-a" --project "usc-research"
  done
fi
