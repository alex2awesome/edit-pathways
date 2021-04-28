if True
then
  for i in 4 6
  do
  gcloud beta compute \
      --project=usc-research instances create-with-container edit-parser-1-$i \
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
      --boot-disk-size=20GB \
      --boot-disk-type=pd-standard \
      --boot-disk-device-name=edit-parser-1-$i \
      --container-image=us.gcr.io/usc-research/edit-parser:latest \
      --container-restart-policy=always \
      --container-privileged \
      --container-stdin \
      --labels=container-vm=cos-stable-85-13310-1041-161 \
      &
  done
fi


if False
then
  for outlet in bbc-2 reuters
  do
    gcloud compute ssh \
          --zone "us-central1-a" "edit-parser-1-1" \
          --project "usc-research" \
          --command "docker run -v /home/:/app/data/ us.gcr.io/usc-research/edit-parser python3 parsing_script.py --source_db_name $outlet --add_to_datastore --n_file 1000000" \
          &
  done
fi

if False
then
## box 1
#gcloud compute ssh --zone "us-central1-a" "edit-parser-1-1" --project "usc-research" --command "docker run -v /home/:/app/data/ us.gcr.io/usc-research/edit-parser python3 parsing_script.py --source_db_name guardian --add_to_datastore --n_file 1000000"  > /dev/null 2>&1 &
#gcloud compute ssh --zone "us-central1-a" "edit-parser-1-1" --project "usc-research" --command "docker run -v /home/:/app/data/ us.gcr.io/usc-research/edit-parser python3 parsing_script.py --source_db_name bbc-1 --add_to_datastore --n_file 1000000"  > /dev/null 2>&1 &
#gcloud compute ssh --zone "us-central1-a" "edit-parser-1-1" --project "usc-research" --command "docker run -v /home/:/app/data/ us.gcr.io/usc-research/edit-parser python3 parsing_script.py --source_db_name bbc-2 --add_to_datastore --n_file 1000000"  > /dev/null 2>&1 &
#gcloud compute ssh --zone "us-central1-a" "edit-parser-1-1" --project "usc-research" --command "docker run -v /home/:/app/data/ us.gcr.io/usc-research/edit-parser python3 parsing_script.py --source_db_name reuters --add_to_datastore --n_file 1000000"  > /dev/null 2>&1 &

## box 2
#gcloud compute ssh --zone "us-central1-a" "edit-parser-1-2" --project "usc-research" --command "docker run -v /home/:/app/data/ us.gcr.io/usc-research/edit-parser python3 parsing_script.py --source_db_name cnn --add_to_datastore --n_file 1000000"  > /dev/null 2>&1 &
#gcloud compute ssh --zone "us-central1-a" "edit-parser-1-2" --project "usc-research" --command "docker run -v /home/:/app/data/ us.gcr.io/usc-research/edit-parser python3 parsing_script.py --source_db_name cbc --add_to_datastore --n_file 1000000"  > /dev/null 2>&1 &
gcloud compute ssh --zone "us-central1-a" "edit-parser-1-2" --project "usc-research" --command "docker run -v /home/:/app/data/ us.gcr.io/usc-research/edit-parser python3 parsing_script.py --source_db_name fox --add_to_datastore --n_file 1000000"  > /dev/null 2>&1 &
gcloud compute ssh --zone "us-central1-a" "edit-parser-1-2" --project "usc-research" --command "docker run -v /home/:/app/data/ us.gcr.io/usc-research/edit-parser python3 parsing_script.py --source_db_name independent --add_to_datastore --n_file 1000000"  > /dev/null 2>&1 &

## box 3
#gcloud compute ssh --zone "us-central1-a" "edit-parser-1-3" --project "usc-research" --command "docker run -v /home/:/app/data/ us.gcr.io/usc-research/edit-parser python3 parsing_script.py --source_db_name dailymail --add_to_datastore --n_file 1000000"  > /dev/null 2>&1 &
gcloud compute ssh --zone "us-central1-a" "edit-parser-1-3" --project "usc-research" --command "docker run -v /home/:/app/data/ us.gcr.io/usc-research/edit-parser python3 parsing_script.py --source_db_name therebel --add_to_datastore --n_file 1000000"  > /dev/null 2>&1 &
gcloud compute ssh --zone "us-central1-a" "edit-parser-1-3" --project "usc-research" --command "docker run -v /home/:/app/data/ us.gcr.io/usc-research/edit-parser python3 parsing_script.py --source_db_name torontostar --add_to_datastore --n_file 1000000"  > /dev/null 2>&1 &
gcloud compute ssh --zone "us-central1-a" "edit-parser-1-3" --project "usc-research" --command "docker run -v /home/:/app/data/ us.gcr.io/usc-research/edit-parser python3 parsing_script.py --source_db_name torontosun --add_to_datastore --n_file 1000000"  > /dev/null 2>&1 &

## box  > /dev/null 2>&1 4
#gcloud compute ssh --zone "us-central1-a" "edit-parser-1-4" --project "usc-research" --command "docker run -v /home/:/app/data/ us.gcr.io/usc-research/edit-parser python3 parsing_script.py --source_db_name calgaryherald --add_to_datastore --n_file 1000000"  > /dev/null 2>&1 &
gcloud compute ssh --zone "us-central1-a" "edit-parser-1-4" --project "usc-research" --command "docker run -v /home/:/app/data/ us.gcr.io/usc-research/edit-parser python3 parsing_script.py --source_db_name globemail --add_to_datastore --n_file 1000000"  > /dev/null 2>&1 &
gcloud compute ssh --zone "us-central1-a" "edit-parser-1-4" --project "usc-research" --command "docker run -v /home/:/app/data/ us.gcr.io/usc-research/edit-parser python3 parsing_script.py --source_db_name canadaland --add_to_datastore --n_file 1000000"  > /dev/null 2>&1 &
gcloud compute ssh --zone "us-central1-a" "edit-parser-1-4" --project "usc-research" --command "docker run -v /home/:/app/data/ us.gcr.io/usc-research/edit-parser python3 parsing_script.py --source_db_name whitehouse --add_to_datastore --n_file 1000000"  > /dev/null 2>&1   > /dev/null 2>&1&

## box  > /dev/null 2>&1 5
#gcloud compute ssh --zone "us-central1-a" "edit-parser-1-5" --project "usc-research" --command "docker run -v /home/:/app/data/ us.gcr.io/usc-research/edit-parser python3 parsing_script.py --source_db_name lapresse --add_to_datastore --n_file 1000000"  > /dev/null 2>&1 &
#gcloud compute ssh --zone "us-central1-a" "edit-parser-1-5" --project "usc-research" --command "docker run -v /home/:/app/data/ us.gcr.io/usc-research/edit-parser python3 parsing_script.py --source_db_name nationalpost --add_to_datastore --n_file 1000000"  > /dev/null 2>&1 &
#gcloud compute ssh --zone "us-central1-a" "edit-parser-1-5" --project "usc-research" --command "docker run -v /home/:/app/data/ us.gcr.io/usc-research/edit-parser python3 parsing_script.py --source_db_name telegraph --add_to_datastore --n_file 1000000"  > /dev/null 2>&1 &

fi