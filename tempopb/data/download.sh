#!/bin/bash
cd "$(dirname "$0")/pb"
total=$(wc -l < ../instances_list.txt)
n=0
while read -r f; do
  n=$((n+1))
  if [ ! -s "$f" ]; then
    curl -s --max-time 120 -o "$f" "https://pabulib.org/download/$f" || echo "FAIL $f" >> ../download_failures.log
    sleep 0.15
  fi
  if [ $((n % 50)) -eq 0 ]; then echo "[$n/$total] downloaded"; fi
done < ../instances_list.txt
echo "DONE $n files attempted"
ls | wc -l
