#!/bin/bash
# One line per run: params, latest step, holdout accuracy, hard-set accuracy.
cd /workspace
printf "%-5s %7s %8s %10s %8s  %s\n" RUN PARAMS STEP HOLDOUT HARD CONFIG
for f in $(ls -1 runs/*.log | sort -V); do
  tag=$(basename "$f" .log)
  cfg=$(grep -m1 'cfg' "$f" | sed 's/\[cfg\] //')
  par=$(echo "$cfg" | grep -o 'params=[0-9]*' | cut -d= -f2)
  last=$(grep -E '^\[ *[0-9]+\]' "$f" | tail -1)
  fin=$(grep -m1 '^\[final\] params' "$f")
  if [ -n "$fin" ]; then
    step=DONE
    ho=$(echo "$fin" | grep -o 'holdout=[0-9.]*' | cut -d= -f2)
    hd=$(echo "$fin" | grep -o 'chain=[0-9.]*' | cut -d= -f2)
  else
    step=$(echo "$last" | grep -o '^\[ *[0-9]*\]' | tr -dc 0-9)
    ho=$(echo "$last" | grep -o 'holdout=[0-9.]*' | cut -d= -f2)
    hd=$(echo "$last" | grep -o 'chain=[0-9.]*' | cut -d= -f2)
  fi
  printf "%-5s %7s %8s %9s%% %7s%%  %s\n" "$tag" "$par" "${step:--}" "${ho:--}" "${hd:--}" \
      "$(echo "$cfg" | sed 's/ params=.*//')"
done
