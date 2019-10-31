gamma='0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1'
beta='0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1'

for g in $gamma;do
  for b in $beta; do
    python train.py -d wdbc --cls rda --gamma $g --beta $b --val-rate 0.2
  done
done
