#!/bin/bash
#SBATCH -p g7e8-spot
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH -J a100_r1_eval
#SBATCH -o /fsx/chris.beckham/zara_speedup/out/a100_r1_eval.log
# Zara's internal validation of the r1 members retrained on the A100 (1 November): the 4-member average, then each member.
W=/fsx/chris.beckham/zara_speedup; R=$W/out/a100_r1/mast/r1; E=$W/out/eval_a100r1
unshare -rm bash $W/scripts/eval_rt_member.sh $E a100_ens $R/rt_ft $R/rt_mix $R/rt_ftw3 $R/rt_nonoise > $E.ens.log 2>&1
for m in rt_ft rt_mix rt_ftw3 rt_nonoise; do unshare -rm bash $W/scripts/eval_rt_member.sh $E a100_$m $R/$m > $E.$m.log 2>&1; done
for f in ens rt_ft rt_mix rt_ftw3 rt_nonoise; do echo "$f: $(grep -h 'S all' $E.$f.log)"; done
