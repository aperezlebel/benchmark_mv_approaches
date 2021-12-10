#!/bin/bash

declare -a arr5=(
    "TB/death_pvals"
    "TB/platelet_pvals"
    "TB/hemo_pvals"
    "TB/hemo"
    "TB/septic_pvals"
    "UKBB/breast_25"
    "UKBB/breast_pvals"
    "UKBB/skin_pvals"
    "UKBB/parkinson_pvals"
    "UKBB/fluid_pvals"
    "MIMIC/septic_pvals"
    "MIMIC/hemo_pvals"
    "NHIS/income_pvals"
    )

declare -a arr_reg=(
    "TB/platelet_pvals"
    "UKBB/fluid_pvals"
    "NHIS/income_pvals"
)

declare -a arr=(
    "TB/hemo"
    "UKBB/breast_25"
)

for method in {20,24,22,26}
do
	for task in "${arr5[@]}"
	do
		if [[ " ${arr_reg[*]} " =~ " ${task} " ]]; then
		       	if (( $method == 20 )) || (($method == 24)); then
				continue
			fi
		else
		        if (( $method == 22 )) || (($method == 26)); then
                                continue
                        fi
		fi

		for T in {0..4}
		do
			#echo "Launching method $method on task $task with trial $T"
			command="salloc --ntasks 1 --cpus-per-task 40 --job-name ${method}T${T}${task} srun --pty python main.py predict $task $method --RS 0 --T $T --nbagging 100"
			session_name="${task}_M${method}_T${T}"
			tmux_command="tmux new-session -d -s $session_name '$command ; read'"
			echo $tmux_command

			#tmux new-session -d -s $session_name "$command"

			# Break for tasks that don't need 5 trials
			if [[ " ${arr[*]} " =~ " ${task} " ]]; then
				break
			fi
		done
	done
done
