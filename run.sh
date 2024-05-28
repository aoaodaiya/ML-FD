T_mins=(25 50 75 100 125 150 175 200 250 300)
FFT_nums=(25 50 75 100 125 150 175 200 250 300 400 500)
alphas=(0.1 0.3 0.5 0.7 0.8 0.9 1.0)

for w in "${T_mins[@]}"; do
    for t in "${FFT_nums[@]}"; do
        for r in "${alphas[@]}"; do
            echo "$w-$t-$r"
            python main.py --T_min "$w" --FFT_num "$t" --alpha "$r" --name "test_$w_$t_$r" >> "result_$w_$t_$r.txt"
        done
    done
done
