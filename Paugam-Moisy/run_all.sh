U=16.0
conn=10 # здесь задать количество входящих и исходящих связей (стандартно 3)

for signal in AAAA AAAB AABA ABAA BAAA AABB ABAB ABBA BABA BAAB BBAA ABBB BABB BBAB BBBA BBBB
do
    for ex in 1 # на какие нейроны приходит сигнал
    do
        for topology in irreg # случайные или упорядоченные связи
        do
            for tau_m in 3 5 7
            do
                for t_ref in 1
                do
                    for stdp in 20
                    do
                        for delay in 0.1
                        do
                            # создать папку для каждого эксперимента
                            # mkdir $signal-$topology-$ex-tau=$tau_m-t_ref=$t_ref-delay=$delay-stdp=$stdp
                            # mkdir $signal-$topology-$ex-tau=$tau_m-t_ref=$t_ref-delay=$delay-stdp=$stdp/images 

                            # # создать папку для всех draw_weight_graph (если выполняется)
                            # mkdir ALL_GRAPHS-$topology-$ex-tau=$tau_m-t_ref=$t_ref-delay=$delay-stdp=$stdp
                            # # создать папку для всех number_of_weights (если выполняется)
                            # mkdir ALL_WEIGHTS-$topology-$ex-tau=$tau_m-t_ref=$t_ref-delay=$delay-stdp=$stdp
                            # # создать папку для всех number_of_simplexes (если выполняется)
                            # mkdir ALL_SIMPLEXES-$topology-$ex-tau=$tau_m-t_ref=$t_ref-delay=$delay-stdp=$stdp
                            # # создать папку для всех plot_weight_heatmap (если выполняется)
                            # mkdir ALL_HEATMAPS-$topology-$ex-tau=$tau_m-t_ref=$t_ref-delay=$delay-stdp=$stdp
                            # # создать папку для всех plot_weight_hist (если выполняется)
                            # mkdir ALL_HISTS-$topology-$ex-tau=$tau_m-t_ref=$t_ref-delay=$delay-stdp=$stdp

                            # ВЫПОЛНИТЬ СИМУЛЯЦИЮ
                            python train_on_synthetic_data.py $tau_m $t_ref $delay $stdp $U $signal $topology $ex $r $conn

                            # ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
                            # python draw_graph.py $tau_m $t_ref $delay $stdp $U $signal $topology $ex # начальный граф
                            # python number_of_weights.py $tau_m $t_ref $delay $stdp $U $signal $topology $ex # количество весов в определённых диапазонах от эпохи
                            # python plot_weight_heatmap.py $tau_m $t_ref $delay $stdp $U $signal $topology $ex # тепловая карта весов от эпохи
                            # for Th in 0.55 #0.7
                            # do
                            #     python number_of_simplexes.py $tau_m $t_ref $delay $stdp $U $signal $topology $ex $Th # количество симплексов от эпохи (только для conn=3)
                            # done
                            # for epoch in 0 199
                            # do
                            #     python plot_weight_hist.py $tau_m $t_ref $delay $stdp $U $signal $topology $ex $epoch # гистограмма весов на определённой эпохе
                            #     for Th in 0.7
                            #     do
                            #         python draw_weight_graph.py $tau_m $t_ref $delay $stdp $U $signal $topology $ex $Th $epoch # граф со связями, веса которых выросли выше порога                                       
                            #     done
                            done
                        done
                    done
                done
            done
        done
    done
done