1. В **run_all** установить параметры, при которых будут выполняться программы:
	- *signal* - сигнал, для которого проводится симуляция (комбинации из A,B,C,D)
	- *ex* - номер, определяющий, на какие нейроны подаётся сигнал (нейроны для данного номера прописать в в signal_parameters)
	- *topology* - топология reg для упорядоченный связей нейронов по часовой стрелке, irreg для случайных связей
	- *tau_m* - период реласксации
	- *stdp*
	- *delay* 
	- *Th* - порог веса
	- *epoch* - эпоха
2. В run_all оставить те программы, которые необходимо запустить. Остальные закомментировать:
	- train_on_synthetic_data.py запустить симмулятор
	- draw_graph.py начертить граф сети (нейроны и связи без весов)
	- number_of_weights.py начертить график зависимости количества весов в определённых диапазонах (диапазоны установить в самом коде программы) в зависимости от эпохи
	- plot_weight_heatmap.py начертить тепловую карту весов от эпохи
	- number_of_simplexes.py начертить изменение количества двумерных и трёхмерных симплексов (только в случае, когда у каждого нейрона три исходящие и три входящие связи, иначе надо поменять расчёт симплексов в ходе программы) от эпохи
	- plot_weight_hist.py - начертить гистограмму весов на определённой эпохе
	- draw_weight_graph.py - начертить граф только с теми связями, которые выросли выше порога на заданной эпохе
Также можно раскомментировать создание отдельных папок, где будут лежать соответствующие графики для всех сигналов. Необходимо так же в самим программах раскомментировать сохранение графиков в эти отдельные папки
3. Запустить
	source ~/.bashrc
	
 	conda activate /s/ls4/groups/g0126/conda_envs/nest
	
 	module load git
	
 	module load openmpi
	
 	cd [...]/Paugam-Moisy
	
 	sh run_all.sh
