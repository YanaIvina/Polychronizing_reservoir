#!/bin/bash

for MODULE_NAME in triangular_stdp_module diehl_neuron_module
do
	if ! nestml \
		--module_name $MODULE_NAME \
		--input_path ${MODULE_NAME} \
		--target_path ${MODULE_NAME}-target \
		#--codegen_opts=${MODULE_NAME}/nest_code_generator_opts.json
	then
		exit 70
	fi
exit
	
	mkdir -p ${MODULE_NAME}-build
	cd ${MODULE_NAME}-build
	cmake -Dwith-nest=`which nest-config` ../${MODULE_NAME}-target &&
	make &&
	make install &&
	if ! test -n "`grep $MODULE_NAME ~/.nestrc`"
	then
		cat >> ~/.nestrc <<-END_OF_INPUT_TO_CAT

			% Added automatically from `pwd`
			($MODULE_NAME) Install
		END_OF_INPUT_TO_CAT
	fi
done
