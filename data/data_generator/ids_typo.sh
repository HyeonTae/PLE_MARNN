mkdir ../network_inputs/iitk-ids_typo-1189
cat ../network_inputs/iitk-ids-1189/data_train_edit.txt ../network_inputs/iitk-typo-1189/data_train_edit.txt > ../network_inputs/iitk-ids_typo-1189/data_train_edit.txt
cat ../network_inputs/iitk-ids-1189/data_val_edit.txt ../network_inputs/iitk-typo-1189/data_val_edit.txt > ../network_inputs/iitk-ids_typo-1189/data_val_edit.txt
shuf -o ../network_inputs/iitk-ids_typo-1189/data_train_edit.txt ../network_inputs/iitk-ids_typo-1189/data_train_edit.txt
shuf -o ../network_inputs/iitk-ids_typo-1189/data_val_edit.txt ../network_inputs/iitk-ids_typo-1189/data_val_edit.txt
