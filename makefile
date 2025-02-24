DATA_DIR = data

all:
	python main.py -B

data: $(DATA_DIR)/car.csv

$(DATA_DIR)/car.csv:
	curl -o $(DATA_DIR)/car.csv https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/all-vehicles-model/exports/csv?lang=en&timezone=America%2FChicago&use_labels=true&delimiter=%3B


clean:
	rm $(DATA_DIR)/car.csv


.PHONY: data clean