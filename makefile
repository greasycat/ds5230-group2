DATA_DIR = data

all:
	python main.py -B

cache:
	python main.py -B --cached


data: $(DATA_DIR)/car.csv $(DATA_DIR)/customer.csv
	

$(DATA_DIR)/car.csv:
	mkdir -p $(DATA_DIR)
	curl -o $(DATA_DIR)/car.csv https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/all-vehicles-model/exports/csv?lang=en&timezone=America%2FChicago&use_labels=false&delimiter=%3B

$(DATA_DIR)/customer.csv:
	mkdir -p $(DATA_DIR)
	curl -L -o $(DATA_DIR)/customer.zip https://www.kaggle.com/api/v1/datasets/download/vjchoudhary7/customer-segmentation-tutorial-in-python
	unzip $(DATA_DIR)/customer.zip -d $(DATA_DIR)
	mv $(DATA_DIR)/Mall_Customers.csv $(DATA_DIR)/customer.csv
	rm $(DATA_DIR)/customer.zip
clean:
	rm $(DATA_DIR)/car.csv
	rm $(DATA_DIR)/customer.csv


.PHONY: data clean