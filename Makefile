.PHONY: install train run test docker

install:
	pip install -r requirements.txt

train:
	python scripts/train_model.py

run:
	streamlit run app/main.py

test:
	pytest -q

docker:
	docker compose -f docker/docker-compose.yml up --build
