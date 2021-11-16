build:
	docker build -t kwang2049/faiss-instant .

build-gpu:
	docker build -t kwang2049/faiss-instant-gpu -f Dockerfile.gpu .

pull:
	docker pull kwang2049/faiss-instant

release:
	docker push kwang2049/faiss-instant

download:
	wget https://public.ukp.informatik.tu-darmstadt.de/kwang/faiss-instant/resources/ids.txt -P ./resources
	wget https://public.ukp.informatik.tu-darmstadt.de/kwang/faiss-instant/resources/pq-384-8b.index -P ./resources

run:
	docker run --detach --rm -it -p 5001:5000 -v ${PWD}/resources:/opt/faiss-instant/resources --name faiss-instant kwang2049/faiss-instant

run-gpu:
	docker run --runtime=nvidia --detach --rm -it -p 5001:5000 -v ${PWD}/resources:/opt/faiss-instant/resources --name faiss-instant-gpu kwang2049/faiss-instant-gpu

remove:
	docker rm -f faiss-instant
	docker image rm kwang2049/faiss-instant

query:
	bash query_example.sh

reload:
	bash reload_example.sh

reload-gpu:
	bash reload_example-gpu.sh

index-list:
	curl -X GET 'http://localhost:5001/index_list'
