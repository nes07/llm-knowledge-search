MEMGRAPH_COMPOSE_PATH=../memgraph-docker/docker-compose.yml

start-memgraph:
	docker compose -f $(MEMGRAPH_COMPOSE_PATH) up -d

stop-memgraph:
	docker compose -f $(MEMGRAPH_COMPOSE_PATH) down

logs-memgraph:
	docker compose -f $(MEMGRAPH_COMPOSE_PATH) logs -f memgraph