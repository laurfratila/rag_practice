seed:
\tdocker compose run --rm data_seed

reseed:
\tdocker compose rm -sf data_seed && docker compose run --rm data_seed
