f"""SELECT * FROM {distance_fn}(
    '{embeddings_str}'::vector({VECTOR_EMBEDDINGS_COUNT}),
    {float(distance_threshold)}::double precision,
    {int(k)});"""