def final_layer_query_ranges(seqs: list) -> list[tuple[int, int]] | None:
    """Find packed Dream query spans consumed by log optimization level 2.

    Full/Refresh inputs contain the entire sequence; Reuse inputs contain the
    current block. Dream's shifted prediction needs the preceding hidden row.
    Keep a 128-row prefix halo, aligned to 128 rows within each request, to
    avoid very small query tiles for long-context Refresh. Metadata is computed
    from host sequence state before forward/capture, without reading GPU scalars.
    """
    ranges = []
    offset = 0
    trimmed = False
    for seq in seqs:
        state = seq.sparse_state
        if state is None:
            block_start = seq.num_prompt_tokens
            block_len = seq.num_tokens - block_start
        else:
            block_start, block_len = state.compute_block_boundaries(
                seq.num_prompt_tokens, seq.num_tokens
            )
        if block_len <= 0:
            return None
        if state is not None and state.should_reuse_cache:
            length = block_len
            start, end = 0, length
        else:
            length = seq.num_tokens
            start = max(0, block_start - 128) // 128 * 128
            end = block_start + block_len
        ranges.append((offset + start, offset + end))
        trimmed |= start > 0 or end < length
        offset += length
    return ranges if trimmed else None
