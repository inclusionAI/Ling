# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
HAS_FLASH_ATTN = False
HAS_FLASH_ATTN_V2_CUDA = False

try:
    import flash_attn_2_cuda
except ImportError:
    pass
else:
    HAS_FLASH_ATTN_V2_CUDA = True

if not HAS_FLASH_ATTN_V2_CUDA:
    try:
        import flash_attn_cuda
    except ImportError:
        pass
    else:
        HAS_FLASH_ATTN = True


def attention(
        q,
        k,
        v,
        out,
        cu_seqlens,
        max_s,
        softmax_scale,
        window_size_left=-1,
):
    if window_size_left <= 0 and window_size_left != -1:
        raise ValueError("`window_size_left` must be > 0 or -1")

    if HAS_FLASH_ATTN_V2_CUDA:
        return flash_attn_2_cuda.varlen_fwd(
            q, k, v, out, cu_seqlens, cu_seqlens, max_s, max_s, 0.0, softmax_scale,
            False, True, window_size_left, 0, False, None
        )
    elif HAS_FLASH_ATTN:
        if window_size_left != -1:
            raise NotImplementedError(
                "window_size_left is only available with flash attn v2"
            )

        # Flash attention v1 requires q, k and v to have the same number of heads
        if k.shape[1] != q.shape[1]:
            # MQA expand
            if k.shape[1] == 1:
                k = k.expand(-1, q.shape[1], -1)
            # Grouped attention reshape
            else:
                original_shape = k.shape
                k = (
                    k.unsqueeze(2)
                    .expand(-1, -1, q.shape[1] // k.shape[1], -1)
                    .reshape(original_shape[0], -1, original_shape[2])
                )
        if v.shape[1] != q.shape[1]:
            # MQA expand
            if v.shape[1] == 1:
                v = v.expand(-1, q.shape[1], -1)
            # Grouped attention reshape
            else:
                original_shape = v.shape
                v = (
                    v.unsqueeze(2)
                    .expand(-1, -1, q.shape[1] // v.shape[1], -1)
                    .reshape(original_shape[0], -1, original_shape[2])
                )

        return flash_attn_cuda.fwd(
            q, k, v, out, cu_seqlens, cu_seqlens, max_s, max_s, 0.0, softmax_scale, False, True, False, 0, None
        )

    else:
        raise AssertionError
