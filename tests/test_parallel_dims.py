"""CPU/gloo-backend tests for `ParallelDims` mesh construction (FSDP-only mesh, all on CPU)."""

from tests.conftest import run_distributed


def _worker_build_fsdp_mesh(rank: int, world_size: int) -> None:
    # `ParallelDims` hardcodes the accelerator type via the module-level `device_type`
    # imported from `src.tools.device_utils`. Since this worker runs in its own spawned
    # interpreter (no CUDA/XPU available in CI), patch it to "cpu" before building the
    # mesh so `init_device_mesh` targets the gloo process group we already initialized.
    import torch.testing._internal.distributed.fake_pg  # noqa: F401  (registers the "fake" backend used for degenerate mesh dims)

    import src.distributed.parallel_dims as parallel_dims_module
    from src.distributed.parallel_dims import ParallelDims

    parallel_dims_module.device_type = "cpu"

    parallel_dims = ParallelDims(
        dp_replicate=1,
        dp_shard=world_size,
        cp=1,
        tp=1,
        pp=1,
        ep=1,
        etp=1,
        world_size=world_size,
    )

    world_mesh = parallel_dims.build_mesh()
    assert world_mesh.size() == world_size

    fsdp_mesh = parallel_dims.get_mesh("fsdp")
    assert fsdp_mesh.size() == world_size
    assert parallel_dims.dp_enabled
    assert parallel_dims.fsdp_enabled
    assert not parallel_dims.tp_enabled
    assert not parallel_dims.pp_enabled
    assert not parallel_dims.ep_enabled
    assert parallel_dims.non_data_parallel_size == 1


def test_parallel_dims_fsdp_only_mesh_over_gloo():
    """A pure FSDP (`dp_shard=world_size`) `ParallelDims` config should build a valid CPU/gloo mesh."""
    run_distributed(_worker_build_fsdp_mesh, world_size=2)


def _worker_build_hsdp_mesh(rank: int, world_size: int) -> None:
    import torch.testing._internal.distributed.fake_pg  # noqa: F401  (registers the "fake" backend used for degenerate mesh dims)

    import src.distributed.parallel_dims as parallel_dims_module
    from src.distributed.parallel_dims import ParallelDims

    parallel_dims_module.device_type = "cpu"

    # world_size=4 split into dp_replicate=2 x dp_shard=2 (HSDP).
    parallel_dims = ParallelDims(
        dp_replicate=2,
        dp_shard=2,
        cp=1,
        tp=1,
        pp=1,
        ep=1,
        etp=1,
        world_size=world_size,
    )
    parallel_dims.build_mesh()

    assert parallel_dims.get_mesh("dp_replicate").size() == 2
    assert parallel_dims.get_mesh("fsdp").size() == 2
    assert parallel_dims.dp_replicate_enabled
    assert parallel_dims.dp_shard_enabled


def test_parallel_dims_hsdp_mesh_over_gloo():
    """A 2x2 HSDP `ParallelDims` config should build valid `dp_replicate`/`fsdp` sub-meshes over CPU/gloo."""
    run_distributed(_worker_build_hsdp_mesh, world_size=4)
